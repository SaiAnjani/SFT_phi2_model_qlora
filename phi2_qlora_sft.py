import pkg_resources
import sys

required_packages = {
    'transformers': '4.34.0',
    'datasets': '2.14.0',
    'torch': '2.0.0',
    'bitsandbytes': '0.42.0',  # Updated version
    'peft': '0.5.0',
    'accelerate': '0.23.0',
    'scipy': '1.9.0'
}

def check_packages():
    missing = []
    for package, min_version in required_packages.items():
        try:
            pkg_resources.require(f"{package}>={min_version}")
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    if missing:
        print("Please install required packages using:")
        print("pip install -r requirements.txt")
        sys.exit(1)

check_packages()

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

# Configuration
MODEL_NAME = "microsoft/phi-2"
DATASET_NAME = "OpenAssistant/oasst1"
OUTPUT_DIR = "./phi2-qlora-sft"
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Load dataset and create train/validation split
dataset = load_dataset(DATASET_NAME)
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
print(f"Train samples: {len(dataset['train'])}, Validation samples: {len(dataset['test'])}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_conversation(example):
    if example["role"] == "assistant":
        return f"Assistant: {example['text']}\n\n"
    else:
        return f"Human: {example['text']}\n"

def preprocess_function(examples):
    conversations = []
    current_conversation = ""
    
    for role, text in zip(examples["role"], examples["text"]):
        if role == "human" and current_conversation:
            conversations.append(current_conversation.strip())
            current_conversation = format_conversation({"role": role, "text": text})
        else:
            current_conversation += format_conversation({"role": role, "text": text})
    
    if current_conversation:
        conversations.append(current_conversation.strip())
    
    tokenized = tokenizer(
        conversations,
        truncation=True,
        max_length=2048,
        padding=True,
        return_tensors="pt"
    )
    
    return tokenized

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Print model architecture to identify correct layer names
print("Model architecture:")
for name, module in model.named_modules():
    print(name)

# Add more detailed model inspection before LoRA config
print("\nDetailed model layer inspection:")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"Linear layer found: {name} -> {module}")

# Configure LoRA with correct target modules for phi-2
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "mlp.dense_h_to_4h",
        "mlp.dense_4h_to_h",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.dense"
    ],  # Updated target modules based on phi-2's architecture
)

# Get PEFT model
model = get_peft_model(model, lora_config)

# Preprocess dataset
processed_dataset = {}
for split in ['train', 'test']:
    processed_dataset[split] = dataset[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset[split].column_names,
    )

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="steps",        # Explicitly set save strategy
    evaluation_strategy="steps",  # Add evaluation strategy
    save_steps=100,
    logging_steps=10,
    eval_steps=100,
    eval_accumulation_steps=1,
    load_best_model_at_end=True,
    warmup_ratio=0.05,
    weight_decay=0.01,
    metric_for_best_model="eval_loss", # Updated metric name
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],  # This is now our validation set
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training
trainer.train()

# Save complete model artifacts
def save_complete_model(trainer, output_dir):
    """Save all required model files"""
    print("Saving model files...")
    
    # 1. Save the PEFT/LoRA model and training args
    trainer.save_model()
    trainer.save_state()  # This saves training_args.bin and other training state
    
    # 2. Save training arguments as json (correct method)
    import json
    training_args_dict = trainer.args.to_dict()
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(training_args_dict, f, indent=2)
    
    # 3. Save the merged model weights separately as pytorch_model.bin
    merged_model = trainer.model.merge_and_unload()
    torch.save(merged_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # 4. Save the base model configuration
    trainer.model.config.save_pretrained(output_dir)
    
    # 5. Save all tokenizer files
    tokenizer.save_pretrained(
        output_dir,
        save_format='all'  # Save all formats (including special tokens, vocab, etc.)
    )
    
    # Print saved files for verification
    print("\nSaved files in output directory:")
    for file in os.listdir(output_dir):
        print(f"- {file}")

# After training, save all files
save_complete_model(trainer, OUTPUT_DIR)

print("\nVerifying required files:")
required_files = [
    "config.json",
    "pytorch_model.bin",  # Full model weights
    "training_args.bin",  # Training configuration
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "adapter_model.bin",  # LoRA weights
    "adapter_config.json"  # LoRA config
]

for file in required_files:
    path = os.path.join(OUTPUT_DIR, file)
    status = "✓" if os.path.exists(path) else "✗"
    print(f"{status} {file}")
