import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Load the dataset
dataset = load_dataset("OpenAssistant/oasst1")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/phi-2")

# Print model's module names to identify the correct target modules
for name, module in model.named_modules():
    print(name)

# Apply QLORA configuration with correct target modules
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, lf.key", "transformer.h.0.attention.self.value"]
    r=8, 
model = get_peft_model(model, lora_config)
    lora_dropout=0.1, 
# Tokenize the dataset..correct module names..."]
)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True) 

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the training argumentsext"], padding="max_length", truncation=True)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
