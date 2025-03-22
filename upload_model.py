import os
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
MODEL_PATH = "./phi2-qlora-sft"  # Path to your saved model
HF_TOKEN = ""  # Your Hugging Face token
REPO_NAME = "your-username/phi2-oasst-sft"  # Replace with your desired repository name
MODEL_NAME = "microsoft/phi-2"

# Login to Hugging Face
login(token=HF_TOKEN)

# Load the original model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

# Create repository and upload
api = HfApi()
api.create_repo(repo_id=REPO_NAME, private=False, exist_ok=True)

# Save and push the model
model.push_to_hub(REPO_NAME, use_temp_dir=True)
tokenizer.push_to_hub(REPO_NAME, use_temp_dir=True)

print(f"Model uploaded successfully to: https://huggingface.co/{REPO_NAME}")
