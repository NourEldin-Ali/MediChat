# Force CPU at the very beginning
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["USE_CUDA"] = "0"

import torch
torch.set_num_threads(4)  # Control CPU threads for better performance

import mlflow
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# Connect to MLflow server
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("llama2-finetune-medical-cpu")

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

# Select only 1000 samples for testing
print("Subsetting dataset to 1000 samples...")
dataset["train"] = dataset["train"].select(range(1000))
if "validation" in dataset:
    dataset["validation"] = dataset["validation"].select(range(min(100, len(dataset["validation"]))))

print(f"Training dataset size: {len(dataset['train'])}")

# Use a very small model for CPU training
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading tokenizer from {model_id}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format the data
def format_prompt(question, answer=None):
    prompt = f"### Patient: {question}\n### Doctor:"
    if answer:
        return prompt + f" {answer}"
    return prompt

def preprocess_function(examples):
    inputs = [format_prompt(q) for q in examples["input"]]
    model_inputs = tokenizer(inputs, truncation=True, max_length=128, padding="max_length")  # Further reduced context length
    
    # Create labels
    labels = tokenizer(
        [format_prompt(q, a) for q, a in zip(examples["input"], examples["output"])],
        truncation=True,
        max_length=128,  # Further reduced context length
        padding="max_length"
    )["input_ids"]
    
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Load model explicitly on CPU
print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,  # Enable memory optimization
    torch_dtype=torch.float32  # Use full precision on CPU
)

# Explicitly move model to CPU
model = model.cpu()

# Check device
for param in model.parameters():
    if param.device.type != 'cpu':
        print(f"WARNING: Parameter on device {param.device} instead of CPU!")
        param.data = param.data.cpu()

print(f"Model device: {next(model.parameters()).device}")

# Configure LoRA with very small parameters for CPU
target_modules = ["q_proj", "v_proj"]  # Common target modules for LLaMA models
lora_config = LoraConfig(
    r=2,               # Very small rank
    lora_alpha=4,      # Very small alpha
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
print("Applying LoRA adapters...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8,
)

# Training arguments optimized for CPU
print("Setting up training arguments for CPU...")
training_args = TrainingArguments(
    output_dir="/models/medical-chat-cpu",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,            # Even lower learning rate
    num_train_epochs=1,
    logging_dir="/models/logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,                # Save less frequently
    report_to="mlflow",
    fp16=False,                    # Disable mixed precision
    bf16=False,                    # Disable bf16
    gradient_checkpointing=False,  # Disable for CPU
    optim="adamw_torch",
    no_cuda=True,                  # Explicitly disable CUDA
    dataloader_num_workers=0,      # Disable multiprocessing
    local_rank=-1,                 # Disable distributed training
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
    data_collator=data_collator,
)

# Verify again model is on CPU
print(f"Model device before training: {next(model.parameters()).device}")

# Train model
print("Starting CPU training...")
with mlflow.start_run() as run:
    mlflow.log_params({
        "model_id": model_id,
        "dataset_size": len(dataset["train"]),
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        "device": "cpu"
    })
    trainer.train()

# Save model
print("Saving model...")
model.save_pretrained("/models/medical-chat-cpu-final")
tokenizer.save_pretrained("/models/medical-chat-cpu-final")
print("Training completed and model saved!")