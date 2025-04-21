FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install git and other dependencies
RUN apt-get update && apt-get install -y git wget

# Install compatible versions of packages
RUN pip install transformers==4.35.2 datasets peft==0.5.0 bitsandbytes==0.41.1 accelerate==0.23.0 mlflow tqdm sentencepiece

# Set working directory
WORKDIR /app

# Copy fine-tuning script
COPY finetune.py .

# Set entrypoint
CMD ["python", "finetune.py"]