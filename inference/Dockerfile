FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install dependencies
RUN pip install transformers==4.35.2 datasets peft==0.5.0 bitsandbytes==0.41.1 accelerate==0.23.0 sentencepiece flask scipy

# Set working directory
WORKDIR /app

# Copy inference script
COPY serve.py .

# Expose port for API
EXPOSE 8000

# Start inference server
CMD ["python", "serve.py"]