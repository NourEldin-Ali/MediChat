from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)

# Load the model and tokenizer
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "/models/llama2-medical-final"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with 8-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_8bit=True,
    device_map="auto",
)

# Load the adapter weights
model = PeftModel.from_pretrained(base_model, adapter_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    question = data.get("question", "")
    
    # Format prompt
    prompt = f"### Patient: {question}\n### Doctor:"
    
    # Generate response
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the doctor's response
    doctor_response = response.split("### Doctor:")[1].strip()
    
    return jsonify({"response": doctor_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)