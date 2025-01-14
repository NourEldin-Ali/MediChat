from unsloth import FastLanguageModel

def load_model(model_name, # More models at https://huggingface.co/unsloth
                max_seq_length=2048, # Choose any! We auto support RoPE Scaling internally!
                dtype=None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
                load_in_4bit=True, # Use 4bit quantization to reduce memory usage. Can be False.
                ):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length, 
        dtype = dtype, 
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    model = FastLanguageModel.get_peft_model(
      model,
      r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
      lora_alpha = 16,
      lora_dropout = 0, # Supports any, but = 0 is optimized
      bias = "none",    # Supports any, but = "none" is optimized
      # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
      use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
      random_state = 3407,
      use_rslora = False,  # We support rank stabilized LoRA
      loftq_config = None, # And LoftQ
    )
    return model, tokenizer

def get_response(model, tokenizer, prompt):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
If you are a doctor, please answer the medical questions based on the patient's description.

### Input:
{}

### Response:
{}"""

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            prompt, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, temperature=0.01,use_cache = True)
    responce = tokenizer.batch_decode(outputs)
    # filter out the prompt to get only from RESPONSE to end of token
    final_result = responce[0].split("### Response:")[1].strip()
    if(final_result.count("###") > 1):
        final_result = final_result.split("###")[0].strip()
    return final_result

if __name__ == "__main__":
    model_name = "unsloth/Meta-Llama-3.1-8B"
    model, tokenizer = load_model(model_name)
    print("Model loaded successfully!")
