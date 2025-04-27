import os
import json
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- model list ----------
model_list = [
    'google/gemma-3-1b-it',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'meta-llama/Llama-3.2-1B-Instruct',
]

# ---------- warm-up -------------
warm_prompts = [
    "Hello, how are you?",
    "Summarise the plot of Hamlet in one sentence.",
    "Write a haiku about benchmark scripts."
]

# --------- prompt list ----------
prompt_list = [
    
]

# ------ input-output dict --------
input_output_dict = {
    "google/gemma-3-1b-it": {},
    "Qwen/Qwen2.5-1.5B-Instruct": {},
    "meta-llama/Llama-3.2-1B-Instruct": {}
}

def benchmark(model_id:str):
    # 1. Authorize with Hugging Face Hub
    try:
        login(token=os.environ.get("HF_TOKEN"))
        print("Logged in to Hugging Face Hub successfully.")
    except Exception as e:
        print(f"Error during login: {e}")
        return
    
    # 2. Load the model and tokenizer
    try:
        print(f"Loading model & tokenizer {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        print("Model & tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model & tokenizer: {e}")
        return
    
    # 3. Run warmup inference
    try:
        print("Running warmup inference...")
        for prompt in warm_prompts:                 # 3× warm-up
            ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            _ = model.generate(**ids, max_new_tokens=16, do_sample=False)
        print("Warmup inference completed.")
    except Exception as e:
        print(f"Error during warmup inference: {e}")
        return
    
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    
    # 4. Run benchmark inference
    try:
        print("Running benchmark inference...")
        for prompt in prompt_list:
            ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model.generate(**ids, max_new_tokens=16, do_sample=False)
            
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Output: {decoded_output}")
            
            # Store input-output pairs in the dictionary
            input_output_dict[model_id] = {
                "input": prompt,
                "output": decoded_output,
                "time": ""
            }
        print("Benchmark inference completed.")
    except Exception as e:
        print(f"Error during benchmark inference: {e}")
        return
    
    # 5. Return the input-output dictionary
    return input_output_dict
    
def run_benchmark():
    for model_id in model_list:
        input_output_dict = benchmark(model_id)
    
    # Save the input-output dictionary to a JSON file
    with open("input_output.json", "w") as f:
        json.dump(input_output_dict, f, indent=4)
    print("Input-output pairs saved to input_output.json")