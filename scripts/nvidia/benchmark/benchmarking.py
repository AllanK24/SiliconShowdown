import os
import json
import torch
import time # For overall timing
# pip install psutil nvidia-ml-py3

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ---------- Model List ----------
model_list = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    'google/gemma-3-1b-it',
    'meta-llama/Llama-3.2-1B-Instruct',
]

# ---------- Warm-up Prompts ----------
warm_prompts = [
    "Hello, how are you today?",
    "What is the capital of France?",
    "Write a short poem about clouds."
]

# ---------- Benchmark Prompts ----------
prompt_list = [
    "Translate the following sentence to German: 'The weather is beautiful today.'",
    "Explain the concept of quantum entanglement in simple terms.",
    "Write a python function that calculates the factorial of a number.",
    "Summarize the main plot points of the movie 'Inception'.",
    "What are the main differences between renewable and non-renewable energy sources?",
    "Create a short story about a robot discovering music.",
    "List three advantages and three disadvantages of remote work.",
    "Explain the importance of biodiversity.",
]

# ---------- Generation Config ----------
MAX_NEW_TOKENS = 512
generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
)

# ---------- Benchmark Function (MPS or CPU Specific) ----------
def benchmark_model_on_prompt_mps(model, tokenizer, prompt, dtype, num_runs=3):
    """Runs benchmark for a single prompt on MPS/CPU, returns metrics."""
    results = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]

        # --- Timed Runs ---
        times_ms = []
        output_tokens = 0
        generated_text = ""

        for i in range(num_runs):
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)

            end_time = time.time()
            iter_time_ms = (end_time - start_time) * 1000
            times_ms.append(iter_time_ms)

            # Decode output only once
            if i == 0:
                 # Decode only new tokens - check indexing carefully
                 # Some models might repeat input tokens, some might not in outputs
                 # A safer way might be len(outputs[0]) - len(inputs.input_ids[0])
                 actual_output_ids = outputs[0][inputs.input_ids.shape[1]:]
                 generated_text = tokenizer.decode(actual_output_ids, skip_special_tokens=True)
                 output_tokens = len(actual_output_ids)

        # Peak memory measurement is not straightforward on MPS; omit or use allocated memory if desired
        peak_memory_mb = None

        # --- Aggregate Results ---
        avg_time_ms = sum(times_ms) / len(times_ms) if times_ms else 0
        tokens_per_sec = (output_tokens / (avg_time_ms / 1000.0)) if avg_time_ms > 0 else 0

        results = {
            "prompt": prompt,
            "status": "success",
            "error_message": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens, # Actual generated tokens
            "avg_time_ms": round(avg_time_ms, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "runs_time_ms": [round(t, 3) for t in times_ms],
            "peak_memory_mb": peak_memory_mb,
            "output_text_preview": generated_text[:100] + "..."
        }

    except Exception as e:
        print(f"ERROR during benchmark for prompt: '{prompt[:50]}...' - {e}")
        results = {
            "prompt": prompt,
            "status": "failed",
            "error_message": str(e),
        }
    return results

# ---------- Main Benchmark Runner (MPS or CPU Specific) ----------
def run_full_benchmark_mps(output_filename="benchmark_results_mps.json"):
    """Runs benchmarks for all models and prompts on MPS/CPU, saving results."""

    all_results = []
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Choose dtype here; MPS generally supports float32 best
    benchmark_dtype = torch.float16
    print(f"--- Running Benchmark on device: {device} with dtype: {benchmark_dtype} ---")

    # --- Hugging Face Login (Do once) ---
    try:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
            print("Logged in to Hugging Face Hub successfully.")
        else:
            print("HF_TOKEN environment variable not set. Skipping login.")
            print("Ensure models are cached locally or accessible without login.")
    except Exception as e:
        print(f"Warning: Hugging Face login failed or skipped: {e}")

    # --- Loop through models ---
    for model_id in model_list:
        print(f"\n{'='*20} Benchmarking Model: {model_id} {'='*20}")

        model = None
        tokenizer = None
        model_load_time = None
        try:
            # --- Load Model & Tokenizer ---
            print(f"Loading tokenizer {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            
            print(f"Loading model {model_id} to CPU first (dtype: {benchmark_dtype})...")
            load_start = time.time()
            # Load to CPU first is still generally safer for memory management
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=benchmark_dtype,
                low_cpu_mem_usage=True
            )
            print(f"Model loaded to CPU. Moving to {device}...")
            model.to(device)
            model.eval() # Set to evaluation mode
            load_end = time.time()
            model_load_time = load_end - load_start
            print(f"Model ready on {device} in {model_load_time:.2f} seconds.")

            # --- Global Warm-up (Once per model) ---
            print("Running global warm-up...")
            for w_prompt in warm_prompts:
                w_inputs = tokenizer(w_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    # Use a smaller max_new_tokens for warmup if desired
                    _ = model.generate(**w_inputs, max_new_tokens=16, do_sample=False)
            print("Global warm-up complete.")

            # --- Benchmark each prompt ---
            for prompt_text in prompt_list:
                print(f"--- Prompt: '{prompt_text[:50]}...' ---")
                prompt_results = benchmark_model_on_prompt_mps(
                    model, tokenizer, prompt_text, benchmark_dtype, num_runs=3
                )

                # Add model/device info to prompt results
                prompt_results["model_id"] = model_id
                prompt_results["device"] = device
                prompt_results["dtype"] = str(benchmark_dtype)
                prompt_results["model_load_time_s"] = round(model_load_time, 2)

                all_results.append(prompt_results)

                # Save intermediate results frequently
                with open(output_filename, "w") as f:
                    json.dump(all_results, f, indent=4)

        except Exception as e:
            print(f"FATAL ERROR during processing for model {model_id}: {e}")
            # Log the failure
            all_results.append({
                "model_id": model_id,
                "status": "load_or_setup_failed",
                "error_message": str(e),
                "device": device,
                "dtype": str(benchmark_dtype)
            })
        finally:
            # --- Cleanup model/tokenizer ---
            print(f"Cleaning up resources for {model_id}...")
            del model
            del tokenizer
            if device == "mps":
                torch.mps.empty_cache()
            print("Cleanup complete.")
            # Save final results again after cleanup
            with open(output_filename, "w") as f:
                 json.dump(all_results, f, indent=4)


    print(f"\nBenchmark run complete. All results saved to {output_filename}")


# --- Run the Benchmark ---
if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Make filename explicitly MPS/CPU
    output_file = f"benchmark_results_mps_{timestamp}.json"
    run_full_benchmark_mps(output_filename=output_file)