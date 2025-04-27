import os
import json
import torch
import time # For overall timing
import psutil # For general system info (optional)
# pip install psutil nvidia-ml-py3
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    print("pynvml initialized successfully for Nvidia GPU monitoring.")
except Exception as e:
    NVML_AVAILABLE = False
    print(f"pynvml initialization failed: {e}. Nvidia GPU temp/power monitoring disabled.")

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ---------- Model List ----------
model_list = [
    'google/gemma-1.1-2b-it',
    'Qwen/Qwen1.5-1.8B-Chat',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
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
MAX_NEW_TOKENS = 128
generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    pad_token_id=50256, # Adjust if needed based on tokenizer below
)

# ---------- NVML Helpers (Nvidia Specific) ----------
def get_nvidia_gpu_details(device_id=0):
    """Gets temperature, memory usage, and power for a specific Nvidia GPU."""
    if not NVML_AVAILABLE:
        return None, None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Getting specific allocated memory is better done with torch.cuda
        # mem_used_mb = mem_info.used / (1024**2) 
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # Watts
        return temp, power
    except pynvml.NVMLError as error:
        print(f"Failed to get NVML details: {error}")
        return None, None
    except Exception as e:
        print(f"Unexpected error getting NVML details: {e}")
        return None, None


# ---------- Benchmark Function (CUDA Specific) ----------
def benchmark_model_on_prompt_cuda(model, tokenizer, prompt, dtype, num_runs=3):
    """Runs benchmark for a single prompt on CUDA, returns metrics."""
    results = {}
    device = "cuda" # Hardcoded for this script
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]

        # --- Timed Runs ---
        gpu_times_ms = []
        output_tokens = 0
        generated_text = ""

        # Reset CUDA memory stats
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_mem_allocated = torch.cuda.memory_allocated(device)

        # Measure Temp/Power before run
        temp_before, power_before = get_nvidia_gpu_details()

        for i in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize(device) # Ensure previous work is done
            start_event.record()

            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)

            end_event.record()
            torch.cuda.synchronize(device) # Wait for kernel to finish
            iter_time_ms = start_event.elapsed_time(end_event)
            gpu_times_ms.append(iter_time_ms)

            # Decode output only once
            if i == 0:
                 # Decode only new tokens - check indexing carefully
                 # Some models might repeat input tokens, some might not in `outputs`
                 # A safer way might be len(outputs[0]) - len(inputs.input_ids[0])
                 actual_output_ids = outputs[0][inputs.input_ids.shape[1]:]
                 generated_text = tokenizer.decode(actual_output_ids, skip_special_tokens=True)
                 output_tokens = len(actual_output_ids)

        # Measure Temp/Power after run
        temp_after, power_after = get_nvidia_gpu_details()
        # Measure peak memory *during* the generation runs
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        # You could also measure end memory allocated if interested
        # end_mem_allocated = torch.cuda.memory_allocated(device)

        # --- Aggregate Results ---
        avg_time_ms = sum(gpu_times_ms) / len(gpu_times_ms) if gpu_times_ms else 0
        tokens_per_sec = (output_tokens / (avg_time_ms / 1000.0)) if avg_time_ms > 0 else 0

        results = {
            "prompt": prompt,
            "status": "success",
            "error_message": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens, # Actual generated tokens
            "avg_gpu_time_ms": round(avg_time_ms, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "runs_gpu_time_ms": [round(t, 3) for t in gpu_times_ms],
            "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
            "temp_before_c": temp_before,
            "temp_after_c": temp_after,
            "power_before_w": power_before,
            "power_after_w": power_after,
            "output_text_preview": generated_text[:100] + "..."
        }

    except Exception as e:
        print(f"ERROR during benchmark for prompt: '{prompt[:50]}...' - {e}")
        # Attempt to get current memory/temp even on error if possible
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else None
        temp_err, power_err = get_nvidia_gpu_details()
        results = {
            "prompt": prompt,
            "status": "failed",
            "error_message": str(e),
            "peak_memory_mb_on_error": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
            "temp_on_error": temp_err,
            "power_on_error": power_err,
        }
    return results

# ---------- Main Benchmark Runner (CUDA Specific) ----------
def run_full_benchmark_cuda(output_filename="benchmark_results_cuda.json"):
    """Runs benchmarks for all models and prompts on CUDA, saving results."""

    all_results = []
    device = "cuda" # Hardcoded
    # Choose dtype here
    benchmark_dtype = torch.float16 # Or torch.bfloat16 / torch.float32
    print(f"--- Running CUDA Benchmark with dtype: {benchmark_dtype} ---")

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
            if tokenizer.pad_token is None:
                 print("Warning: Tokenizer missing pad token, setting to eos_token.")
                 tokenizer.pad_token = tokenizer.eos_token
                 # Ensure generation config also uses this if it relies on pad_token_id
                 if generation_config.pad_token_id is None:
                     generation_config.pad_token_id = tokenizer.eos_token_id

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
            torch.cuda.synchronize(device) # Wait for warmup to finish
            print("Global warm-up complete.")

            # --- Benchmark each prompt ---
            for prompt_text in prompt_list:
                print(f"--- Prompt: '{prompt_text[:50]}...' ---")
                prompt_results = benchmark_model_on_prompt_cuda(
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
            torch.cuda.empty_cache() # Crucial for CUDA
            print("Cleanup complete.")
            # Save final results again after cleanup
            with open(output_filename, "w") as f:
                 json.dump(all_results, f, indent=4)


    print(f"\nCUDA Benchmark run complete. All results saved to {output_filename}")
    if NVML_AVAILABLE:
         pynvml.nvmlShutdown()


# --- Run the Benchmark ---
if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Make filename explicitly CUDA
    output_file = f"benchmark_results_cuda_{timestamp}.json"
    run_full_benchmark_cuda(output_filename=output_file)