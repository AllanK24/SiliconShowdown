import os
import time
import yaml
import json
import torch
import statistics
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

# Load config from YAML
with open("scripts/nvidia/benchmark_py/configs/config_cuda.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------- Model List (LLM Only) ----------
model_list = config["model_list"]

# ---------- Warm-up Prompts ----------
warm_prompts = config["warm_prompts"]
NUM_GLOBAL_WARMUP_RUNS = len(warm_prompts)

# ---------- Benchmark Prompts ----------
prompt_list = config["prompt_list"]
NUM_TIMED_RUNS_PER_PROMPT = config["num_timed_runs_per_prompt"] # Number of repetitions for timing

# ---------- Generation Config ----------
generation_config = GenerationConfig(
    **config["generation_config"],
)
BATCH_SIZE = config["batch_size"] # Fixed batch size

def save_generation_config(generation_config: GenerationConfig, filename="generation_config.json"):
    """Saves the generation config to a JSON file."""
    with open(filename, "w") as f:
        json.dump(generation_config.to_dict(), f, indent=4)

# ---------- NVML Helpers (Nvidia Specific) ----------
def get_nvidia_gpu_details(device_id=0):
    if not NVML_AVAILABLE: return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # Watts
        return temp, power
    except pynvml.NVMLError as error:
        print(f"Warning: Failed to get NVML details: {error}")
        return None, None
    except Exception as e:
        print(f"Warning: Unexpected error getting NVML details: {e}")
        return None, None

# ---------- Benchmark Function (CUDA Specific - Per Prompt) ----------
def benchmark_model_on_prompt_cuda(model, tokenizer, prompt, generation_config_obj, num_runs=3):
    """Runs benchmark for a single prompt on CUDA, returns metrics dictionary."""
    results = {}
    device = "cuda"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]
        
        # --- TTFT Runs ---
        ttft_runs = []
        for _ in range(num_runs):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            with torch.inference_mode():
                _ = model.generate(**inputs,
                                max_new_tokens=1,
                                do_sample=False)
            end_evt.record()
            torch.cuda.synchronize()               # wait so elapsed_time is valid
            ttft_runs.append(start_evt.elapsed_time(end_evt))  # ms
        ttft_ms_avg = round(statistics.mean(ttft_runs), 2)

        # --- Timed Runs ---
        gpu_times_ms = []
        output_tokens = 0
        generated_text = ""

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device) # Reset before the runs for this prompt

        temp_before, power_before = get_nvidia_gpu_details()

        for i in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_event.record()

            with torch.inference_mode():
                # Ensure generation_config is passed correctly
                outputs = model.generate(**inputs, generation_config=generation_config_obj)

            end_event.record()
            torch.cuda.synchronize(device)
            iter_time_ms = start_event.elapsed_time(end_event)
            gpu_times_ms.append(iter_time_ms)

            if i == 0: # Decode only once
                 actual_output_ids = outputs[0][inputs.input_ids.shape[1]:]
                 generated_text = tokenizer.decode(actual_output_ids, skip_special_tokens=True)
                 output_tokens = len(actual_output_ids) # Use actual generated length

        temp_after, power_after = get_nvidia_gpu_details()
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        # --- Calculate Aggregated Metrics ---
        avg_time_ms = statistics.mean(gpu_times_ms) if gpu_times_ms else 0
        stddev_time_ms = statistics.stdev(gpu_times_ms) if len(gpu_times_ms) > 1 else 0
        tokens_per_sec = (output_tokens / (avg_time_ms / 1000.0)) if avg_time_ms > 0 else 0

        # Calculate temp/power stats (handle None values)
        avg_temp_c = (temp_before + temp_after) / 2 if temp_before is not None and temp_after is not None else None
        temp_increase_c = temp_after - temp_before if temp_before is not None and temp_after is not None else None
        avg_power_w = (power_before + power_after) / 2 if power_before is not None and power_after is not None else None

        results = {
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms_avg": ttft_ms_avg,
            "avg_gpu_time_ms": round(avg_time_ms, 3),
            "stddev_gpu_time_ms": round(stddev_time_ms, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "runs_gpu_time_ms": [round(t, 3) for t in gpu_times_ms],
            "peak_gpu_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
            "temp_before_c": temp_before,
            "temp_after_c": temp_after,
            "avg_temp_c": round(avg_temp_c, 1) if avg_temp_c is not None else None,
            "temp_increase_c": round(temp_increase_c, 1) if temp_increase_c is not None else None,
            "power_before_w": round(power_before, 2) if power_before is not None else None,
            "power_after_w": round(power_after, 2) if power_after is not None else None,
            "avg_power_w": round(avg_power_w, 2) if avg_power_w is not None else None,
            "output_text_preview": generated_text[:100] + "..."
        }

    except Exception as e:
        print(f"ERROR during benchmark for prompt: '{prompt[:50]}...' - {e}")
        peak_memory_mb_err = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else None
        temp_err, power_err = get_nvidia_gpu_details()
        results = {
            "prompt": prompt, "status": "failed", "error_message": str(e),
            "peak_gpu_memory_mb_on_error": round(peak_memory_mb_err, 2) if peak_memory_mb_err is not None else None,
            "temp_on_error": temp_err, "power_on_error": power_err,
        }
    return results

# ---------- Main Benchmark Runner (CUDA Specific) ----------
def run_full_benchmark_cuda(output_filename="benchmark_results_cuda.json"):
    """Runs LLM benchmarks for all models and prompts on CUDA, saving results."""

    all_results = []
    device = "cuda"
    benchmark_dtype = getattr(torch, config.get("benchmark_dtype", "float16")) # Recommended dtype
    print(f"--- Running CUDA Benchmark with dtype: {benchmark_dtype}, Batch Size: {BATCH_SIZE} ---")

    # --- HF Login ---
    try:
        token = os.getenv("HF_TOKEN")
        if token: login(token=token); print("Logged in.")
        else: print("HF_TOKEN not set. Ensure models cached/public.")
    except Exception as e: print(f"Login failed: {e}")

    # --- Loop through models ---
    for model_id in model_list:
        print(f"\n{'='*20} Benchmarking Model: {model_id} {'='*20}")
        model, tokenizer, model_load_time = None, None, None
        current_model_params = {} # Store params for this model run

        try:
            # --- Load Model & Tokenizer ---
            print(f"Loading tokenizer {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

            # If needed, set padding token in current_generation_config, however, this is not always necessary
            current_generation_config = generation_config

            print(f"Loading model {model_id} (dtype: {benchmark_dtype})...")
            
            # Preload model to ensure it is downloaded before timing
            _ = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=benchmark_dtype).to(device) # Preload to ensure model is downloaded
            torch.cuda.empty_cache() # Clear cache before loading to avoid memory issues
            print(f"Model {model_id} downloaded, now loading to device...")
            
            # --- Load Model ---
            load_start = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=benchmark_dtype
            ).to(device).eval()
            load_end = time.time()
            model_load_time = load_end - load_start
            print(f"Model ready on {device} in {model_load_time:.2f} seconds.")

            # --- Record Benchmark Parameters for this model ---
            current_model_params = {
                "model_id": model_id,
                "benchmark_dtype": str(benchmark_dtype),
                "batch_size": BATCH_SIZE,
                "num_global_warmup_runs": NUM_GLOBAL_WARMUP_RUNS,
                "num_timed_runs_per_prompt": NUM_TIMED_RUNS_PER_PROMPT,
                "model_load_time_s": round(model_load_time, 2),
                "accelerator_used": "CUDA",
                "quantization_method": "None"
            }

            # --- Global Warm-up ---
            print(f"Running {NUM_GLOBAL_WARMUP_RUNS} global warm-up prompts...")
            for w_prompt in warm_prompts:
                w_inputs = tokenizer(w_prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    _ = model.generate(**w_inputs, generation_config=current_generation_config) # Use potentially updated config
            torch.cuda.synchronize(device)
            print("Global warm-up complete.")

            # --- Benchmark each prompt ---
            for prompt_text in prompt_list:
                print(f"--- Prompt: '{prompt_text[:50]}...' ---")
                # Pass the potentially model-specific generation config
                prompt_metrics = benchmark_model_on_prompt_cuda(
                    model, tokenizer, prompt_text, current_generation_config,
                    num_runs=NUM_TIMED_RUNS_PER_PROMPT
                )

                # Combine model parameters with prompt metrics for final record
                final_record = {**current_model_params, **prompt_metrics}
                all_results.append(final_record)

                # Save intermediate results
                with open(output_filename, "w") as f:
                    json.dump(all_results, f, indent=4)

        except Exception as e:
            print(f"FATAL ERROR for model {model_id}: {e}")
            all_results.append({
                **current_model_params, # Include params even on failure if available
                "prompt": "LOAD_OR_SETUP_FAILURE",
                "status": "load_or_setup_failed",
                "error_message": str(e),
            })
        finally:
            # --- Cleanup ---
            print(f"Cleaning up {model_id}...")
            del model; del tokenizer
            torch.cuda.empty_cache()
            print("Cleanup complete.")
            # Save final results again
            with open(output_filename, "w") as f:
                 json.dump(all_results, f, indent=4)

    print(f"\nCUDA Benchmark run complete. Results: {output_filename}")
    if NVML_AVAILABLE: pynvml.nvmlShutdown()

# --- Run ---
if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"llm_benchmark_results_cuda_{timestamp}.json"
    run_full_benchmark_cuda(output_filename=output_file)
    save_generation_config(generation_config, "generation_config.json")