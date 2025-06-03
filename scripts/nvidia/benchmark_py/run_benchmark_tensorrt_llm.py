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
from tensorrt_llm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load config from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------- Model List (LLM Only) ----------
model_list = config["model_list_tensorrt_llm"]

# ---------- Warm-up Prompts ----------
warm_prompts = config["warm_prompts"]
NUM_GLOBAL_WARMUP_RUNS = len(warm_prompts)

# ---------- Benchmark Prompts ----------
prompt_list = config["prompt_list"]
NUM_TIMED_RUNS_PER_PROMPT = config["num_timed_runs_per_prompt"] # Number of repetitions for timing

# ---------- Generation Config ----------
generation_config = SamplingParams(
    max_tokens=config["max_new_tokens"],
    temperature=0,
)
BATCH_SIZE = config["batch_size"]

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

# ---------- Benchmark Function (CUDA with TensorRT LLM - Per Prompt) ----------
def benchmark_model_on_prompt_tensorrt_llm(model, tokenizer, prompt, generation_config_obj, num_runs=3):
    """Runs benchmark for a single prompt on CUDA with TensorRT-LLM, returns metrics dictionary."""
    results = {}
    device = "cuda"
    try:
        inputs = tokenizer.encode(prompt)
        
        # --- TTFT Runs ---
        ttft_config_obj = SamplingParams(
            max_tokens=1,
            temperature=0,
        )
        ttft_config_obj.max_tokens = 1  # No new tokens for TTFT
        ttft_runs = []
        for _ in range(num_runs):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            _ = model.generate(
                inputs=[inputs],
                sampling_params=ttft_config_obj,
                use_tqdm=True
            )
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

            # Ensure generation_config is passed correctly
            output = model.generate(
                inputs=[inputs],
                sampling_params=generation_config_obj,
                use_tqdm=True
            )

            end_event.record()
            torch.cuda.synchronize(device)
            iter_time_ms = start_event.elapsed_time(end_event)
            gpu_times_ms.append(iter_time_ms)

            if i == 0: # Decode only once
                input_tokens = len(inputs)  # Use original input length
                actual_output_ids = output[0].outputs[0].token_ids
                generated_text = output[0].outputs[0].text
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
            "status": "success",
            "error_message": None,
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
def run_full_benchmark_tensorrt_llm(output_filename="benchmark_results_tensorrt_llm.json"):
    """Runs LLM benchmarks for all models and prompts on CUDA using TensorRT-LLM, saving results."""

    all_results = []
    device = "cuda"
    benchmark_dtype = getattr(torch, config.get("benchmark_dtype", "float16")) # Recommended dtype
    print(f"--- Running TensorRT-LLM Benchmark with dtype: {benchmark_dtype} ---")

    # --- HF Login ---
    try:
        token = os.getenv("HF_TOKEN")
        if token: login(token=token); print("Logged in.")
        else: print("HF_TOKEN not set. Ensure models cached/public.")
    except Exception as e: print(f"Login failed: {e}")

    # --- Loop through models ---
    for model_id in model_list:
        print(f"\n{'='*20} Benchmarking Model: {model_id} {'='*20}")
        model, model_load_time = None, None
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
            print(f"Model {model_id} downloaded, now loading with TensorRT-LLM...")
            
            # --- Load Model ---
            load_start = time.time()
            model = LLM(model=model_id, tokenizer=model_id, trust_remote_code=True,) # Add dtype=
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
                "accelerator_used": "CUDA + TensorRT-LLM",
                "quantization_method": "None"
            }

            # --- Global Warm-up ---
            print(f"Running {NUM_GLOBAL_WARMUP_RUNS} global warm-up prompts...")
            for w_prompt in warm_prompts:
                w_inputs = tokenizer.encode(w_prompt)
                model.generate(
                    inputs=[w_inputs],
                    sampling_params=current_generation_config,
                    use_tqdm=True,
                )
            torch.cuda.synchronize(device)
            print("Global warm-up complete.")

            # --- Benchmark each prompt ---
            for prompt_text in prompt_list:
                print(f"--- Prompt: '{prompt_text[:50]}...' ---")
                # Pass the potentially model-specific generation config
                prompt_metrics = benchmark_model_on_prompt_tensorrt_llm(
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

    print(f"\nCUDA + TensorRT LLM Benchmark run complete. Results: {output_filename}")
    if NVML_AVAILABLE: pynvml.nvmlShutdown()

# --- Run ---
if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"llm_benchmark_results_tensorrt_llm_{timestamp}.json"
    run_full_benchmark_tensorrt_llm(output_filename=output_file)