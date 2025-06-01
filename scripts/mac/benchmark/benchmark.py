import os
import json
import gc
import threading
import time
import subprocess
import yaml

import torch
import psutil
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS backend not available. Please install a compatible PyTorch 2.x build.")

# Load configuration
script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

model_list = cfg["models"]
warm_prompts = cfg["warm_prompts"]
prompt_list = cfg["prompt_list"]

# GenerationConfig parameters from config
gen_config = GenerationConfig(
    max_new_tokens=cfg["generation"]["max_new_tokens"],
    do_sample=cfg["generation"]["do_sample"],
)

# Sampling settings
num_runs = cfg["sampling"]["num_runs_per_prompt"]
rss_interval = cfg["sampling"]["rss_sampler_interval_seconds"]
temp_samples = cfg["sampling"]["temperature_samples"]
temp_delay = cfg["sampling"]["temperature_delay_seconds"]

# Output settings
results_dir = cfg["output"]["results_directory"]
base_output_filename = cfg["output"]["base_output_filename"]


def get_mps_usage_mb():
    """
    Returns current MPS (Metal Performance Shaders) GPU memory allocated and reserved in MB.
    """
    alloc = torch.mps.current_allocated_memory() / (1024 ** 2)
    resv = torch.mps.driver_allocated_memory() / (1024 ** 2)
    return round(alloc, 2), round(resv, 2)


def get_rss_usage_mb():
    """
    Returns current process resident set size (RSS) in MB.
    """
    proc = psutil.Process(os.getpid())
    return round(proc.memory_info().rss / (1024 ** 2), 2)


def start_rss_sampler(proc, interval=0.05):
    """
    Launch a background thread that samples proc.memory_info().rss every `interval` seconds.
    Returns a dict containing 'peak_rss' and a threading.Event to stop sampling.
    """
    stop_event = threading.Event()
    sampler = {"peak_rss": 0, "stop_event": stop_event}

    def _run():
        while not stop_event.is_set():
            rss = proc.memory_info().rss
            if rss > sampler["peak_rss"]:
                sampler["peak_rss"] = rss
            time.sleep(interval)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return sampler


def get_gpu_temperature(samples=3, delay=0.1):
    """
    Measure GPU temperature multiple times using 'smctemp -g' CLI.
    Returns the average temperature in Celsius, or None if unavailable.
    """
    readings = []
    try:
        for _ in range(samples):
            output = subprocess.check_output(["smctemp", "-g"], text=True)
            readings.append(float(output.strip()))
            time.sleep(delay)
        return round(sum(readings) / len(readings), 2)
    except Exception:
        return None


def get_cpu_temperature(samples=3, delay=0.1):
    """
    Measure CPU temperature multiple times using 'smctemp -c' CLI.
    Returns the average temperature in Celsius, or None if unavailable.
    """
    readings = []
    try:
        for _ in range(samples):
            output = subprocess.check_output(["smctemp", "-c"], text=True)
            readings.append(float(output.strip()))
            time.sleep(delay)
        return round(sum(readings) / len(readings), 2)
    except Exception:
        return None




def benchmark_model_on_prompt_mps(model, tokenizer, prompt, dtype):
    """
    Runs benchmark for a single prompt on MPS/CPU, returns metrics.
    """
    results = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]

        process = psutil.Process(os.getpid())
        peak_mem_mb = 0
        peak_mps_mb = 0

        # Measure TTFT runs
        ttft_runs = []
        for _ in range(num_runs):
            start_ttft = time.time()
            with torch.no_grad():
                _ = model.generate(max_new_tokens=1, do_sample=False, **inputs)
            end_ttft = time.time()
            ttft_runs.append((end_ttft - start_ttft) * 1000)
        ttft_ms_avg = round(sum(ttft_runs) / len(ttft_runs), 2)

        # Measure full-generation runs
        full_runs = []
        output_tokens = 0
        generated_text = ""
        mem_before_mb = mem_after_mb = memory_delta_mb = None
        cpu_temp_before = cpu_temp_after = None
        gpu_temp_before = gpu_temp_after = None
        mps_alloc_before = mps_resv_before = None
        mps_alloc_after = mps_resv_after = None
        peak_rss_mb = 0

        for i in range(num_runs):
            # a) “Before” snapshots
            mem_before_mb = process.memory_full_info().rss / (1024 * 1024)
            gpu_temp_before = get_gpu_temperature()
            cpu_temp_before = get_cpu_temperature()
            sampler = start_rss_sampler(process, interval=rss_interval)

            mps_alloc_before, mps_resv_before = get_mps_usage_mb()
            rss_before = get_rss_usage_mb()

            # b) Run inference
            start_full = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    generation_config=gen_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **inputs,
                )
            end_full = time.time()

            # c) “After” snapshots
            mps_alloc_after, mps_resv_after = get_mps_usage_mb()
            # Track peak MPS allocation in MB
            if mps_alloc_after > peak_mps_mb:
                peak_mps_mb = mps_alloc_after
            mem_after_mb = process.memory_full_info().rss / (1024 * 1024)
            sampler["stop_event"].set()
            time.sleep(0.05)
            peak_rss_mb = sampler["peak_rss"] / (1024 ** 2)
            gpu_temp_after = get_gpu_temperature()
            cpu_temp_after = get_cpu_temperature()

            memory_delta_mb = mem_after_mb - mem_before_mb
            current_peak = max(mem_before_mb, mem_after_mb)
            if current_peak > peak_mem_mb:
                peak_mem_mb = current_peak

            iter_time_ms = (end_full - start_full) * 1000
            full_runs.append(iter_time_ms)

            if i == 0:
                actual_output_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                generated_text = tokenizer.decode(actual_output_ids, skip_special_tokens=True)
                output_tokens = len(actual_output_ids)

            torch.mps.empty_cache()
            gc.collect()

        avg_time_ms = round(sum(full_runs) / len(full_runs), 3) if full_runs else 0
        tokens_per_sec = round(output_tokens / (avg_time_ms / 1000.0), 2) if full_runs else 0
        peak_memory_mb = round(peak_mem_mb, 2)
        cpu_temp_delta_c = (
            round(cpu_temp_after - cpu_temp_before, 2)
            if cpu_temp_before is not None and cpu_temp_after is not None
            else None
        )
        gpu_temp_delta_c = (
            round(gpu_temp_after - gpu_temp_before, 2)
            if gpu_temp_before is not None and gpu_temp_after is not None
            else None
        )

        results = {
            "prompt": prompt,
            "status": "success",
            "error_message": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "avg_time_ms": avg_time_ms,
            "tokens_per_sec": tokens_per_sec,
            "ttft_ms_avg": ttft_ms_avg,
            "output_text_preview": generated_text[:100] + "...",
            "cpu_temp_before_c": cpu_temp_before,
            "cpu_temp_after_c": cpu_temp_after,
            "cpu_temp_delta_c": cpu_temp_delta_c,
            "gpu_temp_before_c": gpu_temp_before,
            "gpu_temp_after_c": gpu_temp_after,
            "gpu_temp_delta_c": gpu_temp_delta_c,
            "mps_alloc_before_mb": mps_alloc_before,
            "mps_resv_before_mb": mps_resv_before,
            "rss_before_mb": rss_before,
            "mps_alloc_after_mb": mps_alloc_after,
            "mps_resv_after_mb": mps_resv_after,
            "rss_after_mb": mem_after_mb,
            "peak_rss_during_run_mb": peak_rss_mb,
            "peak_mps_mb": peak_mps_mb,
        }
    except Exception as e:
        results = {"prompt": prompt, "status": "failed", "error_message": str(e)}
    return results


def run_full_benchmark_mps(output_filename):
    """
    Runs benchmarks for all models and prompts on MPS/CPU, saving results.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_results = []
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    benchmark_dtype = torch.float16
    print(f"--- Running Benchmark on device: {device} with dtype: {benchmark_dtype} ---")

    try:
        token = ""
        if token:
            login(token=token)
            print("Logged in to Hugging Face Hub successfully.")
        else:
            print("HF_TOKEN not set. Skipping login.")
    except Exception as e:
        print(f"Warning: Hugging Face login failed or skipped: {e}")

    for model_id in model_list:
        print(f"\n{'='*20} Benchmarking Model: {model_id} {'='*20}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            print(f"Loading model {model_id} to CPU (dtype={benchmark_dtype})...")
            load_start = time.time()
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=benchmark_dtype)
            load_end = time.time()

            rss_after_load = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            print(f"RSS after loading onto CPU: {rss_after_load:.2f} MB")

            print(f"Moving model to {device} and emptying cache...")
            model.to(device)
            model.eval()
            torch.mps.empty_cache()
            gc.collect()

            rss_after_move = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            model_load_time = load_end - load_start
            print(f"Model on {device} in {model_load_time:.2f}s; RSS after move: {rss_after_move:.2f} MB")

            print("Running global warm-up...")
            for w_prompt in warm_prompts:
                w_inputs = tokenizer(w_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    _ = model.generate(**w_inputs, max_new_tokens=16, do_sample=False)
            print("Global warm-up complete.")

            for prompt_text in prompt_list:
                print(f"--- Prompt: '{prompt_text[:50]}...' ---")
                prompt_results = benchmark_model_on_prompt_mps(model, tokenizer, prompt_text, benchmark_dtype)
                prompt_results.update(
                    {
                        "model_id": model_id,
                        "device": device,
                        "dtype": str(benchmark_dtype),
                        "model_load_time_s": round(model_load_time, 2),
                    }
                )
                all_results.append(prompt_results)
                with open(output_filename, "w") as f:
                    json.dump(all_results, f, indent=4)

        except Exception as e:
            print(f"FATAL ERROR for {model_id}: {e}")
            all_results.append({
                "model_id": model_id,
                "status": "load_or_setup_failed",
                "error_message": str(e),
                "device": device,
                "dtype": str(benchmark_dtype),
            })
        finally:
            print(f"Cleaning up resources for {model_id}...")
            try:
                del model
                del tokenizer
            except NameError:
                pass
            if device == "mps":
                torch.mps.empty_cache()
            print("Cleanup complete.")
            with open(output_filename, "w") as f:
                json.dump(all_results, f, indent=4)

    print(f"\nBenchmark run complete. Results saved to {output_filename}")


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(script_dir, f"{base_output_filename}_{timestamp}.json")
    run_full_benchmark_mps(output_filename=output_file)