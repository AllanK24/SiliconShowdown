# mac/benchmark/benchmark_mps.py
import os
import json
import gc
import threading
import time
import subprocess
import signal
import yaml
import tempfile
import sys
import statistics
import random
import numpy as np

import torch
import psutil
from huggingface_hub import login, snapshot_download, HfFolder
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_SEED = 42
CURRENT_SEED_USED = DEFAULT_SEED
DEFAULT_MPS_DTYPE = torch.float16 # ADDED default dtype

def set_seed(seed_value: int):
    # ... (same as before) ...
    global CURRENT_SEED_USED
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    CURRENT_SEED_USED = seed_value
    print(f"INFO: Random seeds set to {seed_value} for Python, NumPy, and PyTorch (including MPS).")


def save_generation_config(generation_config: GenerationConfig, filename="generation_config.json", seed_used=None):
    # ... (same as before) ...
    config_dict = generation_config.to_dict()
    if seed_used is not None:
        config_dict["_note_seed_used_for_run"] = seed_used
    with open(filename, "w") as f:
        json.dump(config_dict, f, indent=4)

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

if not os.path.exists(config_path):
    print(f"FATAL: Configuration file not found at {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

reproducibility_cfg = cfg.get("reproducibility", {})
CONFIG_SEED_VALUE = reproducibility_cfg.get("seed")

# MODIFIED: model_list can now contain dicts with model_id and dtype
model_configs_from_yaml = cfg["models"] # This will be a list of strings or dicts
warm_prompts = cfg["warm_prompts"]
prompt_list = cfg["prompt_list"]

user_generation_params = {
    "max_new_tokens": cfg["generation"]["max_new_tokens"],
    "do_sample": cfg["generation"]["do_sample"],
}
if user_generation_params["do_sample"]:
    if "temperature" in cfg["generation"]:
        user_generation_params["temperature"] = cfg["generation"]["temperature"]
    if "top_k" in cfg["generation"]:
        user_generation_params["top_k"] = cfg["generation"]["top_k"]
    if "top_p" in cfg["generation"]:
        user_generation_params["top_p"] = cfg["generation"]["top_p"]

num_runs_cfg = cfg["sampling"]["num_runs_per_prompt"]
rss_interval_cfg = cfg["sampling"]["rss_sampler_interval_seconds"]
temp_samples_cfg = cfg["sampling"]["temperature_samples"]
temp_delay_cfg = cfg["sampling"]["temperature_delay_seconds"]

results_dir_from_config = cfg["output"]["results_directory"]
base_output_filename_cfg = cfg["output"]["base_output_filename"]

# --- Helper functions (ensure_sudo_active, get_short_powermetrics_sample_w, etc.) ---
# ... (These remain unchanged) ...
def ensure_sudo_active(interactive_prompt_timeout=60):
    sudo_check_process = subprocess.run(
        ["sudo", "-nv"], capture_output=True, text=True, timeout=5
    )
    if sudo_check_process.returncode == 0:
        return True
    else:
        print("\nSudo privileges may have expired or are not available.")
        print("Attempting to refresh/validate sudo (password prompt may appear)...")
        try:
            subprocess.run(
                ["sudo", "-v"], check=True, timeout=interactive_prompt_timeout
            )
            print("Sudo privileges successfully validated/refreshed.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to validate/refresh sudo privileges: {e}")
            return False
        except subprocess.TimeoutExpired:
            print(f"ERROR: Timeout ({interactive_prompt_timeout}s) while waiting for sudo password entry.")
            return False
    return False

def get_short_powermetrics_sample_w(duration_s=2, interval_ms=1000, prefix="short_pm_"):
    if not ensure_sudo_active():
        print(f"Warning: Sudo not active for short powermetrics (prefix: {prefix}). Power sample will be skipped.")
        return None
    pm_proc = None
    temp_pm_file_path = None
    avg_power_w = None
    num_samples_to_take = max(1, int(duration_s * 1000 / interval_ms))
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt", prefix=prefix) as tmp_f:
            temp_pm_file_path = tmp_f.name
        pm_command = ["sudo", "powermetrics", "-i", str(interval_ms),
                      "--samplers", "cpu_power", "-n", str(num_samples_to_take), # Note: This samples CPU_POWER, not GPU directly
                      "-o", temp_pm_file_path]
        pm_proc = subprocess.Popen(pm_command, stderr=subprocess.PIPE, text=True)
        wait_timeout = (num_samples_to_take * interval_ms / 1000.0) + 5.0
        pm_stderr_data = ""
        try:
            _, pm_stderr_data = pm_proc.communicate(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            print(f"Warning: Short powermetrics sample (prefix: {prefix}) timed out. Killing.")
            if pm_proc.poll() is None: pm_proc.kill()
            try: _, pm_stderr_data = pm_proc.communicate(timeout=2)
            except Exception: pass
        if pm_stderr_data and pm_stderr_data.strip():
            print(f"Short powermetrics sample (prefix: {prefix}) stderr: {pm_stderr_data.strip()}")

        power_readings = []
        if temp_pm_file_path and os.path.exists(temp_pm_file_path):
            with open(temp_pm_file_path, "r") as f_in:
                for line in f_in:
                    try:
                        if "Combined Power" in line and "mW" in line and ":" in line: # This is often CPU package power
                            value_part = line.split(":")[1].strip()
                            numeric_value_str = value_part.split("mW")[0].strip()
                            power_mw = float(numeric_value_str)
                            power_w = power_mw / 1000.0
                            power_readings.append(power_w)
                    except ValueError: pass
                    except Exception: pass
            if power_readings:
                avg_power_w = round(sum(power_readings) / len(power_readings), 2)
    except Exception as e:
        print(f"Error during short powermetrics sample (prefix: {prefix}): {e}")
    finally:
        if pm_proc and pm_proc.poll() is None: pm_proc.kill(); pm_proc.wait()
        if temp_pm_file_path and os.path.exists(temp_pm_file_path):
            try: os.remove(temp_pm_file_path)
            except OSError: pass
    return avg_power_w


def get_mps_usage_mb():
    alloc = torch.mps.current_allocated_memory() / (1024 ** 2)
    resv = torch.mps.driver_allocated_memory() / (1024 ** 2)
    return round(alloc, 2), round(resv, 2)

def get_rss_usage_mb():
    proc = psutil.Process(os.getpid())
    return round(proc.memory_info().rss / (1024 ** 2), 2)

def start_rss_sampler(proc, interval=0.05):
    stop_event = threading.Event()
    sampler = {"peak_rss": 0, "stop_event": stop_event}
    def _run():
        while not stop_event.is_set():
            try:
                if not proc.is_running(): break
                rss = proc.memory_info().rss
                if rss > sampler["peak_rss"]: sampler["peak_rss"] = rss
            except psutil.NoSuchProcess: break
            except Exception as e: print(f"RSS Sampler error: {e}"); break
            time.sleep(interval)
    thread = threading.Thread(target=_run, daemon=True); thread.start()
    return sampler

def get_temperature_with_smctemp(param, samples=3, delay=0.1):
    readings = []
    command = ["smctemp", param]
    try:
        for _ in range(samples):
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
            readings.append(float(output.strip()))
            time.sleep(delay)
        return round(sum(readings) / len(readings), 1) if readings else None
    except FileNotFoundError: print(f"Warning: 'smctemp' command not found for '{param}'. Temp will be null."); return None
    except subprocess.CalledProcessError as e: print(f"Warning: 'smctemp {param}' failed: {e}. Temp will be null."); return None
    except Exception as e: print(f"Warning: Error getting temp '{param}': {e}. Temp will be null."); return None

def get_gpu_temperature(samples=3, delay=0.1): return get_temperature_with_smctemp("-g", samples, delay)
def get_cpu_temperature(samples=3, delay=0.1): return get_temperature_with_smctemp("-c", samples, delay)

# benchmark_model_on_prompt_mps takes benchmark_dtype_str from its caller
def benchmark_model_on_prompt_mps(model, tokenizer, prompt, benchmark_gen_config: GenerationConfig, benchmark_dtype_str: str, seed_used: int): # MODIFIED: passed benchmark_dtype_str
    results = {}
    device_str = "mps"
    # benchmark_dtype_str is now passed in, reflecting the actual dtype used for the model load.

    try:
        # ... (rest of the function is largely the same as your previous version) ...
        # It will use the benchmark_dtype_str (passed in) for reporting.
        inputs = tokenizer(prompt, return_tensors="pt").to(device_str)
        torch.mps.synchronize()
        input_tokens = inputs.input_ids.shape[1]
        process = psutil.Process(os.getpid())
        peak_host_mem_during_prompt_mb = 0
        peak_mps_alloc_during_prompt_mb = 0

        ttft_runs_ms = []
        print("Running TTFT measurements...")
        ttft_params = benchmark_gen_config.to_dict()
        ttft_params.update({
            "max_new_tokens": 1,
            "do_sample": False,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "num_beams": 1 
        })
        ttft_params_cleaned = {k: v for k, v in ttft_params.items() if v is not None}
        ttft_gen_config_effective = GenerationConfig(**ttft_params_cleaned)

        for _ in range(num_runs_cfg):
            torch.mps.synchronize()
            start_ttft = time.perf_counter()
            with torch.inference_mode():
                _ = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, generation_config=ttft_gen_config_effective)
            torch.mps.synchronize()
            end_ttft = time.perf_counter()
            ttft_runs_ms.append((end_ttft - start_ttft) * 1000)
        ttft_ms_avg = round(sum(ttft_runs_ms) / len(ttft_runs_ms), 2) if ttft_runs_ms else 0.0
        print(f"TTFT avg: {ttft_ms_avg:.2f} ms")

        power_before_prompt_w = get_short_powermetrics_sample_w(duration_s=2, prefix="mps_pre_prompt_pm_")

        full_run_times_ms = []
        energy_consumption_j_runs = []
        generated_text_preview = ""
        actual_output_tokens = 0

        rss_before_all_runs_mb = get_rss_usage_mb()
        mps_alloc_before_all_runs_mb, mps_resv_before_all_runs_mb = get_mps_usage_mb()
        cpu_temp_before_c = get_cpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
        gpu_temp_before_c = get_gpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)

        print(f"Running {num_runs_cfg} full generation runs with config: {benchmark_gen_config.to_dict()}...")
        for i in range(num_runs_cfg):
            print(f"  Run {i+1}/{num_runs_cfg} for prompt '{prompt[:30]}...'")
            pm_proc, temp_pm_file_path, rss_sampler_thread = None, None, None
            try:
                sudo_ok_for_main_pm = ensure_sudo_active()
                if sudo_ok_for_main_pm:
                    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt", prefix=f"mps_main_pm_run{i}_") as tmp_f:
                        temp_pm_file_path = tmp_f.name
                    pm_command = ["sudo", "powermetrics", "-i", "1000", "--samplers", "cpu_power", "-o", temp_pm_file_path]
                    pm_proc = subprocess.Popen(pm_command, stderr=subprocess.PIPE, text=True)
                    time.sleep(0.5)
                    if pm_proc.poll() is not None:
                        pm_stderr_early = pm_proc.stderr.read() if pm_proc.stderr else ""
                        print(f"Warning: Main Powermetrics (MPS run {i+1}, PID {pm_proc.pid if pm_proc else 'N/A'}) failed/exited early. Stderr: {pm_stderr_early}")
                        pm_proc = None
                else:
                    print(f"Warning: Sudo not active for main powermetrics (MPS run {i+1}). Energy metrics will be skipped.")
                    pm_proc = None

                rss_sampler_thread = start_rss_sampler(process, interval=rss_interval_cfg)


                torch.mps.synchronize()
                start_full_run = time.perf_counter()
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        generation_config=benchmark_gen_config,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                torch.mps.synchronize()
                end_full_run = time.perf_counter()
                inference_duration_s = end_full_run - start_full_run
                full_run_times_ms.append(inference_duration_s * 1000)
                current_mps_alloc_mb, _ = get_mps_usage_mb()
                if current_mps_alloc_mb > peak_mps_alloc_during_prompt_mb:
                    peak_mps_alloc_during_prompt_mb = current_mps_alloc_mb

                pm_stderr_data = ""
                if pm_proc:
                    current_pm_pid = pm_proc.pid
                    if pm_proc.poll() is None: pm_proc.send_signal(signal.SIGINT)
                    try: pm_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if pm_proc.poll() is None: pm_proc.send_signal(signal.SIGTERM)
                        try: pm_proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            if pm_proc.poll() is None: pm_proc.kill(); pm_proc.wait(timeout=2)
                    try:
                        if pm_proc.stderr and not pm_proc.stderr.closed: _, pm_stderr_data = pm_proc.communicate(timeout=2)
                    except Exception as e_comm: print(f"Error during final communicate for main powermetrics (MPS run {i+1}, PID {current_pm_pid}): {e_comm}")
                    if pm_stderr_data and pm_stderr_data.strip(): print(f"Main Powermetrics stderr (MPS run {i+1}, PID {current_pm_pid}): {pm_stderr_data.strip()}")

                power_readings_w = []
                if pm_proc and temp_pm_file_path and os.path.exists(temp_pm_file_path):
                    with open(temp_pm_file_path, "r") as f_in:
                        for line in f_in:
                            try:
                                if "Combined Power" in line and "mW" in line and ":" in line:
                                    value_part = line.split(":")[1].strip(); numeric_value_str = value_part.split("mW")[0].strip()
                                    power_mw = float(numeric_value_str); power_w = power_mw / 1000.0
                                    power_readings_w.append(power_w)
                            except ValueError: print(f"Warning: Could not parse power value from line (MPS run {i+1}): '{line.strip()}'")
                            except Exception as e_parse: print(f"Warning: Unexpected error parsing power line (MPS run {i+1}) '{line.strip()}': {e_parse}")

                energy_j_this_run = None
                if power_readings_w and inference_duration_s > 0:
                    avg_power_w_this_run = sum(power_readings_w) / len(power_readings_w)
                    energy_j_this_run = avg_power_w_this_run * inference_duration_s
                energy_consumption_j_runs.append(energy_j_this_run)

                if i == 0:
                    actual_output_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                    generated_text_preview = tokenizer.decode(actual_output_ids, skip_special_tokens=True)[:100] + "..."
                    actual_output_tokens = len(actual_output_ids)

                if rss_sampler_thread and rss_sampler_thread.get("stop_event"):
                    rss_sampler_thread["stop_event"].set()
                    time.sleep(rss_interval_cfg + 0.1) 
                    current_run_peak_rss_mb = rss_sampler_thread["peak_rss"] / (1024**2)
                    if current_run_peak_rss_mb > peak_host_mem_during_prompt_mb:
                        peak_host_mem_during_prompt_mb = current_run_peak_rss_mb

                torch.mps.empty_cache(); gc.collect()
            except Exception as e_run:
                print(f"ERROR within MPS benchmark run {i+1} for prompt '{prompt[:30]}...': {e_run}")
                import traceback; traceback.print_exc()
                full_run_times_ms.append(None)
                energy_consumption_j_runs.append(None)
            finally:
                if rss_sampler_thread and rss_sampler_thread.get("stop_event") and not rss_sampler_thread["stop_event"].is_set():
                    rss_sampler_thread["stop_event"].set()
                if pm_proc and pm_proc.poll() is None: pm_proc.kill(); pm_proc.wait()
                if temp_pm_file_path and os.path.exists(temp_pm_file_path):
                    try: os.remove(temp_pm_file_path)
                    except OSError as e_rm: print(f"Warning: Could not remove temp powermetrics file {temp_pm_file_path}: {e_rm}")
        rss_after_all_runs_mb = get_rss_usage_mb()
        mps_alloc_after_all_runs_mb, mps_resv_after_all_runs_mb = get_mps_usage_mb()
        cpu_temp_after_c = get_cpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
        gpu_temp_after_c = get_gpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
        power_after_prompt_w = get_short_powermetrics_sample_w(duration_s=2, prefix="mps_post_prompt_pm_")

        valid_full_run_times_ms = [t for t in full_run_times_ms if t is not None]
        avg_time_ms = round(sum(valid_full_run_times_ms) / len(valid_full_run_times_ms), 3) if valid_full_run_times_ms else 0
        
        gpu_times_per_run = [round(t, 3) for t in valid_full_run_times_ms] if valid_full_run_times_ms else []
        stddev_gpu_time_ms = None
        if len(gpu_times_per_run) > 1:
            stddev_gpu_time_ms = round(statistics.stdev(gpu_times_per_run), 3)
        elif len(gpu_times_per_run) == 1: 
            stddev_gpu_time_ms = 0.0

        tokens_per_sec = round(actual_output_tokens / (avg_time_ms / 1000.0), 2) if avg_time_ms > 0 and actual_output_tokens > 0 else 0
        valid_energy_runs = [e for e in energy_consumption_j_runs if e is not None]
        avg_energy_consumption_j = round(sum(valid_energy_runs) / len(valid_energy_runs), 4) if valid_energy_runs else None
        avg_power_during_inference_w = None
        if avg_energy_consumption_j is not None and avg_time_ms > 0:
            avg_power_during_inference_w = round(avg_energy_consumption_j / (avg_time_ms / 1000.0), 2)
        cpu_temp_delta_c = round(cpu_temp_after_c - cpu_temp_before_c, 1) if cpu_temp_before_c is not None and cpu_temp_after_c is not None else None
        gpu_temp_delta_c = round(gpu_temp_after_c - gpu_temp_before_c, 1) if gpu_temp_before_c is not None and gpu_temp_after_c is not None else None
        gpu_temp_avg_c = None
        if gpu_temp_before_c is not None and gpu_temp_after_c is not None:
            gpu_temp_avg_c = round((gpu_temp_before_c + gpu_temp_after_c) / 2.0, 1)
        elif gpu_temp_before_c is not None:
            gpu_temp_avg_c = round(gpu_temp_before_c, 1)
        elif gpu_temp_after_c is not None:
            gpu_temp_avg_c = round(gpu_temp_after_c, 1)
        peak_host_mem_during_prompt_mb = round(peak_host_mem_during_prompt_mb, 2) if peak_host_mem_during_prompt_mb > 0 else None
        peak_mps_alloc_during_prompt_mb = round(peak_mps_alloc_during_prompt_mb, 2) if peak_mps_alloc_during_prompt_mb > 0 else None

        results = {
            "prompt": prompt, "status": "success", "error_message": None,
            "seed_used": seed_used,
            "batch_size": 1,
            "num_timed_runs_per_prompt": num_runs_cfg,
            "input_tokens": input_tokens, "output_tokens": actual_output_tokens,
            "ttft_ms_avg": ttft_ms_avg,
            "avg_time_ms": avg_time_ms,
            "stddev_gpu_time_ms": stddev_gpu_time_ms,
            "runs_gpu_time_ms": gpu_times_per_run,
            "tokens_per_sec": tokens_per_sec,
            "avg_energy_consumption_j": avg_energy_consumption_j,
            "avg_power_during_inference_w": avg_power_during_inference_w,
            "power_before_w": power_before_prompt_w,
            "power_after_w": power_after_prompt_w,
            "output_text_preview": generated_text_preview,
            "cpu_temp_before_c": cpu_temp_before_c, "cpu_temp_after_c": cpu_temp_after_c, "cpu_temp_increase_c": cpu_temp_delta_c,
            "gpu_temp_before_c": gpu_temp_before_c, "gpu_temp_after_c": gpu_temp_after_c, "gpu_temp_avg_c": gpu_temp_avg_c, "gpu_temp_increase_c": gpu_temp_delta_c,
            "mps_alloc_before_mb": round(mps_alloc_before_all_runs_mb, 2), "mps_resv_before_mb": round(mps_resv_before_all_runs_mb, 2), "rss_before_mb": round(rss_before_all_runs_mb, 2),
            "mps_alloc_after_mb": round(mps_alloc_after_all_runs_mb, 2), "mps_resv_after_mb": round(mps_resv_after_all_runs_mb, 2), "rss_after_mb": round(rss_after_all_runs_mb, 2),
            "peak_mps_mb": peak_mps_alloc_during_prompt_mb, "peak_host_memory_mb": peak_host_mem_during_prompt_mb,
        }

    except Exception as e_outer:
        print(f"MAJOR ERROR during MPS benchmark for prompt '{prompt[:50]}...': {e_outer}")
        import traceback; traceback.print_exc()
        results = {"prompt": prompt, "status": "failed", "error_message": str(e_outer), "seed_used": seed_used}
    return results

def download_model_if_needed(model_id: str, token: str = None):
    # ... (same as before) ...
    print(f"Ensuring model '{model_id}' files are downloaded to cache...")
    actual_token_to_use = token if token and token.strip() else None
    try:
        snapshot_path = snapshot_download(
            repo_id=model_id, local_files_only=False,
            resume_download=True, token=actual_token_to_use,
        )
        print(f"Model '{model_id}' files are available in cache: {snapshot_path}")
        return True
    except HfHubHTTPError as e:
        print(f"Error downloading '{model_id}': Hugging Face Hub API error. {e}")
        if "401" in str(e): print(f"This might be a private model ('{model_id}'). Ensure you are logged in or provide a valid token.")
        elif "404" in str(e): print(f"Model or revision for '{model_id}' not found on Hugging Face Hub.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while ensuring model '{model_id}' is downloaded: {e}")
        import traceback; traceback.print_exc()
        return False

def run_full_benchmark_mps(output_filename_param, abs_results_dir_param, effective_seed_to_use):
    if not torch.backends.mps.is_available():
        # ... (MPS unavailable error) ...
        print("FATAL: MPS backend not available. Please install a compatible PyTorch 2.x build. Exiting MPS benchmark.")
        error_result = [{"status": "mps_unavailable", "error_message": "MPS backend not available."}]
        try:
            os.makedirs(os.path.dirname(output_filename_param), exist_ok=True)
            with open(output_filename_param, "w") as f: json.dump(error_result, f, indent=4)
        except Exception as e_write:
            print(f"Could not write MPS unavailable status to {output_filename_param}: {e_write}")
        return

    # ... (sudo validation) ...
    print("Validating sudo privileges for powermetrics (MPS Benchmark)...")
    if not ensure_sudo_active(interactive_prompt_timeout=30):
        print("WARNING: Sudo privileges could not be obtained for MPS. Powermetrics will be skipped.")
    else:
        print("Sudo privileges validated for upcoming powermetrics calls (MPS).")

    all_prompt_results = []
    device_str = "mps" # This script is MPS specific

    hf_token_from_config = cfg.get("huggingface_token")
    try:
        if hf_token_from_config and hf_token_from_config.strip():
            login(token=hf_token_from_config); print("Logged in to Hugging Face Hub (token from config).")
        else:
            if HfFolder.get_token() is not None: print("Found existing Hugging Face login token.")
            else: print("No HF token in config or system. Proceeding with anonymous access.")
    except Exception as e_login: print(f"Warning: HF login/token check failed: {e_login}")

    # Iterate through model configurations from YAML
    for model_config_item in model_configs_from_yaml:
        model_id_for_mps = ""
        requested_dtype_str = None

        if isinstance(model_config_item, str):
            model_id_for_mps = model_config_item
            # Default to float16 if only model_id string is provided
            model_torch_dtype = DEFAULT_MPS_DTYPE
            benchmark_dtype_str_for_model = str(model_torch_dtype)
        elif isinstance(model_config_item, dict):
            model_id_for_mps = model_config_item.get("model_id")
            requested_dtype_str = model_config_item.get("dtype", "float16") # Default to float16 string
            if not model_id_for_mps:
                print(f"Warning: Skipping model entry in config due to missing 'model_id': {model_config_item}")
                continue

            if requested_dtype_str.lower() == "bfloat16":
                model_torch_dtype = torch.bfloat16
            elif requested_dtype_str.lower() == "float16":
                model_torch_dtype = torch.float16
            else:
                print(f"Warning: Unsupported dtype '{requested_dtype_str}' for model {model_id_for_mps}. Defaulting to {str(DEFAULT_MPS_DTYPE)}.")
                model_torch_dtype = DEFAULT_MPS_DTYPE
            benchmark_dtype_str_for_model = str(model_torch_dtype) # e.g. "torch.bfloat16"
        else:
            print(f"Warning: Skipping invalid model entry in config: {model_config_item}")
            continue

        print(f"\n{'='*20} Preparing MPS Model: {model_id_for_mps} with DType: {benchmark_dtype_str_for_model} {'='*20}")
        print(f"--- Running MPS Benchmark on device: {device_str} with DType: {benchmark_dtype_str_for_model} using SEED: {effective_seed_to_use} ---")
        print(f"--- Output will be saved to: {output_filename_param} ---")


        download_successful = download_model_if_needed(model_id_for_mps, token=hf_token_from_config)
        if not download_successful:
            all_prompt_results.append({
                "model_id": model_id_for_mps, "status": "download_failed",
                "error_message": f"Failed to download/cache model files for {model_id_for_mps}.",
                "accelerator_used": device_str.upper(),
                "benchmark_dtype": benchmark_dtype_str_for_model, # Use the determined dtype
                "quantization_method": "None",
                "seed_used": effective_seed_to_use,
            })
            # ... (save intermediate results) ...
            try:
                with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
            except Exception as e_write_dl_fail: print(f"ERROR writing to {output_filename_param} (after MPS download fail): {e_write_dl_fail}")

            continue

        model_mps, tokenizer_mps = None, None
        model_load_time_cpu_s, model_move_to_device_time_s = 0.0, None
        rss_after_cpu_load_mb, rss_after_device_move_mb = None, None
        effective_benchmark_gen_config = None

        try:
            print(f"Loading tokenizer {model_id_for_mps} to CPU (from cache)...")
            cpu_load_start_time = time.perf_counter()
            tokenizer_mps = AutoTokenizer.from_pretrained(model_id_for_mps, use_fast=True, local_files_only=True)

            print(f"Loading model {model_id_for_mps} to CPU (target torch_dtype={benchmark_dtype_str_for_model}) (from cache)...") # Log target dtype
            # Use the determined model_torch_dtype for loading
            model_mps = AutoModelForCausalLM.from_pretrained(model_id_for_mps, torch_dtype=model_torch_dtype, local_files_only=True)
            model_load_time_cpu_s = time.perf_counter() - cpu_load_start_time
            rss_after_cpu_load_mb = get_rss_usage_mb()
            print(f"Model '{model_id_for_mps}' + Tokenizer CPU load (from cache): {model_load_time_cpu_s:.3f}s. RSS after CPU load: {rss_after_cpu_load_mb:.2f} MB")

            # --- Create/Merge GenerationConfig AFTER model is loaded ---
            model_default_gen_config_dict = model_mps.generation_config.to_dict()
            merged_gen_params = model_default_gen_config_dict.copy()
            merged_gen_params.update(user_generation_params)
            if not merged_gen_params.get("do_sample", False):
                merged_gen_params["do_sample"] = False
                merged_gen_params.pop("temperature", None)
                merged_gen_params.pop("top_k", None)
                merged_gen_params.pop("top_p", None)
                merged_gen_params["num_beams"] = merged_gen_params.get("num_beams", 1)
            effective_benchmark_gen_config = GenerationConfig(**merged_gen_params)
            print(f"INFO: Effective GenerationConfig for {model_id_for_mps} (model defaults + user config): {effective_benchmark_gen_config.to_dict()}")
            try:
                gen_config_filename = os.path.join(abs_results_dir_param, f"generation_config_{model_id_for_mps.replace('/', '_')}_{timestamp}.json")
                save_generation_config(effective_benchmark_gen_config, gen_config_filename, seed_used=effective_seed_to_use)
                print(f"Effective GenerationConfig for this model saved to {gen_config_filename}")
            except Exception as e_gen_conf_save:
                print(f"Warning: Could not save effective GenerationConfig for {model_id_for_mps}: {e_gen_conf_save}")

            torch.mps.synchronize()
            move_start_time = time.perf_counter()
            print(f"Moving model to {device_str}...");
            model_mps.to(device_str) # The model is already loaded with the target dtype, this moves it to MPS
            model_mps.eval()
            torch.mps.synchronize()
            model_move_to_device_time_s = time.perf_counter() - move_start_time
            torch.mps.empty_cache(); gc.collect()
            rss_after_device_move_mb = get_rss_usage_mb()
            print(f"Model '{model_id_for_mps}' move to {device_str}: {model_move_to_device_time_s:.3f}s. RSS after device move: {rss_after_device_move_mb:.2f} MB")

            total_model_load_time_s = model_load_time_cpu_s + (model_move_to_device_time_s if model_move_to_device_time_s is not None else 0.0)

            print("Global MPS warm-up for the model...")
            warmup_gen_config = GenerationConfig(max_new_tokens=16, do_sample=False)
            for w_prompt_idx, w_prompt_text in enumerate(warm_prompts):
                w_inputs = tokenizer_mps(w_prompt_text, return_tensors="pt").to(device_str)
                torch.mps.synchronize()
                with torch.inference_mode():
                    _ = model_mps.generate(input_ids=w_inputs.input_ids, attention_mask=w_inputs.attention_mask, generation_config=warmup_gen_config)
                torch.mps.synchronize()
            print("Global MPS warm-up complete.")

            for current_prompt_text in prompt_list:
                print(f"--- MPS Prompt: '{current_prompt_text[:50]}...' ---")
                single_prompt_run_results = benchmark_model_on_prompt_mps(
                    model_mps, tokenizer_mps, current_prompt_text,
                    effective_benchmark_gen_config,
                    benchmark_dtype_str_for_model, # Pass the actual dtype string used
                    effective_seed_to_use
                )
                single_prompt_run_results.update({
                    "model_id": model_id_for_mps,
                    "accelerator_used": device_str.upper(),
                    "benchmark_dtype": benchmark_dtype_str_for_model, # Report the actual dtype
                    "quantization_method": "None",
                    "num_global_warmup_runs": len(warm_prompts) if warm_prompts else 0,
                    "model_load_time_s": round(total_model_load_time_s, 3),
                    "model_load_cpu_s": round(model_load_time_cpu_s, 3),
                    "model_move_to_device_time_s": round(model_move_to_device_time_s, 3) if model_move_to_device_time_s is not None else None,
                    "rss_after_load_mb": rss_after_cpu_load_mb,
                    "rss_after_device_move_mb": rss_after_device_move_mb,
                })
                all_prompt_results.append(single_prompt_run_results)
                # ... (save incremental results) ...
                try:
                    with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
                except Exception as e_write_inc_mps: print(f"ERROR writing MPS results to {output_filename_param} (incremental): {e_write_inc_mps}")


        except LocalEntryNotFoundError as e_local_mps:
            # ... (error handling, include benchmark_dtype_str_for_model and seed) ...
            print(f"FATAL ERROR for MPS model {model_id_for_mps}: Could not load model from local cache. {e_local_mps}")
            import traceback; traceback.print_exc()
            all_prompt_results.append({
                "model_id": model_id_for_mps, "status": "load_from_cache_failed",
                "error_message": str(e_local_mps),
                "accelerator_used": device_str.upper(), "benchmark_dtype": benchmark_dtype_str_for_model, "quantization_method": "None",
                "seed_used": effective_seed_to_use,
            })
        except Exception as e_model_scope_mps:
            # ... (error handling, include benchmark_dtype_str_for_model and seed) ...
            print(f"FATAL ERROR during MPS setup/benchmark for {model_id_for_mps}: {e_model_scope_mps}")
            import traceback; traceback.print_exc()
            all_prompt_results.append({
                "model_id": model_id_for_mps, "status": "load_or_setup_failed",
                "error_message": str(e_model_scope_mps),
                "accelerator_used": device_str.upper(), "benchmark_dtype": benchmark_dtype_str_for_model, "quantization_method": "None",
                "seed_used": effective_seed_to_use,
            })
        finally:
            # ... (cleanup) ...
            print(f"Cleaning up MPS resources for {model_id_for_mps}...")
            del model_mps; del tokenizer_mps; del effective_benchmark_gen_config
            model_mps, tokenizer_mps, effective_benchmark_gen_config = None, None, None
            torch.mps.empty_cache(); gc.collect()
            print("MPS Cleanup complete.")
            try:
                with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
            except Exception as e_final_write_mps: print(f"ERROR writing MPS results to {output_filename_param} (final for model): {e_final_write_mps}")


    print(f"\nMPS Benchmark run complete. Results saved to {output_filename_param}")


if __name__ == "__main__":
    # ... (seed setting as before) ...
    print("MPS Benchmark script starting...")
    
    seed_to_actually_use = DEFAULT_SEED 
    if CONFIG_SEED_VALUE is not None:
        try:
            seed_to_actually_use = int(CONFIG_SEED_VALUE)
        except ValueError:
            print(f"Warning: Invalid seed value '{CONFIG_SEED_VALUE}' in config. Using default seed {DEFAULT_SEED}.")
    else:
        print(f"Info: No seed specified in config. Using default seed {DEFAULT_SEED} for reproducibility.")
    set_seed(seed_to_actually_use)

    # ... (rest of __main__ setup: timestamp, dirs, MPS check, permissions) ...
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    abs_results_dir = results_dir_from_config
    if not os.path.isabs(results_dir_from_config):
        abs_results_dir = os.path.join(script_dir, results_dir_from_config)

    if not torch.backends.mps.is_available():
        print("FATAL: MPS backend not available. Please install a compatible PyTorch 2.x build. Exiting.")
        os.makedirs(abs_results_dir, exist_ok=True)
        output_file_mps_error = os.path.join(abs_results_dir, f"{base_output_filename_cfg}_{timestamp}_MPS_UNAVAILABLE.json")
        error_data = [{"model_id": "N/A", "status": "mps_unavailable", "error_message": "MPS backend not available.", "seed_used": seed_to_actually_use}]
        try:
            with open(output_file_mps_error, "w") as f: json.dump(error_data, f, indent=4)
            print(f"MPS unavailability status saved to {output_file_mps_error}")
        except Exception as e_err_write: print(f"Could not write MPS unavailability status: {e_err_write}")
        sys.exit(1)

    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Effective user ID: {os.geteuid()} (0 is root)")
    print(f"Attempting to use results directory for MPS: {abs_results_dir}")
    try:
        os.makedirs(abs_results_dir, exist_ok=True)
        print(f"Ensured MPS results directory exists: {abs_results_dir}")
        test_file_path_mps = os.path.join(abs_results_dir, f".permission_test_mps_{timestamp}")
        with open(test_file_path_mps, "w") as test_f: test_f.write("test_mps")
        os.remove(test_file_path_mps)
        print(f"Write permission confirmed for MPS results in: {abs_results_dir}")
    except OSError as e_dir_mps:
        print(f"ERROR: Could not create/write to MPS results directory {abs_results_dir}: {e_dir_mps}\nExiting.")
        sys.exit(1)

    output_file_mps = os.path.join(abs_results_dir, f"{base_output_filename_cfg}_{timestamp}.json")

    if os.geteuid() != 0:
        print("Script not running as root. Sudo will be used for powermetrics (MPS).")


    run_full_benchmark_mps(
        output_filename_param=output_file_mps,
        abs_results_dir_param=abs_results_dir,
        effective_seed_to_use=CURRENT_SEED_USED
    )