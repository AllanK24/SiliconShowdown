# mac/benchmark/benchmark_mlx.py
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
import statistics # For stdev
import random     # For seeding
import numpy as np  # For seeding

import mlx.core as mx
# import mlx.nn as nn # Not directly used in this script's logic
from mlx_lm import load as mlx_load_model
from mlx_lm import generate as mlx_generate_text
from huggingface_hub import snapshot_download, HfFolder # Added HfFolder for token check
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

import psutil

# --- Configuration & Globals ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_SEED = 42
CURRENT_SEED_USED = DEFAULT_SEED
DEFAULT_MLX_DTYPE_NOTE = "mlx_internal (float16/bfloat16)"

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path_shared = os.path.join(script_dir, "config.yaml")

if not os.path.exists(config_path_shared):
    print(f"FATAL: Shared configuration file not found at {config_path_shared}")
    sys.exit(1)

with open(config_path_shared, "r") as f:
    cfg = yaml.safe_load(f)

# --- Seed Function ---
def set_seed(seed_value: int):
    global CURRENT_SEED_USED
    random.seed(seed_value)
    np.random.seed(seed_value)
    mx.random.seed(seed_value)
    CURRENT_SEED_USED = seed_value
    print(f"INFO: Random seeds set to {seed_value} for Python, NumPy, and MLX.")

# --- Parse Config ---
reproducibility_cfg = cfg.get("reproducibility", {})
CONFIG_SEED_VALUE = reproducibility_cfg.get("seed")

model_configs_from_yaml = cfg.get("models_mlx", cfg["models"])
warm_prompts = cfg["warm_prompts"]
prompt_list = cfg["prompt_list"]

user_gen_params_from_config = {
    "max_tokens": cfg["generation"]["max_new_tokens"],
}
# Store do_sample from config for logic, but don't pass it to mlx_generate_text
_do_sample_config = cfg["generation"]["do_sample"] # Temporary variable for logic

if _do_sample_config:
    user_gen_params_from_config["temp"] = cfg.get("generation", {}).get("temperature", 0.7)
    if "top_p" in cfg["generation"]:
        user_gen_params_from_config["top_p"] = cfg["generation"]["top_p"]
else:
    user_gen_params_from_config["temp"] = 0.0

num_timed_runs_per_prompt_cfg = cfg["sampling"]["num_runs_per_prompt"]
rss_interval_cfg = cfg["sampling"]["rss_sampler_interval_seconds"]
temp_samples_cfg = cfg["sampling"]["temperature_samples"]
temp_delay_cfg = cfg["sampling"]["temperature_delay_seconds"]

results_dir_from_config = cfg["output"]["results_directory"]
base_output_filename_cfg = cfg["output"].get("base_output_filename_mlx", "mlx_benchmark_results")

def save_generation_params_mlx(gen_params_dict, filename="generation_params_mlx.json", seed_used=None):
    params_to_save = gen_params_dict.copy()
    if seed_used is not None:
        params_to_save["_note_seed_used_for_run"] = seed_used
    with open(filename, "w") as f:
        json.dump(params_to_save, f, indent=4)

# --- Helper Functions (Sudo, Powermetrics, RSS, Temp) ---
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
                      "--samplers", "cpu_power", "-n", str(num_samples_to_take),
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
                        if "Combined Power" in line and "mW" in line and ":" in line:
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

# --- MLX Specific Helpers ---
def get_token_count(tokenizer, text: str):
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        print(f"Warning: Could not determine token count via tokenizer.encode: {e}. Falling back.")
        return len(text) // 4

# --- Download Function for MLX (Pre-caching with snapshot_download) ---
def download_mlx_model_if_needed(model_id: str, token: str = None): # ADDED THIS FUNCTION BACK
    """
    Ensures model files are in the Hugging Face cache. mlx_lm.load will then
    use these cached files for its specific loading/conversion process.
    """
    print(f"Ensuring model '{model_id}' files are downloaded to Hugging Face cache for MLX...")
    try:
        snapshot_path = snapshot_download(
            repo_id=model_id, local_files_only=False, resume_download=True, token=token,
            allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer.*", "*.bin", "*.gguf", "*.py", "*.md"] # Broader patterns
        )
        print(f"Model '{model_id}' files are available in HF cache via snapshot_download: {snapshot_path}")
        return True
    except HfHubHTTPError as e:
        print(f"Error downloading '{model_id}' via snapshot_download: {e}")
        if "401" in str(e): print("This might be a private model. Ensure a valid token is used or you are logged in via huggingface-cli.")
        elif "404" in str(e): print(f"Model or revision for '{model_id}' not found on Hugging Face Hub.")
        return False
    except Exception as e:
        print(f"Unexpected error ensuring model '{model_id}' is downloaded: {e}")
        import traceback; traceback.print_exc()
        return False

# --- Benchmark Function for a Single Prompt (MLX) ---
def benchmark_model_on_prompt_mlx(model, tokenizer, prompt: str,
                                  main_gen_params_for_call: dict, # Params excluding temp if it's 0.0
                                  seed_used: int,
                                  benchmark_dtype_note: str):
    results = {}
    input_tokens = get_token_count(tokenizer, prompt)
    process = psutil.Process(os.getpid())
    peak_rss_mb_overall = 0

    # --- TTFT Runs ---
    ttft_runs_ms = []
    ttft_max_tokens = 1
    print(f"Running TTFT measurements with max_tokens={ttft_max_tokens} (relying on default temp=0.0)...")
    for _ in range(num_timed_runs_per_prompt_cfg):
        start_ttft = time.perf_counter()
        _ = mlx_generate_text(
            model, tokenizer, prompt=prompt, verbose=False,
            max_tokens=ttft_max_tokens
            # temp is omitted for TTFT, relying on its default of 0.0 in mlx_generate_text
        )
        mx.eval()
        end_ttft = time.perf_counter()
        ttft_runs_ms.append((end_ttft - start_ttft) * 1000)
    ttft_ms_avg = round(sum(ttft_runs_ms) / len(ttft_runs_ms), 2) if ttft_runs_ms else 0.0
    print(f"TTFT avg: {ttft_ms_avg:.2f} ms")

    power_before_w = get_short_powermetrics_sample_w(duration_s=2, prefix="mlx_pre_prompt_pm_")
    run_times_ms = []
    energy_consumption_j_runs = []
    generated_text_preview = ""
    actual_output_tokens = 0

    rss_before_all_runs_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    cpu_temp_before_c = get_cpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
    gpu_temp_before_c = get_gpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)

    # main_gen_params_for_call is already prepared to omit temp if it's 0.0
    print(f"Running {num_timed_runs_per_prompt_cfg} full generation runs with effective call params: {main_gen_params_for_call}...")
    for i in range(num_timed_runs_per_prompt_cfg):
        print(f"  Run {i+1}/{num_timed_runs_per_prompt_cfg} for prompt '{prompt[:30]}...'")
        pm_proc, temp_pm_file_path, rss_sampler_thread = None, None, None
        try:
            sudo_ok_for_main_pm = ensure_sudo_active()
            if sudo_ok_for_main_pm:
                with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt", prefix=f"mlx_main_pm_run{i}_") as tmp_f:
                    temp_pm_file_path = tmp_f.name
                pm_command = ["sudo", "powermetrics", "-i", "1000", "--samplers", "cpu_power", "-o", temp_pm_file_path]
                pm_proc = subprocess.Popen(pm_command, stderr=subprocess.PIPE, text=True)
                time.sleep(0.5)
                if pm_proc.poll() is not None:
                    pm_stderr_early = pm_proc.stderr.read() if pm_proc.stderr else ""
                    print(f"Warning: Main Powermetrics (MLX run {i+1}, PID {pm_proc.pid if pm_proc else 'N/A'}) failed/exited early. Stderr: {pm_stderr_early}")
                    pm_proc = None
            else:
                print(f"Warning: Sudo not active for main powermetrics (MLX run {i+1}). Energy metrics will be skipped.")
                pm_proc = None
            rss_sampler_thread = start_rss_sampler(process, interval=rss_interval_cfg)

            start_full_run = time.perf_counter()
            full_generated_text = mlx_generate_text(model, tokenizer, prompt=prompt, verbose=False, **main_gen_params_for_call)
            mx.eval()
            end_full_run = time.perf_counter()
            inference_duration_s = end_full_run - start_full_run
            run_times_ms.append(inference_duration_s * 1000)

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
                except Exception as e_comm: print(f"Error during final communicate for main powermetrics (MLX run {i+1}, PID {current_pm_pid}): {e_comm}")
                if pm_stderr_data and pm_stderr_data.strip(): print(f"Main Powermetrics stderr (MLX run {i+1}, PID {current_pm_pid}): {pm_stderr_data.strip()}")

            power_readings_w = []
            if pm_proc and temp_pm_file_path and os.path.exists(temp_pm_file_path):
                with open(temp_pm_file_path, "r") as f_in:
                    for line in f_in:
                        try:
                            if "Combined Power" in line and "mW" in line and ":" in line:
                                value_part = line.split(":")[1].strip(); numeric_value_str = value_part.split("mW")[0].strip()
                                power_mw = float(numeric_value_str); power_w = power_mw / 1000.0
                                power_readings_w.append(power_w)
                        except ValueError: print(f"Warning: Could not parse power value from line (MLX run {i+1}): '{line.strip()}'")
                        except Exception as e_parse: print(f"Warning: Unexpected error parsing power line (MLX run {i+1}) '{line.strip()}': {e_parse}")
            energy_j_this_run = None
            if power_readings_w and inference_duration_s > 0:
                avg_power_w_this_run = sum(power_readings_w) / len(power_readings_w)
                energy_j_this_run = avg_power_w_this_run * inference_duration_s
            energy_consumption_j_runs.append(energy_j_this_run)

            if i == 0:
                generated_text_preview = full_generated_text[:100] + "..."
                if full_generated_text.startswith(prompt):
                    only_generated_part = full_generated_text[len(prompt):]
                else:
                    only_generated_part = full_generated_text
                    print("Warning: Generated text doesn't start with prompt. Output token count might be inaccurate.")
                actual_output_tokens = get_token_count(tokenizer, only_generated_part)

            if rss_sampler_thread and rss_sampler_thread.get("stop_event"):
                rss_sampler_thread["stop_event"].set()
                time.sleep(rss_interval_cfg + 0.1)
                current_run_peak_rss_mb = rss_sampler_thread["peak_rss"] / (1024**2)
                if current_run_peak_rss_mb > peak_rss_mb_overall:
                    peak_rss_mb_overall = current_run_peak_rss_mb
            gc.collect()
        except Exception as e_run:
            print(f"ERROR within MLX benchmark run {i+1} for prompt '{prompt[:30]}...': {e_run}")
            import traceback; traceback.print_exc()
            run_times_ms.append(None)
            energy_consumption_j_runs.append(None)
        finally:
            if rss_sampler_thread and rss_sampler_thread.get("stop_event") and not rss_sampler_thread["stop_event"].is_set():
                rss_sampler_thread["stop_event"].set()
            if pm_proc and pm_proc.poll() is None: pm_proc.kill(); pm_proc.wait()
            if temp_pm_file_path and os.path.exists(temp_pm_file_path):
                try: os.remove(temp_pm_file_path)
                except OSError as e_rm: print(f"Warning: Could not remove temp powermetrics file {temp_pm_file_path}: {e_rm}")

    rss_after_all_runs_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    cpu_temp_after_c = get_cpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
    gpu_temp_after_c = get_gpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
    power_after_w = get_short_powermetrics_sample_w(duration_s=2, prefix="mlx_post_prompt_pm_")

    valid_run_times_ms = [t for t in run_times_ms if t is not None]
    avg_time_ms = round(sum(valid_run_times_ms) / len(valid_run_times_ms), 3) if valid_run_times_ms else 0.0
    
    runs_time_ms_rounded = [round(t, 3) for t in valid_run_times_ms]
    stddev_time_ms = None
    if len(runs_time_ms_rounded) > 1:
        stddev_time_ms = round(statistics.stdev(runs_time_ms_rounded), 3)
    elif len(runs_time_ms_rounded) == 1:
         stddev_time_ms = 0.0

    tokens_per_sec = round(actual_output_tokens / (avg_time_ms / 1000.0), 2) if avg_time_ms > 0 and actual_output_tokens > 0 else 0.0
    valid_energy_runs = [e for e in energy_consumption_j_runs if e is not None]
    avg_energy_consumption_j = round(sum(valid_energy_runs) / len(valid_energy_runs), 4) if valid_energy_runs else None
    avg_power_during_inference_w = round(avg_energy_consumption_j / (avg_time_ms / 1000.0), 2) if avg_energy_consumption_j is not None and avg_time_ms > 0 else None
    
    cpu_temp_increase_c = round(cpu_temp_after_c - cpu_temp_before_c, 1) if cpu_temp_before_c is not None and cpu_temp_after_c is not None else None
    gpu_temp_increase_c = round(gpu_temp_after_c - gpu_temp_before_c, 1) if gpu_temp_before_c is not None and gpu_temp_after_c is not None else None
    gpu_temp_avg_c = None
    if gpu_temp_before_c is not None and gpu_temp_after_c is not None:
        gpu_temp_avg_c = round((gpu_temp_before_c + gpu_temp_after_c) / 2.0, 1)
    elif gpu_temp_before_c is not None: gpu_temp_avg_c = gpu_temp_before_c
    elif gpu_temp_after_c is not None: gpu_temp_avg_c = gpu_temp_after_c
        
    peak_rss_mb_overall = round(peak_rss_mb_overall, 2) if peak_rss_mb_overall > 0 else None

    results = {
        "model_id": None, 
        "benchmark_dtype": benchmark_dtype_note, 
        "batch_size": 1, 
        "num_global_warmup_runs": None, 
        "num_timed_runs_per_prompt": num_timed_runs_per_prompt_cfg,
        "model_load_time_s": None, 
        "accelerator_used": "MLX_GPU", 
        "quantization_method": "None", 
        "prompt": prompt,
        "status": "success",
        "error_message": None,
        "seed_used": seed_used,
        "input_tokens": input_tokens,
        "output_tokens": actual_output_tokens,
        "ttft_ms_avg": ttft_ms_avg,
        "avg_gpu_time_ms": avg_time_ms, # Renamed for consistency with Llama output's intent
        "stddev_gpu_time_ms": stddev_time_ms, 
        "tokens_per_sec": tokens_per_sec,
        "runs_gpu_time_ms": runs_time_ms_rounded, 
        "peak_gpu_memory_mb": None, 
        "peak_host_memory_mb": peak_rss_mb_overall,
        "rss_before_mb": round(rss_before_all_runs_mb, 2),
        "rss_after_mb": round(rss_after_all_runs_mb, 2),
        "temp_before_c": gpu_temp_before_c, 
        "temp_after_c": gpu_temp_after_c,   
        "avg_temp_c": gpu_temp_avg_c,       
        "temp_increase_c": gpu_temp_increase_c, 
        "cpu_temp_before_c": cpu_temp_before_c,
        "cpu_temp_after_c": cpu_temp_after_c,
        "cpu_temp_increase_c": cpu_temp_increase_c,
        "power_before_w": power_before_w, 
        "power_after_w": power_after_w,   
        "avg_power_w": avg_power_during_inference_w, 
        "avg_energy_consumption_j": avg_energy_consumption_j,
        "output_text_preview": generated_text_preview,
    }
    return results


# --- Main Benchmark Orchestration (MLX) ---
def run_full_benchmark_mlx(output_filename_param, abs_results_dir_param, effective_seed_to_use):
    print("Validating sudo privileges for powermetrics (MLX Benchmark)...")
    if not ensure_sudo_active(interactive_prompt_timeout=30):
        print("WARNING: Sudo privileges could not be obtained. Powermetrics will be skipped.")
    else:
        print("Sudo privileges validated for upcoming powermetrics calls.")

    all_prompt_results = []
    hf_token_from_config = cfg.get("huggingface_token")
    try:
        if hf_token_from_config and hf_token_from_config.strip():
            print("Using Hugging Face token from config for downloads if needed.")
        elif HfFolder.get_token() is not None: # Checks if token is available from login or env var
            print("Found existing Hugging Face login token for downloads if needed.")
        else:
            print("No HF token. Proceeding with anonymous access for downloads.")
    except Exception as e_token_check:
        print(f"Warning: HF token check failed: {e_token_check}")

    for model_config_item in model_configs_from_yaml:
        model_id_for_mlx = ""
        dtype_note_for_model = DEFAULT_MLX_DTYPE_NOTE

        if isinstance(model_config_item, str):
            model_id_for_mlx = model_config_item
        elif isinstance(model_config_item, dict):
            model_id_for_mlx = model_config_item.get("model_id")
            dtype_note_for_model = model_config_item.get("dtype_note_mlx", dtype_note_for_model)
            if not model_id_for_mlx:
                print(f"Warning: Skipping model entry due to missing 'model_id': {model_config_item}")
                continue
        else:
            print(f"Warning: Skipping invalid model entry: {model_config_item}")
            continue

        print(f"\n{'='*20} Preparing MLX Model: {model_id_for_mlx} (DType Note: {dtype_note_for_model}) {'='*20}")
        print(f"--- Running MLX Benchmark (DType Note: {dtype_note_for_model}) using SEED: {effective_seed_to_use} ---")
        
        download_successful = download_mlx_model_if_needed(model_id_for_mlx, token=hf_token_from_config)
        if not download_successful:
            all_prompt_results.append({
                "model_id": model_id_for_mlx, "benchmark_dtype": dtype_note_for_model,
                "status": "download_failed", "seed_used": effective_seed_to_use,
                "error_message": f"Failed to pre-download/cache model files for {model_id_for_mlx}.",
                "accelerator_used": "MLX_GPU", "quantization_method": "None", "batch_size": 1,
                "num_global_warmup_runs": len(warm_prompts) if warm_prompts else 0,
                "num_timed_runs_per_prompt": num_timed_runs_per_prompt_cfg
            })
            try:
                with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
            except Exception as e_write_dl_fail: print(f"ERROR writing to {output_filename_param} (after MLX download fail): {e_write_dl_fail}")
            continue

        model_mlx, tokenizer_mlx = None, None
        model_load_time_s = 0.0
        rss_after_load_mb = 0.0
        
        current_model_intended_gen_params = user_gen_params_from_config.copy()
        # Use _do_sample_config for logic when model-specific overrides are applied
        model_do_sample_logic = _do_sample_config

        if isinstance(model_config_item, dict) and "generation_params_mlx" in model_config_item:
            print(f"Applying model-specific generation_params_mlx for {model_id_for_mlx}")
            model_specific_overrides = model_config_item["generation_params_mlx"]
            current_model_intended_gen_params.update(model_specific_overrides)
            
            # If 'do_sample' was in model_specific_overrides, use that for logic
            if "do_sample" in model_specific_overrides:
                model_do_sample_logic = model_specific_overrides["do_sample"]

            if not model_do_sample_logic: # If final decision is not to sample
                current_model_intended_gen_params["temp"] = 0.0
            # If model_do_sample_logic is True and temp wasn't set by model_specific_overrides,
            # it would have already been set by user_gen_params_from_config or needs to be default cfg temp
            elif "temp" not in model_specific_overrides: 
                 current_model_intended_gen_params["temp"] = cfg.get("generation", {}).get("temperature", 0.7)
        
        current_model_intended_gen_params.pop("do_sample", None) # Clean up logic key
        
        print(f"Intended main generation parameters for {model_id_for_mlx} (before removing temp=0.0 for call): {current_model_intended_gen_params}")
        
        try:
            gen_params_timestamp = time.strftime("%Y%m%d-%H%M%S") # Unique timestamp for this file
            gen_params_filename = os.path.join(abs_results_dir_param, f"generation_params_mlx_{model_id_for_mlx.replace('/', '_')}_{gen_params_timestamp}.json")
            save_generation_params_mlx(current_model_intended_gen_params.copy(), gen_params_filename, seed_used=effective_seed_to_use)
            print(f"Intended MLX generation parameters saved to {gen_params_filename}")
        except Exception as e_gen_param_save:
            print(f"Warning: Could not save MLX generation parameters: {e_gen_param_save}")

        params_for_actual_generate_call = current_model_intended_gen_params.copy()
        if params_for_actual_generate_call.get("temp", 0.0) == 0.0:
            params_for_actual_generate_call.pop("temp", None)
        print(f"Parameters actually passed to mlx_generate_text for main runs: {params_for_actual_generate_call}")

        try:
            print(f"Loading MLX model and tokenizer {model_id_for_mlx}...")
            load_mlx_start_time = time.perf_counter()
            model_mlx, tokenizer_mlx = mlx_load_model(model_id_for_mlx)
            mx.eval(model_mlx.parameters())
            model_load_time_s = time.perf_counter() - load_mlx_start_time
            rss_after_load_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            print(f"MLX Model '{model_id_for_mlx}' load time: {model_load_time_s:.3f}s. RSS after load: {rss_after_load_mb:.2f} MB")

            num_actual_warmup_runs = len(warm_prompts) if warm_prompts else 0
            print(f"Global MLX warm-up for the model ({num_actual_warmup_runs} runs)...")
            warmup_max_tokens_val = 16
            for i in range(num_actual_warmup_runs):
                 w_prompt_text = warm_prompts[i % len(warm_prompts)]
                 _ = mlx_generate_text(
                     model_mlx, tokenizer_mlx, prompt=w_prompt_text, verbose=False,
                     max_tokens=warmup_max_tokens_val
                    )
                 mx.eval()
            print("Global MLX warm-up complete.")

            for current_prompt_text in prompt_list:
                print(f"--- MLX Prompt: '{current_prompt_text[:50]}...' ---")
                single_prompt_run_results = benchmark_model_on_prompt_mlx(
                    model_mlx, tokenizer_mlx, current_prompt_text,
                    params_for_actual_generate_call,
                    effective_seed_to_use,
                    dtype_note_for_model
                )
                single_prompt_run_results.update({
                    "model_id": model_id_for_mlx,
                    "num_global_warmup_runs": num_actual_warmup_runs,
                    "model_load_time_s": round(model_load_time_s, 3),
                    "model_load_cpu_s": None, 
                    "model_move_to_device_time_s": None, 
                    "rss_after_load_mb": round(rss_after_load_mb, 2),
                    "rss_after_device_move_mb": None, 
                })
                all_prompt_results.append(single_prompt_run_results)
                try:
                    with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
                except Exception as e_write_inc: print(f"ERROR writing MLX results (incremental): {e_write_inc}")

        except LocalEntryNotFoundError as e_local_mlx:
            print(f"FATAL ERROR for MLX model {model_id_for_mlx}: mlx_lm.load() failed. {e_local_mlx}")
            all_prompt_results.append({
                "model_id": model_id_for_mlx, "benchmark_dtype": dtype_note_for_model,
                "status": "mlx_load_failed", "seed_used": effective_seed_to_use,
                "error_message": str(e_local_mlx), "accelerator_used": "MLX_GPU", "quantization_method": "None",
                "batch_size": 1, "num_global_warmup_runs": len(warm_prompts) if warm_prompts else 0,
                "num_timed_runs_per_prompt": num_timed_runs_per_prompt_cfg
            })
        except Exception as e_model_scope_mlx:
            print(f"FATAL ERROR during MLX setup/benchmark for {model_id_for_mlx}: {e_model_scope_mlx}")
            import traceback; traceback.print_exc()
            all_prompt_results.append({
                "model_id": model_id_for_mlx, "benchmark_dtype": dtype_note_for_model,
                "status": "mlx_setup_or_run_failed", "seed_used": effective_seed_to_use,
                "error_message": str(e_model_scope_mlx), "accelerator_used": "MLX_GPU", "quantization_method": "None",
                "batch_size": 1, "num_global_warmup_runs": len(warm_prompts) if warm_prompts else 0,
                "num_timed_runs_per_prompt": num_timed_runs_per_prompt_cfg
            })
        finally:
            print(f"Cleaning up MLX resources for {model_id_for_mlx}...")
            del model_mlx; del tokenizer_mlx
            model_mlx, tokenizer_mlx = None, None
            gc.collect()
            print("MLX Cleanup complete.")
            try:
                with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
            except Exception as e_final_write_mlx: print(f"ERROR writing MLX results (final for model): {e_final_write_mlx}")

    print(f"\nMLX Benchmark run complete. Results saved to {output_filename_param}")

# --- Main Execution ---
if __name__ == "__main__":
    print("MLX Benchmark script starting...")
    seed_to_actually_use = DEFAULT_SEED
    if CONFIG_SEED_VALUE is not None:
        try: seed_to_actually_use = int(CONFIG_SEED_VALUE)
        except ValueError: print(f"Warning: Invalid seed value '{CONFIG_SEED_VALUE}'. Using default: {DEFAULT_SEED}.")
    else:
        print(f"Info: No seed specified. Using default: {DEFAULT_SEED} for reproducibility.")
    set_seed(seed_to_actually_use)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    abs_results_dir = results_dir_from_config
    if not os.path.isabs(results_dir_from_config):
        abs_results_dir = os.path.join(script_dir, results_dir_from_config)

    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Attempting to use results directory for MLX: {abs_results_dir}")
    try:
        os.makedirs(abs_results_dir, exist_ok=True)
        print(f"Ensured MLX results directory exists: {abs_results_dir}")
        test_file_path_mlx = os.path.join(abs_results_dir, f".permission_test_mlx_{timestamp}")
        with open(test_file_path_mlx, "w") as test_f: test_f.write("test_mlx")
        os.remove(test_file_path_mlx)
        print(f"Write permission confirmed for MLX results in: {abs_results_dir}")
    except OSError as e_dir:
        print(f"ERROR: Could not create/write to MLX results directory {abs_results_dir}: {e_dir}\nExiting.")
        sys.exit(1)

    output_file_mlx = os.path.join(abs_results_dir, f"{base_output_filename_cfg}_{timestamp}.json")
    if os.geteuid() != 0:
        print("Script not running as root. Sudo will be used for powermetrics if available.")

    run_full_benchmark_mlx(
        output_filename_param=output_file_mlx,
        abs_results_dir_param=abs_results_dir,
        effective_seed_to_use=CURRENT_SEED_USED
    )