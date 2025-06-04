# mac/benchmark/benchmark_mps.py
import os
import json
import gc
import threading
import time # Keep for time.strftime, but use time.perf_counter for durations
import subprocess
import signal
import yaml
import tempfile
import sys

import torch
import psutil
from huggingface_hub import login, snapshot_download, HfFolder
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_generation_config(generation_config: GenerationConfig, filename="generation_config.json"):
    """Saves the generation config to a JSON file."""
    with open(filename, "w") as f:
        json.dump(generation_config.to_dict(), f, indent=4)


script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

if not os.path.exists(config_path):
    print(f"FATAL: Configuration file not found at {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

model_list = cfg["models"]
warm_prompts = cfg["warm_prompts"]
prompt_list = cfg["prompt_list"]

# --- GenerationConfig Setup ---
# Create a base dictionary for generation parameters
generation_params = {
    "max_new_tokens": cfg["generation"]["max_new_tokens"],
    "do_sample": cfg["generation"]["do_sample"],
    # Only include sampling-specific parameters if do_sample is True
}

if generation_params["do_sample"]:
    # Add sampling parameters from config if they exist
    if "temperature" in cfg["generation"]:
        generation_params["temperature"] = cfg["generation"]["temperature"]
    if "top_k" in cfg["generation"]:
        generation_params["top_k"] = cfg["generation"]["top_k"]
    if "top_p" in cfg["generation"]:
        generation_params["top_p"] = cfg["generation"]["top_p"]
    # Add other sampling params as needed: repetition_penalty, etc.
else:
    # If not sampling, ensure sampling-specific params that might cause warnings are not set
    # or are set to values that don't conflict.
    # For example, some libraries might still warn if top_k is set even if do_sample is False.
    # It's often cleaner to just not pass them if do_sample is False.
    # However, GenerationConfig might have defaults. We will let it use its defaults
    # if not do_sample, and only explicitly pass what's needed for greedy.
    pass # Let GenerationConfig handle defaults for non-sampling

gen_config = GenerationConfig(**generation_params)

# --- Save the effective GenerationConfig for reference (optional) ---
# def save_effective_generation_config(gen_cfg_obj, filename="generation_config_mps_effective.json"):
#     try:
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
#         with open(filename, "w") as f:
#             json.dump(gen_cfg_obj.to_dict(), f, indent=4)
#         print(f"Effective MPS GenerationConfig saved to {filename}")
#     except Exception as e:
#         print(f"Warning: Could not save effective MPS GenerationConfig: {e}")
# save_effective_generation_config(gen_config, os.path.join(script_dir, "results", "generation_config_mps_effective.json"))


num_runs_cfg = cfg["sampling"]["num_runs_per_prompt"] # Renamed for clarity
rss_interval_cfg = cfg["sampling"]["rss_sampler_interval_seconds"]
temp_samples_cfg = cfg["sampling"]["temperature_samples"]
temp_delay_cfg = cfg["sampling"]["temperature_delay_seconds"]

results_dir_from_config = cfg["output"]["results_directory"]
base_output_filename_cfg = cfg["output"]["base_output_filename"] # Renamed for clarity


save_generation_config(gen_config, os.path.join(script_dir, "results/generation_config.json"))

# --- Sudo Helper Function ---
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

# --- Helper function to run powermetrics for a short duration and get average power ---
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
            except Exception: pass # Ignore errors during cleanup communicate
        if pm_stderr_data and pm_stderr_data.strip(): # Check if there's actual stderr content
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
                    except ValueError: pass # Ignore lines that can't be parsed
                    except Exception: pass # Ignore other parsing errors
            if power_readings:
                avg_power_w = round(sum(power_readings) / len(power_readings), 2)
    except Exception as e:
        print(f"Error during short powermetrics sample (prefix: {prefix}): {e}")
    finally:
        if pm_proc and pm_proc.poll() is None: pm_proc.kill(); pm_proc.wait()
        if temp_pm_file_path and os.path.exists(temp_pm_file_path):
            try: os.remove(temp_pm_file_path)
            except OSError: pass # Ignore if removal fails (e.g., already deleted)
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
        return round(sum(readings) / len(readings), 2) if readings else None
    except FileNotFoundError: print(f"Warning: 'smctemp' command not found for '{param}'. Temp will be null."); return None
    except subprocess.CalledProcessError as e: print(f"Warning: 'smctemp {param}' failed: {e}. Temp will be null."); return None
    except Exception as e: print(f"Warning: Error getting temp '{param}': {e}. Temp will be null."); return None

def get_gpu_temperature(samples=3, delay=0.1): return get_temperature_with_smctemp("-g", samples, delay)
def get_cpu_temperature(samples=3, delay=0.1): return get_temperature_with_smctemp("-c", samples, delay)

def benchmark_model_on_prompt_mps(model, tokenizer, prompt, effective_gen_config: GenerationConfig, dtype_str: str):
    results = {}
    device_str = "mps" # Already checked availability
    is_mps_device = True # Assuming this function is only called if MPS is available

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device_str)
        torch.mps.synchronize()
        input_tokens = inputs.input_ids.shape[1]
        process = psutil.Process(os.getpid())
        peak_host_mem_during_prompt_mb = 0 # Tracks peak host RSS for this prompt's runs
        peak_mps_alloc_during_prompt_mb = 0   # Tracks peak MPS allocated for this prompt

        # --- TTFT Runs ---
        ttft_runs_ms = []
        print("Running TTFT measurements...")
        # Create a specific GenerationConfig for TTFT (1 token, no sampling)
        ttft_gen_config = GenerationConfig(max_new_tokens=1, do_sample=False)
        for _ in range(num_runs_cfg):
            torch.mps.synchronize()
            start_ttft = time.perf_counter()
            with torch.inference_mode():
                _ = model.generate(generation_config=ttft_gen_config, **inputs)
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

        print(f"Running {num_runs_cfg} full generation runs...") # MLX-style output
        for i in range(num_runs_cfg):
            print(f"  Run {i+1}/{num_runs_cfg} for prompt '{prompt[:30]}...'") # MLX-style output
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
                    outputs = model.generate(generation_config=effective_gen_config, return_dict_in_generate=True, output_scores=False, **inputs) # output_scores=False if not needed
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
                    if pm_proc.poll() is None:
                        pm_proc.send_signal(signal.SIGINT)
                        try: pm_proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            # print(f"Warning: Powermetrics (MPS run {i+1}, PID {current_pm_pid}) did not terminate after SIGINT. Sending SIGTERM.")
                            if pm_proc.poll() is None: pm_proc.send_signal(signal.SIGTERM)
                            try: pm_proc.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                print(f"Warning: Powermetrics (MPS run {i+1}, PID {current_pm_pid}) did not terminate after SIGTERM. Sending SIGKILL.")
                                if pm_proc.poll() is None: pm_proc.kill(); pm_proc.wait(timeout=2)
                    try:
                        if pm_proc.stderr and not pm_proc.stderr.closed:
                           _, pm_stderr_data = pm_proc.communicate(timeout=2)
                    except Exception as e_comm:
                        print(f"Error during final communicate for main powermetrics (MPS run {i+1}, PID {current_pm_pid}): {e_comm}")

                    if pm_stderr_data and pm_stderr_data.strip():
                        print(f"Main Powermetrics stderr (MPS run {i+1}, PID {current_pm_pid}): {pm_stderr_data.strip()}")

                power_readings_w = []
                if pm_proc and temp_pm_file_path and os.path.exists(temp_pm_file_path):
                    with open(temp_pm_file_path, "r") as f_in:
                        for line in f_in:
                            try:
                                if "Combined Power" in line and "mW" in line and ":" in line:
                                    value_part = line.split(":")[1].strip(); numeric_value_str = value_part.split("mW")[0].strip()
                                    power_mw = float(numeric_value_str); power_w = power_mw / 1000.0
                                    power_readings_w.append(power_w)
                            except ValueError:
                                print(f"Warning: Could not parse power value from line (MPS run {i+1}): '{line.strip()}'")
                            except Exception as e_parse:
                                 print(f"Warning: Unexpected error parsing power line (MPS run {i+1}) '{line.strip()}': {e_parse}")

                energy_j_this_run = None
                if power_readings_w and inference_duration_s > 0:
                    avg_power_w_this_run = sum(power_readings_w) / len(power_readings_w)
                    energy_j_this_run = avg_power_w_this_run * inference_duration_s
                energy_consumption_j_runs.append(energy_j_this_run)

                if i == 0: # Only for the first full run
                    actual_output_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                    generated_text_preview = tokenizer.decode(actual_output_ids, skip_special_tokens=True)[:100] + "..."
                    actual_output_tokens = len(actual_output_ids)

                if rss_sampler_thread and rss_sampler_thread.get("stop_event"):
                    rss_sampler_thread["stop_event"].set()
                    time.sleep(rss_interval_cfg + 0.1)
                    current_run_peak_rss_mb = rss_sampler_thread["peak_rss"] / (1024**2)
                    if current_run_peak_rss_mb > peak_host_mem_during_prompt_mb:
                        peak_host_mem_during_prompt_mb = current_run_peak_rss_mb

                torch.mps.empty_cache()
                gc.collect()

            except Exception as e_run:
                print(f"ERROR within MPS benchmark run {i+1} for prompt '{prompt[:30]}...': {e_run}")
                import traceback; traceback.print_exc()
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

        avg_time_ms = round(sum(full_run_times_ms) / len(full_run_times_ms), 3) if full_run_times_ms else 0
        tokens_per_sec = round(actual_output_tokens / (avg_time_ms / 1000.0), 2) if avg_time_ms > 0 and actual_output_tokens > 0 else 0

        valid_energy_runs = [e for e in energy_consumption_j_runs if e is not None]
        avg_energy_consumption_j = round(sum(valid_energy_runs) / len(valid_energy_runs), 4) if valid_energy_runs else None
        avg_power_during_inference_w = None
        if avg_energy_consumption_j is not None and avg_time_ms > 0:
            avg_power_during_inference_w = round(avg_energy_consumption_j / (avg_time_ms / 1000.0), 2)

        cpu_temp_delta_c = round(cpu_temp_after_c - cpu_temp_before_c, 2) if cpu_temp_before_c is not None and cpu_temp_after_c is not None else None
        gpu_temp_delta_c = round(gpu_temp_after_c - gpu_temp_before_c, 2) if gpu_temp_before_c is not None and gpu_temp_after_c is not None else None

        peak_host_mem_during_prompt_mb = round(peak_host_mem_during_prompt_mb, 2) if peak_host_mem_during_prompt_mb > 0 else None
        peak_mps_alloc_during_prompt_mb = round(peak_mps_alloc_during_prompt_mb, 2) if peak_mps_alloc_during_prompt_mb > 0 else None


        results = {
            "prompt": prompt, "status": "success", "error_message": None,
            "input_tokens": input_tokens, "output_tokens": actual_output_tokens,
            "avg_time_ms": avg_time_ms, "tokens_per_sec": tokens_per_sec,
            "avg_energy_consumption_j": avg_energy_consumption_j,
            "avg_power_during_inference_w": avg_power_during_inference_w,
            "power_before_prompt_w": power_before_prompt_w,
            "power_after_prompt_w": power_after_prompt_w,
            "ttft_ms_avg": ttft_ms_avg,
            "output_text_preview": generated_text_preview,
            "cpu_temp_before_c": cpu_temp_before_c, "cpu_temp_after_c": cpu_temp_after_c, "cpu_temp_delta_c": cpu_temp_delta_c,
            "gpu_temp_before_c": gpu_temp_before_c, "gpu_temp_after_c": gpu_temp_after_c, "gpu_temp_delta_c": gpu_temp_delta_c,
            "mps_alloc_before_mb": round(mps_alloc_before_all_runs_mb, 2),
            "mps_resv_before_mb": round(mps_resv_before_all_runs_mb, 2),
            "rss_before_mb": round(rss_before_all_runs_mb, 2),
            "mps_alloc_after_mb": round(mps_alloc_after_all_runs_mb, 2),
            "mps_resv_after_mb": round(mps_resv_after_all_runs_mb, 2),
            "rss_after_mb": round(rss_after_all_runs_mb, 2),
            "peak_rss_during_all_runs_mb": peak_host_mem_during_prompt_mb, # Renamed for consistency with MLX output if needed
            "peak_mps_mb": peak_mps_alloc_during_prompt_mb, # Peak allocated MPS memory during this prompt's runs
            "peak_host_memory_mb": peak_host_mem_during_prompt_mb, # For direct comparison if you rename peak_rss... above
        }

    except Exception as e_outer:
        print(f"MAJOR ERROR during MPS benchmark for prompt '{prompt[:50]}...': {e_outer}")
        import traceback; traceback.print_exc()
        results = {"prompt": prompt, "status": "failed", "error_message": str(e_outer)}
    return results


def download_model_if_needed(model_id: str, token: str = None):
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

def run_full_benchmark_mps(output_filename_param):
    if not torch.backends.mps.is_available():
        print("FATAL: MPS backend not available. Please install a compatible PyTorch 2.x build. Exiting MPS benchmark.")
        # Write an error status to the output file if possible
        error_result = [{"status": "mps_unavailable", "error_message": "MPS backend not available."}]
        try:
            os.makedirs(os.path.dirname(output_filename_param), exist_ok=True)
            with open(output_filename_param, "w") as f: json.dump(error_result, f, indent=4)
        except Exception as e_write:
            print(f"Could not write MPS unavailable status to {output_filename_param}: {e_write}")
        return # Exit this function

    print("Validating sudo privileges for powermetrics (MPS Benchmark)...")
    if not ensure_sudo_active(interactive_prompt_timeout=30):
        print("WARNING: Sudo privileges could not be obtained for MPS. Powermetrics will be skipped.")
    else:
        print("Sudo privileges validated for upcoming powermetrics calls (MPS).")

    all_prompt_results = []
    device_str = "mps"
    benchmark_dtype = torch.float16 # Standard for MPS benchmarks
    benchmark_dtype_str = str(benchmark_dtype)

    print(f"--- Running MPS Benchmark on device: {device_str} with dtype: {benchmark_dtype_str} ---")
    print(f"--- Output will be saved to: {output_filename_param} ---")

    hf_token_from_config = cfg.get("huggingface_token")
    try:
        if hf_token_from_config and hf_token_from_config.strip():
            login(token=hf_token_from_config); print("Logged in to Hugging Face Hub (token from config).")
        else:
            if HfFolder.get_token() is not None: print("Found existing Hugging Face login token.")
            else: print("No HF token in config or system. Proceeding with anonymous access.")
    except Exception as e_login: print(f"Warning: HF login/token check failed: {e_login}")

    for model_id_for_mps in model_list:
        print(f"\n{'='*20} Preparing MPS Model: {model_id_for_mps} {'='*20}")

        download_successful = download_model_if_needed(model_id_for_mps, token=hf_token_from_config)
        if not download_successful:
            print(f"Skipping MPS benchmark for model {model_id_for_mps} due to download failure.")
            all_prompt_results.append({
                "model_id": model_id_for_mps, "status": "download_failed",
                "error_message": f"Failed to download/cache model files for {model_id_for_mps}.",
                "device": device_str, "dtype": benchmark_dtype_str
            })
            # Save intermediate results
            try:
                with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
            except Exception as e_write_dl_fail: print(f"ERROR writing to {output_filename_param} (after MPS download fail): {e_write_dl_fail}")
            continue

        print(f"\n{'='*10} Benchmarking MPS Model (from cache): {model_id_for_mps} {'='*10}")
        model_mps, tokenizer_mps = None, None
        model_load_time_cpu_s, model_move_to_device_time_s = 0.0, 0.0
        rss_after_cpu_load_mb, rss_after_device_move_mb = 0.0, 0.0

        try:
            print(f"Loading tokenizer {model_id_for_mps} to CPU (from cache)...")
            load_cpu_start_time = time.perf_counter()
            tokenizer_mps = AutoTokenizer.from_pretrained(model_id_for_mps, use_fast=True, local_files_only=True)
            print(f"Loading model {model_id_for_mps} to CPU (dtype={benchmark_dtype_str}) (from cache)...")
            # Load model to CPU first
            model_mps = AutoModelForCausalLM.from_pretrained(model_id_for_mps, torch_dtype=benchmark_dtype, local_files_only=True)
            model_load_time_cpu_s = time.perf_counter() - load_cpu_start_time
            rss_after_cpu_load_mb = get_rss_usage_mb()
            print(f"Model '{model_id_for_mps}' CPU load (from cache): {model_load_time_cpu_s:.2f}s. RSS after CPU load: {rss_after_cpu_load_mb:.2f} MB")

            torch.mps.synchronize() # Ensure CPU work is done if any MPS interaction happened before
            move_start_time = time.perf_counter()
            print(f"Moving model to {device_str}...");
            model_mps.to(device_str)
            model_mps.eval()
            torch.mps.synchronize() # Wait for .to(device_str) to complete
            model_move_to_device_time_s = time.perf_counter() - move_start_time
            torch.mps.empty_cache(); gc.collect()
            rss_after_device_move_mb = get_rss_usage_mb()
            print(f"Model '{model_id_for_mps}' move to {device_str}: {model_move_to_device_time_s:.2f}s. RSS after device move: {rss_after_device_move_mb:.2f} MB")

            print("Global MPS warm-up for the model...")
            for w_prompt_idx, w_prompt_text in enumerate(warm_prompts):
                w_inputs = tokenizer_mps(w_prompt_text, return_tensors="pt").to(device_str)
                torch.mps.synchronize()
                with torch.inference_mode():
                    _ = model_mps.generate(**w_inputs, max_new_tokens=16, do_sample=False) # Simple warm-up
                torch.mps.synchronize()
            print("Global MPS warm-up complete.")

            for current_prompt_text in prompt_list:
                print(f"--- MPS Prompt: '{current_prompt_text[:50]}...' ---") # Clearer prompt origin
                single_prompt_run_results = benchmark_model_on_prompt_mps(
                    model_mps, tokenizer_mps, current_prompt_text, gen_config, benchmark_dtype_str
                )
                single_prompt_run_results.update({
                    "model_id": model_id_for_mps,
                    "device": device_str,
                    "dtype": benchmark_dtype_str,
                    "model_load_time_cpu_s": round(model_load_time_cpu_s, 3),
                    "model_move_to_device_time_s": round(model_move_to_device_time_s, 3),
                    "rss_after_cpu_load_mb": round(rss_after_cpu_load_mb, 2),
                    "rss_after_device_move_mb": round(rss_after_device_move_mb, 2),
                })
                all_prompt_results.append(single_prompt_run_results)
                # Save incrementally
                try:
                    with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
                except Exception as e_write_inc_mps: print(f"ERROR writing MPS results to {output_filename_param} (incremental): {e_write_inc_mps}")

        except LocalEntryNotFoundError as e_local_mps:
            print(f"FATAL ERROR for MPS model {model_id_for_mps}: Could not load model from local cache. {e_local_mps}")
            import traceback; traceback.print_exc()
            all_prompt_results.append({
                "model_id": model_id_for_mps, "status": "load_from_cache_failed",
                "error_message": str(e_local_mps), "device": device_str, "dtype": benchmark_dtype_str
            })
        except Exception as e_model_scope_mps:
            print(f"FATAL ERROR during MPS setup/benchmark for {model_id_for_mps}: {e_model_scope_mps}")
            import traceback; traceback.print_exc()
            all_prompt_results.append({
                "model_id": model_id_for_mps, "status": "load_or_setup_failed",
                "error_message": str(e_model_scope_mps), "device": device_str, "dtype": benchmark_dtype_str
            })
        finally:
            print(f"Cleaning up MPS resources for {model_id_for_mps}...")
            del model_mps
            del tokenizer_mps
            model_mps, tokenizer_mps = None, None
            torch.mps.empty_cache()
            gc.collect()
            print("MPS Cleanup complete.")
            # Final save for this model
            try:
                with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
            except Exception as e_final_write_mps: print(f"ERROR writing MPS results to {output_filename_param} (final for model): {e_final_write_mps}")

    print(f"\nMPS Benchmark run complete. Results saved to {output_filename_param}")

if __name__ == "__main__":
    print("MPS Benchmark script starting...")
    # Moved MPS availability check here so script can start and report error if needed
    if not torch.backends.mps.is_available():
        print("FATAL: MPS backend not available. Please install a compatible PyTorch 2.x build. Exiting.")
        # Attempt to write a minimal error output if possible
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        abs_results_dir = results_dir_from_config
        if not os.path.isabs(results_dir_from_config):
            abs_results_dir = os.path.join(script_dir, results_dir_from_config)
        os.makedirs(abs_results_dir, exist_ok=True) # Try to create dir
        output_file_mps_error = os.path.join(abs_results_dir, f"{base_output_filename_cfg}_{timestamp}_MPS_UNAVAILABLE.json")
        error_data = [{"model_id": "N/A", "status": "mps_unavailable", "error_message": "MPS backend not available."}]
        try:
            with open(output_file_mps_error, "w") as f: json.dump(error_data, f, indent=4)
            print(f"MPS unavailability status saved to {output_file_mps_error}")
        except Exception as e_err_write:
            print(f"Could not write MPS unavailability status: {e_err_write}")
        sys.exit(1) # Exit if MPS is not available

    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Effective user ID: {os.geteuid()} (0 is root)")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    abs_results_dir = results_dir_from_config
    if not os.path.isabs(results_dir_from_config):
        abs_results_dir = os.path.join(script_dir, results_dir_from_config)

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

    run_full_benchmark_mps(output_filename_param=output_file_mps)