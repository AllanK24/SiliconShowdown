# mac/benchmark/benchmark_mlx.py
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

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load_model # Renamed to avoid conflict
from mlx_lm import generate as mlx_generate_text # Renamed
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

import psutil # For RSS

# --- Configuration ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config_mlx.yaml") # Consider a separate config or sections in one
# For now, let's assume we adapt the existing config.yaml for MLX relevant parts
config_path_shared = os.path.join(script_dir, "config.yaml")

if not os.path.exists(config_path_shared):
    print(f"FATAL: Shared configuration file not found at {config_path_shared}")
    sys.exit(1)

with open(config_path_shared, "r") as f:
    cfg = yaml.safe_load(f)

# MLX specific models or use the same list?
# If using the same, ensure they are compatible with mlx-lm
model_list_from_config = cfg.get("models_mlx", cfg["models"]) # Allow overriding for MLX
warm_prompts = cfg["warm_prompts"]
prompt_list = cfg["prompt_list"]

# Generation parameters from config
# mlx_lm.generate doesn't use GenerationConfig object directly
# We'll pass max_tokens, temp, etc. as direct arguments.
max_new_tokens_cfg = cfg["generation"]["max_new_tokens"]
# mlx_lm.generate doesn't have a direct do_sample like HF transformers.
# Temperature > 0 implies sampling. temp=0.0 aims for deterministic.
# We will set temp based on do_sample. If do_sample is false, use temp=0.0.
do_sample_cfg = cfg["generation"]["do_sample"]
generation_temp_cfg = cfg.get("generation", {}).get("temperature", 0.7) # Default if not in config
if not do_sample_cfg:
    generation_temp_cfg = 0.0 # For deterministic output if do_sample is false

num_runs_per_prompt_cfg = cfg["sampling"]["num_runs_per_prompt"]
rss_interval_cfg = cfg["sampling"]["rss_sampler_interval_seconds"]
temp_samples_cfg = cfg["sampling"]["temperature_samples"]
temp_delay_cfg = cfg["sampling"]["temperature_delay_seconds"]

results_dir_from_config = cfg["output"]["results_directory"]
# Modify base output filename for MLX
base_output_filename_cfg = cfg["output"].get("base_output_filename_mlx", "benchmark_results_mlx")


# --- Sudo Helper Function (Identical to MPS version) ---
def ensure_sudo_active(interactive_prompt_timeout=60):
    # ... (Copy verbatim from your MPS script) ...
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
    return False # Should not be reached if logic is correct


# --- Powermetrics Helper (Identical to MPS version) ---
def get_short_powermetrics_sample_w(duration_s=2, interval_ms=1000, prefix="short_pm_"):
    # ... (Copy verbatim from your MPS script) ...
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


# --- RSS Sampler (Identical to MPS version) ---
def start_rss_sampler(proc, interval=0.05):
    # ... (Copy verbatim from your MPS script) ...
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
            time.sleep(interval) # Using standard time.sleep for thread delay
    thread = threading.Thread(target=_run, daemon=True); thread.start()
    return sampler


# --- Temperature Sampler (Identical to MPS version) ---
def get_temperature_with_smctemp(param, samples=3, delay=0.1):
    # ... (Copy verbatim from your MPS script) ...
    readings = []
    command = ["smctemp", param]
    try:
        for _ in range(samples):
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
            readings.append(float(output.strip()))
            time.sleep(delay) # Using standard time.sleep
        return round(sum(readings) / len(readings), 2) if readings else None
    except FileNotFoundError: print(f"Warning: 'smctemp' command not found for '{param}'. Temp will be null."); return None
    except subprocess.CalledProcessError as e: print(f"Warning: 'smctemp {param}' failed: {e}. Temp will be null."); return None
    except Exception as e: print(f"Warning: Error getting temp '{param}': {e}. Temp will be null."); return None

def get_gpu_temperature(samples=3, delay=0.1): return get_temperature_with_smctemp("-g", samples, delay) # Note: MLX uses GPU
def get_cpu_temperature(samples=3, delay=0.1): return get_temperature_with_smctemp("-c", samples, delay)

# --- MLX Specific Helper ---
def get_token_count(tokenizer, text: str):
    if hasattr(tokenizer, "encode"): # Standard Hugging Face tokenizer interface
        return len(tokenizer.encode(text))
    elif hasattr(tokenizer, "tokenize") and hasattr(tokenizer, "convert_tokens_to_ids"): # Some tokenizers
        return len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
    else:
        # Fallback for SentencePiece directly if tokenizer is the SPModel
        # This is a bit heuristic, mlx_lm tokenizer object should ideally expose a clear way
        try:
            return len(tokenizer.piece_to_id(tokenizer.encode_as_pieces(text)))
        except:
            print("Warning: Could not determine token count accurately. Falling back to character length / 4 (approx).")
            return len(text) // 4 # Very rough estimate

def benchmark_model_on_prompt_mlx(model, tokenizer, model_id_str: str, prompt: str):
    results = {}
    device_str = "mlx_gpu" # MLX uses GPU on Apple Silicon by default

    try:
        # Input tokenization
        # For MLX, tokenizer.encode might return a list of IDs directly
        input_ids_list = tokenizer.encode(prompt) # mlx-lm tokenizer should have .encode()
        input_tokens = len(input_ids_list)
        # No explicit .to(device) for input_ids with mlx_lm.generate, it handles it.

        process = psutil.Process(os.getpid())
        peak_rss_mb_overall = 0 # Tracks peak RSS across all runs for this prompt

        # --- TTFT Runs ---
        ttft_runs_ms = []
        print("Running TTFT measurements...")
        for _ in range(num_runs_per_prompt_cfg):
            # Ensure MLX device is idle before starting timer
            # Forcibly evaluate something small to ensure queue is clear if needed,
            # though typically not required before a fresh generate call.
            # mx.eval(mx.array([1])) # Optional small eval
            start_ttft = time.perf_counter()
            # Generate only one token for TTFT
            _ = mlx_generate_text(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
            mx.eval() # Crucial: wait for the generation of the first token to complete
            end_ttft = time.perf_counter()
            ttft_runs_ms.append((end_ttft - start_ttft) * 1000)
        ttft_ms_avg = round(sum(ttft_runs_ms) / len(ttft_runs_ms), 2) if ttft_runs_ms else 0.0
        print(f"TTFT avg: {ttft_ms_avg:.2f} ms")

        power_before_prompt_w = get_short_powermetrics_sample_w(duration_s=2, prefix="mlx_pre_prompt_pm_")

        full_run_times_ms = []
        energy_consumption_j_runs = []
        generated_text_preview = ""
        actual_output_tokens = 0 # To store the token count of the first full run's output

        # For other metrics, take before/after once around all runs
        rss_before_all_runs_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        cpu_temp_before_c = get_cpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
        gpu_temp_before_c = get_gpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg) # MLX uses GPU

        print(f"Running {num_runs_per_prompt_cfg} full generation runs...")
        for i in range(num_runs_per_prompt_cfg):
            print(f"  Run {i+1}/{num_runs_per_prompt_cfg} for prompt '{prompt[:30]}...'")
            pm_proc, temp_pm_file_path, rss_sampler_thread = None, None, None
            try:
                sudo_ok_for_main_pm = ensure_sudo_active()
                if sudo_ok_for_main_pm:
                    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt", prefix=f"mlx_main_pm_run{i}_") as tmp_f:
                        temp_pm_file_path = tmp_f.name
                    pm_command = ["sudo", "powermetrics", "-i", "1000", "--samplers", "cpu_power", "-o", temp_pm_file_path] # Using cpu_power for now
                    pm_proc = subprocess.Popen(pm_command, stderr=subprocess.PIPE, text=True)
                    time.sleep(0.5) # Allow powermetrics to start
                    if pm_proc.poll() is not None: # Check if it died early
                        pm_stderr_early = pm_proc.stderr.read() if pm_proc.stderr else ""
                        print(f"Warning: Main Powermetrics (MLX run {i+1}, PID: {pm_proc.pid if pm_proc else 'N/A'}) failed/exited early. Stderr: {pm_stderr_early}")
                        pm_proc = None # Mark as not running
                else:
                    print(f"Warning: Sudo not active for main powermetrics (MLX run {i+1}). Energy metrics will be skipped.")
                    pm_proc = None

                rss_sampler_thread = start_rss_sampler(process, interval=rss_interval_cfg)

                # mx.eval() here might not be strictly necessary if the device was idle
                start_full_run = time.perf_counter()
                full_generated_text = mlx_generate_text(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_new_tokens_cfg,
                    verbose=False
                )
                mx.eval() # CRITICAL: Ensure all MLX computations are finished
                end_full_run = time.perf_counter()
                inference_duration_s = end_full_run - start_full_run
                full_run_times_ms.append(inference_duration_s * 1000)

                # Stop powermetrics
                pm_stderr_data = ""
                if pm_proc: # Only if it was started successfully
                    current_pm_pid = pm_proc.pid # Store before it might be killed
                    if pm_proc.poll() is None: # Still running
                        pm_proc.send_signal(signal.SIGINT)
                        try: pm_proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if pm_proc.poll() is None: pm_proc.send_signal(signal.SIGTERM)
                            try: pm_proc.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                if pm_proc.poll() is None: pm_proc.kill(); pm_proc.wait(timeout=2) # Force kill
                    # Try to get stderr after stopping/killing
                    try:
                        if pm_proc.stderr and not pm_proc.stderr.closed:
                            _, pm_stderr_data = pm_proc.communicate(timeout=2) # Short timeout
                    except Exception as e_comm:
                        print(f"Error during final communicate for main powermetrics (MLX run {i+1}, PID {current_pm_pid}): {e_comm}")

                    if pm_stderr_data and pm_stderr_data.strip():
                        print(f"Main Powermetrics stderr (MLX run {i+1}, PID {current_pm_pid}): {pm_stderr_data.strip()}")


                power_readings_w = []
                if pm_proc and temp_pm_file_path and os.path.exists(temp_pm_file_path):
                    with open(temp_pm_file_path, "r") as f_in:
                        for line in f_in:
                            try:
                                if "Combined Power" in line and "mW" in line and ":" in line:
                                    value_part = line.split(":")[1].strip()
                                    numeric_value_str = value_part.split("mW")[0].strip()
                                    power_mw = float(numeric_value_str)
                                    power_readings_w.append(power_mw / 1000.0)
                            except ValueError:
                                print(f"Warning: Could not parse power value from line (MLX run {i+1}): '{line.strip()}'")
                            except Exception as e_parse:
                                 print(f"Warning: Unexpected error parsing power line (MLX run {i+1}) '{line.strip()}': {e_parse}")


                energy_j_this_run = None
                if power_readings_w and inference_duration_s > 0:
                    avg_power_w_this_run = sum(power_readings_w) / len(power_readings_w)
                    energy_j_this_run = avg_power_w_this_run * inference_duration_s
                energy_consumption_j_runs.append(energy_j_this_run) # Append None if not calculated

                if i == 0: # Only for the first full run
                    generated_text_preview = full_generated_text[:100] + "..."
                    # Calculate output tokens based on the actual generated part
                    # full_generated_text contains the prompt.
                    # A robust way is to subtract prompt from the full text, then tokenize the remainder.
                    if full_generated_text.startswith(prompt):
                        only_generated_part = full_generated_text[len(prompt):]
                    else: # Fallback if prompt isn't exactly at the start (less likely with mlx_lm)
                        only_generated_part = full_generated_text # This would be an overestimation
                        print("Warning: Generated text doesn't start with prompt. Output token count might be inaccurate.")
                    actual_output_tokens = get_token_count(tokenizer, only_generated_part)

                if rss_sampler_thread and rss_sampler_thread.get("stop_event"):
                    rss_sampler_thread["stop_event"].set()
                    time.sleep(rss_interval_cfg + 0.1) # Give thread time to stop
                    current_run_peak_rss_mb = rss_sampler_thread["peak_rss"] / (1024**2)
                    if current_run_peak_rss_mb > peak_rss_mb_overall:
                        peak_rss_mb_overall = current_run_peak_rss_mb

                gc.collect() # MLX also benefits from Python GC

            except Exception as e_run:
                print(f"ERROR within MLX benchmark run {i+1} for prompt '{prompt[:30]}...': {e_run}")
                import traceback; traceback.print_exc()
                energy_consumption_j_runs.append(None) # Ensure list length matches num_runs
            finally:
                if rss_sampler_thread and rss_sampler_thread.get("stop_event") and not rss_sampler_thread["stop_event"].is_set():
                    rss_sampler_thread["stop_event"].set()
                if pm_proc and pm_proc.poll() is None: # Ensure powermetrics is stopped
                    pm_proc.kill()
                    pm_proc.wait()
                if temp_pm_file_path and os.path.exists(temp_pm_file_path):
                    try: os.remove(temp_pm_file_path)
                    except OSError as e_rm: print(f"Warning: Could not remove temp powermetrics file {temp_pm_file_path}: {e_rm}")


        # After all runs for this prompt
        rss_after_all_runs_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        cpu_temp_after_c = get_cpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
        gpu_temp_after_c = get_gpu_temperature(samples=temp_samples_cfg, delay=temp_delay_cfg)
        power_after_prompt_w = get_short_powermetrics_sample_w(duration_s=2, prefix="mlx_post_prompt_pm_")

        avg_time_ms = round(sum(full_run_times_ms) / len(full_run_times_ms), 3) if full_run_times_ms else 0
        tokens_per_sec = round(actual_output_tokens / (avg_time_ms / 1000.0), 2) if avg_time_ms > 0 and actual_output_tokens > 0 else 0

        valid_energy_runs = [e for e in energy_consumption_j_runs if e is not None]
        avg_energy_consumption_j = round(sum(valid_energy_runs) / len(valid_energy_runs), 4) if valid_energy_runs else None

        avg_power_during_inference_w = None
        if avg_energy_consumption_j is not None and avg_time_ms > 0:
            avg_power_during_inference_w = round(avg_energy_consumption_j / (avg_time_ms / 1000.0), 2)

        cpu_temp_delta_c = round(cpu_temp_after_c - cpu_temp_before_c, 2) if cpu_temp_before_c is not None and cpu_temp_after_c is not None else None
        gpu_temp_delta_c = round(gpu_temp_after_c - gpu_temp_before_c, 2) if gpu_temp_before_c is not None and gpu_temp_after_c is not None else None

        peak_rss_mb_overall = round(peak_rss_mb_overall, 2) if peak_rss_mb_overall > 0 else None

        results = {
            "prompt": prompt,
            "status": "success",
            "error_message": None,
            "input_tokens": input_tokens,
            "output_tokens": actual_output_tokens, # Tokens for the generated part
            "avg_time_ms": avg_time_ms,
            "tokens_per_sec": tokens_per_sec,
            "avg_energy_consumption_j": avg_energy_consumption_j,
            "avg_power_during_inference_w": avg_power_during_inference_w,
            "power_before_prompt_w": power_before_prompt_w,
            "power_after_prompt_w": power_after_prompt_w,
            "ttft_ms_avg": ttft_ms_avg,
            "output_text_preview": generated_text_preview,
            "cpu_temp_before_c": cpu_temp_before_c,
            "cpu_temp_after_c": cpu_temp_after_c,
            "cpu_temp_delta_c": cpu_temp_delta_c,
            "gpu_temp_before_c": gpu_temp_before_c, # For MLX, this is relevant
            "gpu_temp_after_c": gpu_temp_after_c,
            "gpu_temp_delta_c": gpu_temp_delta_c,
            "rss_before_mb": round(rss_before_all_runs_mb, 2), # RSS before this prompt's runs
            "rss_after_mb": round(rss_after_all_runs_mb, 2),   # RSS after this prompt's runs
            "peak_rss_during_all_runs_mb": peak_rss_mb_overall, # Peak during this prompt's runs
            # MLX doesn't have explicit 'mps_alloc' or 'mps_resv'.
            # Peak host memory is effectively peak_rss for MLX.
            "peak_host_memory_mb": peak_rss_mb_overall, # Or can track a global peak if needed
        }

    except Exception as e_outer:
        print(f"MAJOR ERROR during MLX benchmark for prompt '{prompt[:50]}...': {e_outer}")
        import traceback; traceback.print_exc()
        results = {"prompt": prompt, "status": "failed", "error_message": str(e_outer)}
    return results

def download_mlx_model_if_needed(model_id: str, token: str = None):
    """
    MLX models are typically downloaded on first use by mlx_lm.load().
    This function can pre-download the snapshot using huggingface_hub
    if you want to separate download from first load, but mlx_lm.load()
    will still manage its own caching and potential conversion.
    For mlx-lm, simply ensuring the HF cache has the files is often enough.
    """
    print(f"Ensuring model '{model_id}' files are downloaded to Hugging Face cache for MLX...")
    try:
        # snapshot_download will download all files for the repo to HF cache
        snapshot_path = snapshot_download(
            repo_id=model_id,
            local_files_only=False, # Force download if not present
            resume_download=True,
            token=token,
            allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer.*"] # Common patterns
        )
        print(f"Model '{model_id}' files are available in cache via snapshot_download: {snapshot_path}")
        return True
    except HfHubHTTPError as e:
        print(f"Error downloading '{model_id}' via snapshot_download: Hugging Face Hub API error. {e}")
        if "401" in str(e): print(f"This might be a private model ('{model_id}'). Ensure a valid token is used.")
        elif "404" in str(e): print(f"Model or revision for '{model_id}' not found on Hugging Face Hub.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while ensuring model '{model_id}' is downloaded: {e}")
        import traceback; traceback.print_exc()
        return False


def run_full_benchmark_mlx(output_filename_param):
    print("Validating sudo privileges for powermetrics (MLX Benchmark)...")
    if not ensure_sudo_active(interactive_prompt_timeout=30):
        print("WARNING: Sudo privileges could not be obtained for MLX. Powermetrics will be skipped.")
    else:
        print("Sudo privileges validated for upcoming powermetrics calls (MLX).")

    all_prompt_results = []
    device_str = "mlx_gpu" # MLX primarily uses GPU via Unified Memory
    # MLX doesn't have explicit dtypes like torch.float16 at the mlx_lm.load level,
    # it handles precision internally, often using float16 or bfloat16 for weights.
    # We can note the framework implies optimized precision.
    mlx_precision_note = "mlx_internal (typically float16/bfloat16)"

    print(f"--- Running MLX Benchmark on device: {device_str} (Precision: {mlx_precision_note}) ---")
    print(f"--- Output will be saved to: {output_filename_param} ---")

    hf_token_from_config = cfg.get("huggingface_token")
    # No explicit hf_login() for mlx_lm, it uses huggingface_hub's default auth.
    # snapshot_download for pre-caching will use it.

    for model_id_for_mlx in model_list_from_config:
        print(f"\n{'='*20} Preparing MLX Model: {model_id_for_mlx} {'='*20}")

        # Pre-download files to HF cache if you want to separate this step.
        # mlx_lm.load() will also download if not cached.
        download_successful = download_mlx_model_if_needed(model_id_for_mlx, token=hf_token_from_config)
        if not download_successful:
            print(f"Skipping MLX benchmark for model {model_id_for_mlx} due to download/caching failure.")
            all_prompt_results.append({
                "model_id": model_id_for_mlx, "status": "download_failed",
                "error_message": f"Failed to pre-download/cache model files for {model_id_for_mlx} for MLX.",
                "device": device_str, "dtype": mlx_precision_note
            })
            try:
                with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
            except Exception as e_write: print(f"ERROR writing to {output_filename_param} (after MLX download fail): {e_write}")
            continue

        print(f"\n{'='*10} Benchmarking MLX Model (from cache): {model_id_for_mlx} {'='*10}")
        model_mlx, tokenizer_mlx = None, None
        model_load_time_s = 0.0
        rss_after_load_mb = 0.0

        try:
            # --- MLX Model Load Timing ---
            print(f"Loading MLX model and tokenizer {model_id_for_mlx} (this includes JIT, etc.)...")
            load_mlx_start_time = time.perf_counter()
            model_mlx, tokenizer_mlx = mlx_load_model(model_id_for_mlx) # This is the main loading step
            # Ensure model is fully loaded and ready on the MLX device (GPU)
            mx.eval(model_mlx.parameters()) # Evaluate all model parameters
            model_load_time_s = time.perf_counter() - load_mlx_start_time

            rss_after_load_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            print(f"MLX Model '{model_id_for_mlx}' load time: {model_load_time_s:.2f}s. RSS after load: {rss_after_load_mb:.2f} MB")

            # --- MLX Warm-up ---
            print("Global MLX warm-up for the model...")
            for w_prompt_idx, w_prompt_text in enumerate(warm_prompts):
                _ = mlx_generate_text(model_mlx, tokenizer_mlx, prompt=w_prompt_text, max_tokens=16, verbose=False)
                mx.eval() # Ensure generation is complete
            print("Global MLX warm-up complete.")

            for current_prompt_text in prompt_list:
                print(f"--- MLX Prompt: '{current_prompt_text[:50]}...' ---")
                single_prompt_run_results = benchmark_model_on_prompt_mlx(
                    model_mlx, tokenizer_mlx, model_id_for_mlx, current_prompt_text
                )
                single_prompt_run_results.update({
                    "model_id": model_id_for_mlx,
                    "device": device_str,
                    "dtype": mlx_precision_note, # How MLX handles precision
                    "model_load_time_s": round(model_load_time_s, 3),
                    "rss_after_load_mb": round(rss_after_load_mb, 2),
                    # No separate "move to device time" for MLX in this flow
                    "model_move_to_device_time_s": None, # Placeholder for schema consistency
                    "rss_after_device_move_mb": None,    # Placeholder
                })
                all_prompt_results.append(single_prompt_run_results)
                try:
                    with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
                except Exception as e_write_inc: print(f"ERROR writing MLX results to {output_filename_param} (incremental): {e_write_inc}")

        except LocalEntryNotFoundError as e_local_mlx:
            print(f"FATAL ERROR for MLX model {model_id_for_mlx}: Could not load model using mlx_lm.load(). {e_local_mlx}")
            print("This might mean the model is not MLX compatible, cache is corrupted, or files are missing.")
            import traceback; traceback.print_exc()
            all_prompt_results.append({
                "model_id": model_id_for_mlx, "status": "mlx_load_failed",
                "error_message": str(e_local_mlx), "device": device_str, "dtype": mlx_precision_note
            })
        except Exception as e_model_scope_mlx:
            print(f"FATAL ERROR during MLX setup/benchmark for {model_id_for_mlx}: {e_model_scope_mlx}")
            import traceback; traceback.print_exc()
            all_prompt_results.append({
                "model_id": model_id_for_mlx, "status": "mlx_setup_or_run_failed",
                "error_message": str(e_model_scope_mlx), "device": device_str, "dtype": mlx_precision_note
            })
        finally:
            print(f"Cleaning up MLX resources for {model_id_for_mlx}...")
            del model_mlx
            del tokenizer_mlx
            model_mlx, tokenizer_mlx = None, None
            # MLX manages its memory; explicit cache clearing isn't like torch.mps.empty_cache()
            # We rely on Python's GC and MLX's internal management.
            gc.collect()
            print("MLX Cleanup complete.")
            # Final save
            try:
                with open(output_filename_param, "w") as f: json.dump(all_prompt_results, f, indent=4)
            except Exception as e_final_write_mlx: print(f"ERROR writing MLX results to {output_filename_param} (final): {e_final_write_mlx}")

    print(f"\nMLX Benchmark run complete. Results saved to {output_filename_param}")


if __name__ == "__main__":
    print("MLX Benchmark script starting...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Effective user ID: {os.geteuid()} (0 is root)")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    abs_results_dir = results_dir_from_config
    if not os.path.isabs(results_dir_from_config):
        abs_results_dir = os.path.join(script_dir, results_dir_from_config)

    print(f"Attempting to use results directory for MLX: {abs_results_dir}")
    try:
        os.makedirs(abs_results_dir, exist_ok=True)
        print(f"Ensured MLX results directory exists: {abs_results_dir}")
        # Permission test
        test_file_path_mlx = os.path.join(abs_results_dir, f".permission_test_mlx_{timestamp}")
        with open(test_file_path_mlx, "w") as test_f: test_f.write("test_mlx")
        os.remove(test_file_path_mlx)
        print(f"Write permission confirmed for MLX results in: {abs_results_dir}")
    except OSError as e_dir:
        print(f"ERROR: Could not create/write to MLX results directory {abs_results_dir}: {e_dir}\nExiting.")
        sys.exit(1)

    output_file_mlx = os.path.join(abs_results_dir, f"{base_output_filename_cfg}_{timestamp}.json")

    if os.geteuid() != 0:
        print("Script not running as root. Sudo will be used for powermetrics (MLX).")

    run_full_benchmark_mlx(output_filename_param=output_file_mlx)