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

import torch
import psutil
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS backend not available. Please install a compatible PyTorch 2.x build.")

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

model_list = cfg["models"]
warm_prompts = cfg["warm_prompts"]
prompt_list = cfg["prompt_list"]

gen_config = GenerationConfig(
    max_new_tokens=cfg["generation"]["max_new_tokens"],
    do_sample=cfg["generation"]["do_sample"],
)

num_runs = cfg["sampling"]["num_runs_per_prompt"]
rss_interval = cfg["sampling"]["rss_sampler_interval_seconds"]
temp_samples = cfg["sampling"]["temperature_samples"]
temp_delay = cfg["sampling"]["temperature_delay_seconds"]

results_dir_from_config = cfg["output"]["results_directory"]
base_output_filename = cfg["output"]["base_output_filename"]

# --- Helper function to run powermetrics for a short duration and get average power ---
def get_short_powermetrics_sample_w(duration_s=2, interval_ms=1000, prefix="short_pm_"):
    """
    Runs powermetrics for a short duration and returns average 'Combined Power' in Watts.
    Returns None if unsuccessful.
    """
    pm_proc = None
    temp_pm_file_path = None
    avg_power_w = None
    num_samples_to_take = max(1, int(duration_s * 1000 / interval_ms))

    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt", prefix=prefix) as tmp_f:
            temp_pm_file_path = tmp_f.name
        
        # -n {num_samples_to_take} specifies number of samples, -i {interval_ms} is interval
        pm_command = ["sudo", "powermetrics", "-i", str(interval_ms), 
                      "--samplers", "cpu_power", "-n", str(num_samples_to_take),
                      "-o", temp_pm_file_path]
        
        pm_proc = subprocess.Popen(pm_command, stderr=subprocess.PIPE, text=True)
        
        # Wait for powermetrics to complete (it will exit after -n samples)
        # Timeout should be slightly longer than num_samples_to_take * interval_ms
        wait_timeout = (num_samples_to_take * interval_ms / 1000.0) + 3.0 # Add 3s buffer
        
        pm_stderr_data = ""
        try:
            _, pm_stderr_data = pm_proc.communicate(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            print(f"Warning: Short powermetrics sample (prefix: {prefix}) timed out. Killing.")
            if pm_proc.poll() is None: pm_proc.kill()
            try: _, pm_stderr_data = pm_proc.communicate(timeout=2) # Try to get stderr after kill
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
                    except ValueError: pass # Silently ignore parsing errors for short sample
                    except Exception: pass
            if power_readings:
                avg_power_w = round(sum(power_readings) / len(power_readings), 2)
                
    except Exception as e:
        print(f"Error during short powermetrics sample (prefix: {prefix}): {e}")
    finally:
        if pm_proc and pm_proc.poll() is None:
            pm_proc.kill(); pm_proc.wait()
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
        return round(sum(readings) / len(readings), 2) if readings else None
    except FileNotFoundError: print(f"Warning: 'smctemp' command not found for '{param}'. Temp will be null."); return None
    except subprocess.CalledProcessError as e: print(f"Warning: 'smctemp {param}' failed: {e}. Temp will be null."); return None
    except Exception as e: print(f"Warning: Error getting temp '{param}': {e}. Temp will be null."); return None

def get_gpu_temperature(samples=3, delay=0.1): return get_temperature_with_smctemp("-g", samples, delay)
def get_cpu_temperature(samples=3, delay=0.1): return get_temperature_with_smctemp("-c", samples, delay)

def benchmark_model_on_prompt_mps(model, tokenizer, prompt, dtype):
    results = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        energy_runs = []
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]
        process = psutil.Process(os.getpid())
        peak_mem_mb, peak_mps_mb = 0, 0
        
        # --- TTFT Runs ---
        ttft_runs = []
        for _ in range(num_runs): # TTFT is short, separate from main power measurement
            start_ttft = time.time()
            with torch.no_grad(): _ = model.generate(max_new_tokens=1, do_sample=False, **inputs)
            end_ttft = time.time()
            ttft_runs.append((end_ttft - start_ttft) * 1000)
        ttft_ms_avg = round(sum(ttft_runs) / len(ttft_runs), 2) if ttft_runs else 0.0

        # --- Power sample BEFORE the main inference loop for the prompt ---
        print("Taking pre-prompt power sample...")
        power_before_prompt_w = get_short_powermetrics_sample_w(duration_s=2, prefix="pre_prompt_pm_")
        print(f"Pre-prompt power sample: {power_before_prompt_w} W")

        full_runs, output_tokens, generated_text = [], 0, ""
        mem_before_mb, mem_after_mb, memory_delta_mb = None, None, None
        cpu_temp_before, cpu_temp_after, cpu_temp_delta_c = None, None, None
        gpu_temp_before, gpu_temp_after, gpu_temp_delta_c = None, None, None
        mps_alloc_before, mps_resv_before, rss_before = None, None, None
        mps_alloc_after, mps_resv_after, rss_after_mb_val = None, None, None
        peak_rss_mb_overall = 0

        for i in range(num_runs): # Main inference runs for the prompt
            pm_proc, temp_pm_file_path, sampler = None, None, None
            try:
                with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt", prefix=f"main_pm_run{i}_") as tmp_f:
                    temp_pm_file_path = tmp_f.name
                # For main runs, powermetrics runs for the duration of inference
                pm_command = ["sudo", "powermetrics", "-i", "1000", "--samplers", "cpu_power", "-o", temp_pm_file_path]
                pm_proc = subprocess.Popen(pm_command, stderr=subprocess.PIPE, text=True)
                time.sleep(0.5) # Allow powermetrics to start
                if pm_proc.poll() is not None:
                    pm_stderr_early = pm_proc.stderr.read() if pm_proc.stderr else ""
                    raise RuntimeError(f"Main Powermetrics (PID: {pm_proc.pid if pm_proc else 'N/A'}) failed/exited. Stderr: {pm_stderr_early}")

                # Snapshots before this specific run
                if i == 0: # Only take "before" CPU/GPU temps once for the prompt
                    mem_before_mb = process.memory_full_info().rss / (1024 * 1024)
                    gpu_temp_before = get_gpu_temperature(samples=temp_samples, delay=temp_delay)
                    cpu_temp_before = get_cpu_temperature(samples=temp_samples, delay=temp_delay)
                    mps_alloc_before, mps_resv_before = get_mps_usage_mb()
                    rss_before = get_rss_usage_mb()

                sampler = start_rss_sampler(process, interval=rss_interval)
                
                start_full = time.time()
                with torch.no_grad():
                    outputs = model.generate(generation_config=gen_config, return_dict_in_generate=True, output_scores=True, **inputs)
                end_full = time.time()
                inference_duration_s = end_full - start_full

                # Aggressive termination for main powermetrics
                pm_stderr_data = ""
                if pm_proc:
                    # ... (aggressive termination logic as before) ...
                    current_pm_pid = pm_proc.pid
                    if pm_proc.poll() is None:
                        print(f"Sending SIGINT to main powermetrics (PID: {current_pm_pid})...")
                        pm_proc.send_signal(signal.SIGINT)
                        try: pm_proc.wait(timeout=5); print(f"Main Powermetrics (PID: {current_pm_pid}) terminated via SIGINT.")
                        except subprocess.TimeoutExpired:
                            # print(f"Warning: Main powermetrics (PID: {current_pm_pid}) no SIGINT response. Sending SIGTERM...")
                            if pm_proc.poll() is None: pm_proc.send_signal(signal.SIGTERM)
                            try: pm_proc.wait(timeout=3); print(f"Main Powermetrics (PID: {current_pm_pid}) terminated via SIGTERM.")
                            except subprocess.TimeoutExpired:
                                print(f"Warning: Main powermetrics (PID: {current_pm_pid}) no SIGTERM response. Sending SIGKILL.")
                                if pm_proc.poll() is None: pm_proc.kill(); pm_proc.wait(timeout=2)
                                print(f"Main Powermetrics (PID: {current_pm_pid}) killed.")
                    else: print(f"Main Powermetrics (PID: {current_pm_pid}) already terminated.")
                    try:
                        if pm_proc.stderr and not pm_proc.stderr.closed: _, pm_stderr_data = pm_proc.communicate(timeout=2)
                    except Exception as e_comm: print(f"Error final communicate main powermetrics (PID: {current_pm_pid}): {e_comm}")
                    if pm_stderr_data and pm_stderr_data.strip(): print(f"Main Powermetrics stderr (run {i+1}, PID {current_pm_pid}): {pm_stderr_data.strip()}")
                
                power_readings = []
                if temp_pm_file_path and os.path.exists(temp_pm_file_path):
                    with open(temp_pm_file_path, "r") as f_in:
                        for line in f_in: # Parse main powermetrics file
                            try:
                                if "Combined Power" in line and "mW" in line and ":" in line:
                                    value_part = line.split(":")[1].strip(); numeric_value_str = value_part.split("mW")[0].strip()
                                    power_mw = float(numeric_value_str); power_w = power_mw / 1000.0
                                    power_readings.append(power_w)
                            except ValueError: print(f"Warning: Could not parse power value from line: '{line.strip()}' (run {i+1})")
                            except Exception as e: print(f"Warning: Unexpected error parsing power line '{line.strip()}': {e} (run {i+1})")
                
                energy_j_run = None
                if power_readings and inference_duration_s > 0:
                    avg_power_w_run = sum(power_readings) / len(power_readings)
                    energy_j_run = avg_power_w_run * inference_duration_s
                else: print(f"Warning: No/invalid power readings or duration for main run {i+1}. Energy will be null.")
                energy_runs.append(energy_j_run)

                # Snapshots after this specific run
                if i == num_runs - 1: # Only take "after" CPU/GPU temps once for the prompt (after last run)
                    mps_alloc_after, mps_resv_after = get_mps_usage_mb()
                    mem_after_mb = process.memory_full_info().rss / (1024 * 1024); rss_after_mb_val = mem_after_mb
                    gpu_temp_after = get_gpu_temperature(samples=temp_samples, delay=temp_delay)
                    cpu_temp_after = get_cpu_temperature(samples=temp_samples, delay=temp_delay)

                if mps_alloc_after is not None and mps_alloc_after > peak_mps_mb: peak_mps_mb = mps_alloc_after
                
                if sampler and sampler.get("stop_event"):
                    sampler["stop_event"].set(); time.sleep(rss_interval + 0.05)
                    current_run_peak_rss = sampler["peak_rss"] / (1024 ** 2)
                    if current_run_peak_rss > peak_rss_mb_overall: peak_rss_mb_overall = current_run_peak_rss
                
                current_peak_host_mem = max(mem_before_mb or 0, (mem_after_mb if i == num_runs -1 else 0) or (process.memory_full_info().rss / (1024*1024)))
                if current_peak_host_mem > peak_mem_mb: peak_mem_mb = current_peak_host_mem
                
                iter_time_ms = inference_duration_s * 1000; full_runs.append(iter_time_ms)
                if i == 0: # Decode output only from the first run
                    actual_output_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                    generated_text = tokenizer.decode(actual_output_ids, skip_special_tokens=True)
                    output_tokens = len(actual_output_ids)
                torch.mps.empty_cache(); gc.collect()
            except Exception as e_run:
                print(f"ERROR within benchmark run {i+1} for prompt '{prompt[:30]}...': {e_run}")
                import traceback; traceback.print_exc(); energy_runs.append(None)
            finally: # Cleanup for each run
                if sampler and sampler.get("stop_event") and not sampler["stop_event"].is_set(): sampler["stop_event"].set()
                if pm_proc and pm_proc.poll() is None: print(f"Warning: Main powermetrics (PID {pm_proc.pid}) still running in run's finally. Killing."); pm_proc.kill(); pm_proc.wait()
                if temp_pm_file_path and os.path.exists(temp_pm_file_path):
                    try: os.remove(temp_pm_file_path)
                    except OSError as e: print(f"Warning: Could not remove temp main powermetrics file {temp_pm_file_path}: {e}")
        
        # --- Power sample AFTER the main inference loop for the prompt ---
        print("Taking post-prompt power sample...")
        power_after_prompt_w = get_short_powermetrics_sample_w(duration_s=2, prefix="post_prompt_pm_")
        print(f"Post-prompt power sample: {power_after_prompt_w} W")

        # --- Aggregate results for the prompt ---
        avg_time_ms = round(sum(full_runs) / len(full_runs), 3) if full_runs else 0
        
        valid_energy_duration_pairs = []
        for idx, energy_j in enumerate(energy_runs):
            if energy_j is not None and idx < len(full_runs) and full_runs[idx] > 0:
                valid_energy_duration_pairs.append({"energy": energy_j, "duration": full_runs[idx] / 1000.0})

        avg_energy_consumption_j, avg_power_consumption_w_during = None, None # Renamed for clarity
        if valid_energy_duration_pairs:
            total_energy_j_valid = sum(p["energy"] for p in valid_energy_duration_pairs)
            total_duration_s_valid = sum(p["duration"] for p in valid_energy_duration_pairs)
            avg_energy_consumption_j = round(total_energy_j_valid / len(valid_energy_duration_pairs), 4)
            if total_duration_s_valid > 0:
                avg_power_consumption_w_during = round(total_energy_j_valid / total_duration_s_valid, 2)
        
        if not valid_energy_duration_pairs and any(er is None for er in energy_runs): print(f"Warning: Energy/power for prompt '{prompt[:30]}...' may be null.")

        tokens_per_sec = round(output_tokens / (avg_time_ms / 1000.0), 2) if avg_time_ms > 0 else 0
        peak_memory_mb_val = round(peak_mem_mb, 2)
        if cpu_temp_before is not None and cpu_temp_after is not None: cpu_temp_delta_c = round(cpu_temp_after - cpu_temp_before, 2)
        if gpu_temp_before is not None and gpu_temp_after is not None: gpu_temp_delta_c = round(gpu_temp_after - gpu_temp_before, 2)
        
        results = {
            "prompt": prompt, "status": "success", "error_message": None,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "avg_time_ms": avg_time_ms, "tokens_per_sec": tokens_per_sec,
            "avg_energy_consumption_j": avg_energy_consumption_j,
            "avg_power_during_inference_w": avg_power_consumption_w_during, # Average power during inference runs
            "power_before_prompt_w": power_before_prompt_w,          # Power sample before runs
            "power_after_prompt_w": power_after_prompt_w,           # Power sample after runs
            "ttft_ms_avg": ttft_ms_avg,
            "output_text_preview": generated_text[:100] + "...",
            "cpu_temp_before_c": cpu_temp_before, "cpu_temp_after_c": cpu_temp_after, "cpu_temp_delta_c": cpu_temp_delta_c,
            "gpu_temp_before_c": gpu_temp_before, "gpu_temp_after_c": gpu_temp_after, "gpu_temp_delta_c": gpu_temp_delta_c,
            "mps_alloc_before_mb": mps_alloc_before, "mps_resv_before_mb": mps_resv_before,
            "rss_before_mb": round(rss_before, 2) if rss_before is not None else None,
            "mps_alloc_after_mb": mps_alloc_after, "mps_resv_after_mb": mps_resv_after,
            "rss_after_mb": round(rss_after_mb_val, 2) if rss_after_mb_val is not None else None,
            "peak_rss_during_all_runs_mb": round(peak_rss_mb_overall, 2) if peak_rss_mb_overall != 0 else None,
            "peak_mps_mb": round(peak_mps_mb, 2), "peak_host_memory_mb": peak_memory_mb_val,
        }
    except Exception as e_outer:
        print(f"MAJOR ERROR during benchmark_model_on_prompt_mps for '{prompt[:50]}...': {e_outer}")
        import traceback; traceback.print_exc()
        results = {"prompt": prompt, "status": "failed", "error_message": str(e_outer)}
    return results

def run_full_benchmark_mps(output_filename_param):
    try: # Sudo validation
        subprocess.run(["sudo", "-nv"], check=True, timeout=5); print("Sudo privileges validated non-interactively.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("Non-interactive sudo validation failed, attempting interactive `sudo -v`...")
        try: subprocess.run(["sudo", "-v"], check=True, timeout=30); print("Sudo privileges validated interactively.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as sudo_err: print(f"ERROR: Could not validate sudo privileges: {sudo_err}.")

    all_results = []
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    benchmark_dtype = torch.float16
    print(f"--- Running Benchmark on device: {device} with dtype: {benchmark_dtype} ---")
    print(f"--- Output will be saved to: {output_filename_param} ---")

    try: # Hugging Face Login
        token = cfg.get("huggingface_token")
        if token: login(token=token); print("Logged in to Hugging Face Hub (token from config).")
        else: print("HF token not in config. Skipping login or using anonymous access.")
    except Exception as e: print(f"Warning: HF login failed/skipped: {e}")

    for model_id in model_list: # Loop through models
        print(f"\n{'='*20} Benchmarking Model: {model_id} {'='*20}")
        model, tokenizer = None, None
        model_load_time, rss_after_load, rss_after_move = 0.0, 0.0, 0.0
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            print(f"Loading model {model_id} to CPU (dtype={benchmark_dtype})..."); load_start = time.time()
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=benchmark_dtype)
            model_load_time = time.time() - load_start
            rss_after_load = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            print(f"RSS after CPU load: {rss_after_load:.2f} MB")
            print(f"Moving model to {device}..."); model.to(device); model.eval()
            if device == "mps": torch.mps.empty_cache()
            gc.collect()
            rss_after_move = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            print(f"Model on {device} in {model_load_time:.2f}s; RSS after move: {rss_after_move:.2f} MB")
            
            print("Global warm-up...");
            for w_prompt in warm_prompts: # Warm-up runs
                w_inputs = tokenizer(w_prompt, return_tensors="pt").to(device)
                with torch.no_grad(): _ = model.generate(**w_inputs, max_new_tokens=16, do_sample=False)
            print("Global warm-up complete.")
            
            for prompt_text in prompt_list: # Loop through prompts
                print(f"--- Prompt: '{prompt_text[:50]}...' ---")
                prompt_results = benchmark_model_on_prompt_mps(model, tokenizer, prompt_text, benchmark_dtype)
                prompt_results.update({
                    "model_id": model_id, "device": device, "dtype": str(benchmark_dtype),
                    "model_load_time_s": round(model_load_time, 2),
                    "rss_after_cpu_load_mb": round(rss_after_load, 2),
                    "rss_after_device_move_mb": round(rss_after_move, 2),
                })
                all_results.append(prompt_results)
                try: # Incremental save
                    with open(output_filename_param, "w") as f: json.dump(all_results, f, indent=4)
                    print(f"--- Results for prompt saved. Avg Power During: {prompt_results.get('avg_power_during_inference_w')} W; Energy: {prompt_results.get('avg_energy_consumption_j')} J ---")
                except Exception as e_write: print(f"ERROR writing to {output_filename_param} (incremental): {e_write}")
        except Exception as e_model_scope:
            print(f"FATAL ERROR for {model_id}: {e_model_scope}")
            import traceback; traceback.print_exc()
            all_results.append({"model_id": model_id, "status": "load_or_setup_failed","error_message": str(e_model_scope), "device": device, "dtype": str(benchmark_dtype)})
        finally: # Cleanup for model
            print(f"Cleaning up resources for {model_id}..."); del model; del tokenizer
            model, tokenizer = None, None
            if device == "mps": torch.mps.empty_cache()
            gc.collect(); print("Cleanup complete.")
            try: # Final save
                with open(output_filename_param, "w") as f: json.dump(all_results, f, indent=4)
            except Exception as e_final_write: print(f"ERROR writing to {output_filename_param} (final): {e_final_write}")
    print(f"\nBenchmark run complete. Results saved to {output_filename_param}")

if __name__ == "__main__":
    print("Benchmark script starting...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Effective user ID: {os.geteuid()} (0 is root)")
    if os.geteuid() == 0: print("Script is running as root.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    abs_results_dir = results_dir_from_config
    if not os.path.isabs(results_dir_from_config): abs_results_dir = os.path.join(script_dir, results_dir_from_config)
    print(f"Attempting to use results directory: {abs_results_dir}")
    try:
        os.makedirs(abs_results_dir, exist_ok=True)
        print(f"Ensured results directory exists: {abs_results_dir}")
        test_file_path = os.path.join(abs_results_dir, f".permission_test_{timestamp}")
        with open(test_file_path, "w") as test_f: test_f.write("test"); os.remove(test_file_path)
        print(f"Write permission confirmed for: {abs_results_dir}")
    except OSError as e:
        print(f"ERROR: Could not create/write to results directory {abs_results_dir}: {e}\nExiting."); sys.exit(1)
    
    output_file = os.path.join(abs_results_dir, f"{base_output_filename}_{timestamp}.json")
    run_full_benchmark_mps(output_filename_param=output_file)