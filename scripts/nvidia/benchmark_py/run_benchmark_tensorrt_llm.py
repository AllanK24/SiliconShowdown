import os
import sys
import json
import torch
import time
import statistics
import argparse # For specifying engine directory
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    print("pynvml initialized successfully for Nvidia GPU monitoring.")
except Exception as e:
    NVML_AVAILABLE = False
    print(f"pynvml initialization failed: {e}. Nvidia GPU temp/power monitoring disabled.")

# --- TensorRT-LLM ---
try:
    import tensorrt_llm
    import tensorrt_llm.runtime as trt_runtime
    from tensorrt_llm.runtime.generation import SamplingConfig # Or GenerationConfig if that's the class name
    print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
    TRT_LLM_AVAILABLE = True
except ImportError:
    print("ERROR: tensorrt_llm package not found. Please install it following Nvidia's documentation.")
    TRT_LLM_AVAILABLE = False
    sys.exit(1)

from huggingface_hub import login
from transformers import AutoTokenizer, GenerationConfig # Keep HF tokenizer and base GenerationConfig

# ---------- Model List (Needs matching built engines) ----------
model_list = [
    # These IDs should correspond to folder names within your base engine directory
    # AND you need the original HF model downloaded for the tokenizer
    'google/gemma-1.1-2b-it',
    'Qwen/Qwen1.5-1.8B-Chat',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
]
# Corresponding HF model IDs if different from engine folder names (usually the same)
hf_model_ids = {
    'google/gemma-1.1-2b-it': 'google/gemma-1.1-2b-it',
    'Qwen/Qwen1.5-1.8B-Chat': 'Qwen/Qwen1.5-1.8B-Chat',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
}


# ---------- Warm-up Prompts ----------
warm_prompts = [ "Hello?", "What is 1+1?", "Write a word." ] # Shorter warmups might suffice
NUM_GLOBAL_WARMUP_RUNS = len(warm_prompts)

# ---------- Benchmark Prompts ----------
prompt_list = [
    "Translate the following sentence to German: 'The weather is beautiful today.'",
    "Explain the concept of quantum entanglement in simple terms.",
    "Write a python function that calculates the factorial of a number.",
    "Summarize the main plot points of the movie 'Inception'.",
    "What are the main differences between renewable and non-renewable energy sources?",
]
NUM_TIMED_RUNS_PER_PROMPT = 3

# ---------- Generation Config (Base - map to TRT later) ----------
MAX_NEW_TOKENS = 512
# We'll use this to configure TRT's SamplingConfig
hf_generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    # TRT doesn't use pad_token_id directly in the same way usually
)
BATCH_SIZE = 1 # Fixed batch size

# ---------- NVML Helpers (Same as CUDA script) ----------
def get_nvidia_gpu_details(device_id=0):
    # ... (Keep the exact same function as in the CUDA script) ...
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

# ---------- Benchmark Function (TensorRT Specific - Per Prompt) ----------
def benchmark_model_on_prompt_trt(trt_session, tokenizer, prompt, sampling_config, num_runs=3):
    """Runs benchmark for a single prompt using TensorRT-LLM session."""
    results = {}
    device = "cuda" # TRT runs on CUDA
    try:
        # --- Prepare inputs ---
        # TRT-LLM usually expects input_ids as a list or numpy array for batch=1
        # Or a padded tensor for batch > 1
        input_ids = tokenizer.encode(prompt, return_tensors=None, add_special_tokens=False) # Get list of IDs
        input_lengths = torch.tensor([len(input_ids)], dtype=torch.int32, device=device)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.int32, device=device) # Make it [1, seq_len]

        # --- Timed Runs ---
        gpu_times_ms = []
        output_tokens = 0
        generated_text = ""

        torch.cuda.synchronize(device)
        # Peak memory from PyTorch allocator might be less relevant/accurate for TRT
        # TRT engine has its own memory footprint + activation memory managed potentially outside torch
        torch.cuda.reset_peak_memory_stats(device)

        temp_before, power_before = get_nvidia_gpu_details()

        for i in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_event.record()

            # --- TRT-LLM Inference Call ---
            # The exact API might vary slightly based on tensorrt_llm version
            # Assumes 'decode' or similar method handles generation loop internally
            # Pass input_ids as tensor, input_lengths tensor, and SamplingConfig
            trt_outputs = trt_session.generate(
                 input_ids=input_ids_tensor,
                 input_lengths=input_lengths,
                 sampling_config=sampling_config,
                 # Might need max_new_tokens here too depending on API version
            )
            # --- End Inference ---

            end_event.record()
            torch.cuda.synchronize(device)
            iter_time_ms = start_event.elapsed_time(end_event)
            gpu_times_ms.append(iter_time_ms)

            # Decode output only once - Check trt_outputs structure
            # It might return output IDs directly, potentially including input IDs
            if i == 0:
                # Assuming trt_outputs['output_ids'] gives shape [batch_size, beam_width, seq_len]
                # For batch=1, beam=1 -> [1, 1, seq_len]
                output_ids_tensor = trt_outputs['output_ids'][0, 0]
                # Exclude input tokens if they are included in the output
                # Note: Some TRT setups might only return *new* tokens
                # Verify the behavior of your specific engine build/runtime version!
                start_index = input_ids_tensor.shape[1] # Index after input tokens
                actual_output_ids_tensor = output_ids_tensor[start_index:]
                output_tokens = len(actual_output_ids_tensor)
                generated_text = tokenizer.decode(actual_output_ids_tensor, skip_special_tokens=True)


        temp_after, power_after = get_nvidia_gpu_details()
        # Report torch peak memory, but be aware it might not capture all TRT usage
        peak_torch_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        # --- Calculate Aggregated Metrics ---
        avg_time_ms = statistics.mean(gpu_times_ms) if gpu_times_ms else 0
        stddev_time_ms = statistics.stdev(gpu_times_ms) if len(gpu_times_ms) > 1 else 0
        tokens_per_sec = (output_tokens / (avg_time_ms / 1000.0)) if avg_time_ms > 0 else 0
        avg_temp_c = (temp_before + temp_after) / 2 if temp_before is not None and temp_after is not None else None
        temp_increase_c = temp_after - temp_before if temp_before is not None and temp_after is not None else None
        avg_power_w = (power_before + power_after) / 2 if power_before is not None and power_after is not None else None

        results = {
            "prompt": prompt, "status": "success", "error_message": None,
            "input_tokens": len(input_ids), # Use original list length
            "output_tokens": output_tokens,
            "avg_gpu_time_ms": round(avg_time_ms, 3),
            "stddev_gpu_time_ms": round(stddev_time_ms, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "runs_gpu_time_ms": [round(t, 3) for t in gpu_times_ms],
            "peak_torch_gpu_memory_mb": round(peak_torch_memory_mb, 2) if peak_torch_memory_mb is not None else None, # Note: PyTorch allocated only
            "temp_before_c": temp_before, "temp_after_c": temp_after,
            "avg_temp_c": round(avg_temp_c, 1) if avg_temp_c is not None else None,
            "temp_increase_c": round(temp_increase_c, 1) if temp_increase_c is not None else None,
            "power_before_w": round(power_before, 2) if power_before is not None else None,
            "power_after_w": round(power_after, 2) if power_after is not None else None,
            "avg_power_w": round(avg_power_w, 2) if avg_power_w is not None else None,
            "output_text_preview": generated_text[:100] + "..."
        }

    except Exception as e:
        print(f"ERROR during TRT benchmark for prompt: '{prompt[:50]}...' - {e}")
        # Attempt to get current memory/temp even on error if possible
        peak_memory_mb_err = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else None
        temp_err, power_err = get_nvidia_gpu_details()
        results = {
            "prompt": prompt, "status": "failed", "error_message": str(e),
            "peak_torch_gpu_memory_mb_on_error": round(peak_memory_mb_err, 2) if peak_memory_mb_err is not None else None,
            "temp_on_error": temp_err, "power_on_error": power_err,
        }
    return results

# ---------- Main Benchmark Runner (TensorRT Specific) ----------
def run_full_benchmark_trt(engine_dir_base, output_filename="benchmark_results_trt.json"):
    """Runs TRT-LLM benchmarks for all models/prompts, saving results."""
    if not TRT_LLM_AVAILABLE:
        print("TensorRT-LLM is not available. Exiting.")
        return

    all_results = []
    device = "cuda"
    # Dtype is typically determined during engine build, we record it
    benchmark_dtype = "float16" # Assuming FP16 engines, adjust if using others
    print(f"--- Running TensorRT Benchmark (Assumed {benchmark_dtype}), Batch Size: {BATCH_SIZE} ---")
    print(f"--- Looking for engines in subdirs of: {engine_dir_base} ---")


    # --- HF Login (for tokenizer) ---
    try:
        token = os.environ.get("HF_TOKEN")
        if token: login(token=token); print("Logged in.")
        else: print("HF_TOKEN not set. Ensure models cached/public.")
    except Exception as e: print(f"Login failed: {e}")

    # --- Loop through models (engine subdirs) ---
    for model_subdir_name in model_list: # Use model_list to find engine dirs and tokenizers
        print(f"\n{'='*20} Benchmarking Model Engine: {model_subdir_name} {'='*20}")

        # Construct path to the pre-built engine directory
        # Assumes structure like: <engine_dir_base>/<model_subdir_name>/fp16/1-gpu/
        engine_path = os.path.join(engine_dir_base, model_subdir_name, "fp16", "1-gpu") # Adjust path structure if needed

        if not os.path.isdir(engine_path):
            print(f"ERROR: Engine directory not found: {engine_path}. Skipping model.")
            all_results.append({ "model_id": model_subdir_name, "status": "engine_not_found", "engine_path_searched": engine_path })
            continue

        trt_session, tokenizer = None, None
        engine_load_time = None
        current_model_params = {}

        try:
            # --- Load Tokenizer ---
            hf_model_id = hf_model_ids[model_subdir_name] # Get HF ID for tokenizer
            print(f"Loading tokenizer for {hf_model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True, padding_side='left') # TRT often prefers left padding
            # Ensure pad token exists for tokenizer if needed by specific logic
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Set tokenizer pad_token to eos_token.")

            # --- Load TRT Engine & Create Session ---
            print(f"Loading TRT engine from: {engine_path}...")
            load_start = time.time()
            # You might need to load config.json from the engine dir first
            # Adjust runtime mapping based on your setup (GPU rank)
            runtime_mapping = tensorrt_llm.Mapping(world_size=1, rank=0, gpus=list(range(1))) # Single GPU

            # Example loading - API might change slightly
            # Need engine path and tokenizer
            trt_session = trt_runtime.GenerationSession.from_dir(
                 engine_dir=engine_path,
                 tokenizer=tokenizer, # Pass tokenizer for potential internal use
                 runtime_mapping=runtime_mapping,
                 # stream = torch.cuda.current_stream() # Optional stream management
            )

            load_end = time.time()
            engine_load_time = load_end - load_start
            print(f"TRT session ready in {engine_load_time:.2f} seconds.")

            # --- Configure TRT Sampling/Generation ---
            # Map HF GenerationConfig to TRT SamplingConfig
            sampling_config = trt_runtime.SamplingConfig(
                end_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                # For greedy search (do_sample=False):
                num_beams=1, # Must be 1 for greedy
                # top_k=1, # Often implicit with beam=1, but can be set
                # top_p=1.0, # Ensure sampling is off
                # temperature=1.0, # Usually ignored for greedy
            )
            # Note: max_new_tokens is often passed to the generate call itself in TRT-LLM

            # --- Record Benchmark Parameters ---
            current_model_params = {
                "model_id": model_subdir_name, # Use engine name/HF ID
                "benchmark_dtype": benchmark_dtype, # From assumption/build
                "batch_size": BATCH_SIZE,
                "generation_config": hf_generation_config.to_dict(), # Store base HF config for reference
                "trt_sampling_config": sampling_config.__dict__, # Store TRT config used
                "num_global_warmup_runs": NUM_GLOBAL_WARMUP_RUNS,
                "num_timed_runs_per_prompt": NUM_TIMED_RUNS_PER_PROMPT,
                "engine_load_time_s": round(engine_load_time, 2),
                "accelerator_used": "TensorRT",
                "quantization_method": "None", # Or specify if engine is quantized
                "engine_path": engine_path
            }

            # --- Global Warm-up (TRT Session) ---
            print(f"Running {NUM_GLOBAL_WARMUP_RUNS} global warm-up prompts...")
            for w_prompt in warm_prompts:
                 w_input_ids = tokenizer.encode(w_prompt, return_tensors=None, add_special_tokens=False)
                 w_input_lengths = torch.tensor([len(w_input_ids)], dtype=torch.int32, device=device)
                 w_input_ids_tensor = torch.tensor([w_input_ids], dtype=torch.int32, device=device)
                 _ = trt_session.generate(
                     input_ids=w_input_ids_tensor,
                     input_lengths=w_input_lengths,
                     sampling_config=sampling_config,
                     max_new_tokens=16 # Generate few tokens for warmup
                 )
            torch.cuda.synchronize(device)
            print("Global warm-up complete.")

            # --- Benchmark each prompt ---
            for prompt_text in prompt_list:
                print(f"--- Prompt: '{prompt_text[:50]}...' ---")
                prompt_metrics = benchmark_model_on_prompt_trt(
                    trt_session, tokenizer, prompt_text, sampling_config,
                    num_runs=NUM_TIMED_RUNS_PER_PROMPT
                )

                # Combine model parameters with prompt metrics
                final_record = {**current_model_params, **prompt_metrics}
                all_results.append(final_record)

                # Save intermediate results
                with open(output_filename, "w") as f:
                    json.dump(all_results, f, indent=4)

        except Exception as e:
            print(f"FATAL ERROR for model engine {model_subdir_name}: {e}")
            all_results.append({
                **current_model_params,
                "prompt": "LOAD_OR_SETUP_FAILURE", "status": "load_or_setup_failed",
                "error_message": str(e), "engine_path_searched": engine_path
            })
        finally:
            # --- Cleanup TRT session ---
            print(f"Cleaning up TRT resources for {model_subdir_name}...")
            del trt_session # Important to release TRT resources
            del tokenizer
            torch.cuda.empty_cache()
            print("Cleanup complete.")
            # Save final results again
            with open(output_filename, "w") as f:
                 json.dump(all_results, f, indent=4)

    print(f"\nTensorRT Benchmark run complete. Results: {output_filename}")
    if NVML_AVAILABLE: pynvml.nvmlShutdown()

# --- Run ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TensorRT-LLM Benchmark")
    parser.add_argument(
        "--engine_dir", type=str, required=True,
        help="Base directory containing the pre-built TensorRT-LLM engine subdirectories (e.g., ./trt_engines)"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Optional: Specify output JSON filename."
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = args.output_file or f"llm_benchmark_results_trt_{timestamp}.json"

    if not os.path.isdir(args.engine_dir):
        print(f"ERROR: Specified engine directory does not exist: {args.engine_dir}")
    else:
        run_full_benchmark_trt(engine_dir_base=args.engine_dir, output_filename=output_file)