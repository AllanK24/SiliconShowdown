import platform
import subprocess
import sys
import json

def get_nvidia_driver_version():
    """Attempts to get the Nvidia driver version using nvidia-smi."""
    try:
        # Query specifically for driver version, format for easy parsing
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,compute_mode,clocks_throttle_reasons.supported", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL # Hide errors if command fails
        )
        # Split the output by commas and strip whitespace
        return output.strip().split(', ')
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as e:
        print(f"Could not get Nvidia driver version via nvidia-smi: {e}")
        return "nvidia-smi not found or failed"
    except Exception as e:
        print(f"Unexpected error getting driver version: {e}")
        return "Error fetching driver version"

def get_cuda_version_nvcc():
    """Attempts to get the CUDA toolkit version using nvcc."""
    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True)
        # Extract version line, e.g., "Cuda compilation tools, release 11.8, V11.8.89"
        for line in output.splitlines():
            if "release" in line:
                return line.strip()
        return "Version line not found in nvcc output"
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Could not get CUDA version via nvcc: {e}")
        return "nvcc not found or failed"
    except Exception as e:
        print(f"Unexpected error getting nvcc version: {e}")
        return "Error fetching nvcc version"

def collect_system_info_nvidia(venv_python, output_file="system_info_nvidia.json"):
    """
    Collects system information relevant for Nvidia GPU benchmarking.

    Args:
        venv_python (str): The absolute path to the Python executable within
                           the virtual environment being used for benchmarks.
        output_file (str): The path to save the JSON output file.
    """
    info = {}
    print(f"Collecting system info using Python: {venv_python}")

    # --- OS Info ---
    info["os_system"] = platform.system()
    info["os_release"] = platform.release()
    info["os_version"] = platform.version()
    info["os_platform"] = platform.platform() # More detailed platform string
    
    # --- Hardware Info ---
    info["cpu_architecture"] = platform.machine()
    info["cpu_model"] = platform.processor() # Best effort for CPU model

    try:
        import psutil
        ram = psutil.virtual_memory()
        info["system_ram_total_gb"] = round(ram.total / (1024**3), 2)
        
        # Disk Type Info (SSD/HDD)
        disk_info = psutil.disk_partitions()
        for partition in disk_info:
            if partition.device.startswith('C:'):
                info['disk_type'] = "SSD"
    except ImportError:
        info["system_ram_total_gb"] = "psutil not installed"
        print("Warning: psutil not found. System RAM and Disk Type info unavailable.")

    # --- Python Environment ---
    info["benchmark_python_executable"] = venv_python
    info["benchmark_python_version"] = sys.version # Version of Python running *this* script

    # --- Nvidia Specific Info ---
    fields = get_nvidia_driver_version()
    # Assign parsed values
    info['nvidia_driver_version'] = fields[0] if len(fields) > 0 else None
    info['gpu_compute_mode'] = fields[1] if len(fields) > 1 else None
    info['thermal_throttling_detected'] = fields[2].strip().lower() == 'enabled' if len(fields) > 2 else None

    # --- PyTorch & CUDA Info ---
    try:
        import torch
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            # This correctly gets GPU VRAM
            info["gpu_vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            info["gpu_cuda_capability"] = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
        else:
             # This case shouldn't happen based on user guarantee, but good to have
            info["torch_cuda_version_compiled"] = None
            info["gpu_name"] = "CUDA not detected by PyTorch"
            info["gpu_vram_total_gb"] = None
            info["gpu_cuda_capability"] = None
            print("ERROR: PyTorch reports CUDA is not available!")

    except ImportError:
        info["gpu_name"] = "PyTorch not installed"
        info["gpu_vram_total_gb"] = None
        info["gpu_cuda_capability"] = None
        print("ERROR: PyTorch not found. Cannot get GPU/CUDA details via Torch.")

    # --- System CUDA Toolkit Version (Optional Supplement) ---
    info["system_cuda_version_nvcc"] = get_cuda_version_nvcc()

    
    # --- Save to JSON ---
    try:
        with open(output_file, "w") as f:
            json.dump(info, f, indent=4, sort_keys=True) # Sort keys for readability
        print(f"System information collected and saved to {output_file}")
    except Exception as e:
        print(f"ERROR saving system info to JSON: {e}")
        
        
if __name__ == "__main__":
    # Example usage: pass the path to the Python executable in the virtual environment
    venv_python = sys.executable  # This will be the Python executable of the current environment
    collect_system_info_nvidia(venv_python)