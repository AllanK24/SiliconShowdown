import platform
import subprocess
import sys
import json
import os # Added for clarity on venv_python path handling

def get_nvidia_driver_version():
    """Attempts to get the Nvidia driver version using nvidia-smi."""
    try:
        # Query specifically for driver version, format for easy parsing
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL # Hide errors if command fails
        )
        # Take the first line in case of multiple GPUs (drivers are system-wide)
        return output.strip().splitlines()[0]
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
    except ImportError:
        info["system_ram_total_gb"] = "psutil not installed"
        print("Warning: psutil not found. System RAM info unavailable.")

    # --- Python Environment ---
    info["benchmark_python_executable"] = venv_python
    info["benchmark_python_version"] = sys.version # Version of Python running *this* script

    # --- Nvidia Specific Info ---
    info["nvidia_driver_version"] = get_nvidia_driver_version()

    # --- PyTorch & CUDA Info ---
    try:
        import torch
        info["torch_version"] = torch.__version__

        # Check CUDA status via PyTorch (even though we assume it's available)
        info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["torch_cuda_version_compiled"] = torch.version.cuda # Version PyTorch built against
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
        info["torch_version"] = "PyTorch not installed"
        info["torch_cuda_available"] = False
        info["torch_cuda_version_compiled"] = None
        info["gpu_name"] = "PyTorch not installed"
        info["gpu_vram_total_gb"] = None
        info["gpu_cuda_capability"] = None
        print("ERROR: PyTorch not found. Cannot get GPU/CUDA details via Torch.")

    # --- System CUDA Toolkit Version (Optional Supplement) ---
    info["system_cuda_version_nvcc"] = get_cuda_version_nvcc()

    # --- Installed Packages ---
    try:
        # Ensure the path to venv python is correct
        if not os.path.isfile(venv_python):
             raise FileNotFoundError(f"Virtual environment Python not found at: {venv_python}")

        result = subprocess.check_output([venv_python, "-m", "pip", "freeze"], text=True)
        info["pip_packages"] = sorted(result.splitlines()) # Sort for consistency
    except FileNotFoundError as e:
        print(f"ERROR getting pip freeze: {e}")
        info["pip_packages"] = f"Failed - Venv Python not found: {venv_python}"
    except subprocess.CalledProcessError as e:
        print(f"ERROR running pip freeze: {e}")
        info["pip_packages"] = "Failed to run pip freeze"
    except Exception as e:
        print(f"Unexpected error getting pip freeze: {e}")
        info["pip_packages"] = "Failed - Unexpected error"

    # --- Save to JSON ---
    try:
        with open(output_file, "w") as f:
            json.dump(info, f, indent=4, sort_keys=True) # Sort keys for readability
        print(f"System information collected and saved to {output_file}")
    except Exception as e:
        print(f"ERROR saving system info to JSON: {e}")