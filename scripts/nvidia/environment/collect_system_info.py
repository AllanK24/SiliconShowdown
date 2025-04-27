import platform
import subprocess
import sys
import json

def collect_system_info(venv_python, output_file="system_info.json"):
    info = {}

    # OS info
    info["os_system"] = platform.system()
    info["os_release"] = platform.release()
    info["os_version"] = platform.version()

    # CPU info
    info["cpu"] = platform.processor()

    # RAM info (optional, if psutil is installed)
    try:
        import psutil
        ram = psutil.virtual_memory()
        info["ram_total_gb"] = round(ram.total / (1024**3), 2)
    except ImportError:
        info["ram_total_gb"] = "psutil not installed"

    # Python version
    info["python_version"] = sys.version

    # CUDA Version (via nvcc if available)
    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True)
        info["cuda_version"] = output
    except Exception:
        info["cuda_version"] = "nvcc not found, checking PyTorch..."

    # PyTorch and device info
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except ImportError:
        info["torch_version"] = "PyTorch not installed"
        info["cuda_available"] = False

    # Pip freeze
    try:
        result = subprocess.check_output([venv_python, "-m", "pip", "freeze"], text=True)
        info["pip_packages"] = result.splitlines()
    except Exception:
        info["pip_packages"] = "Failed to collect pip freeze"

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(info, f, indent=4)

    print(f"System information collected and saved to {output_file}")
