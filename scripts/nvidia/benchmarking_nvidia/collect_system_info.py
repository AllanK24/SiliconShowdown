# collect_system_info.py
# Collects system information for NVIDIA GPU environments before running benchmarks.
# This script is designed to run inside WSL2 and gather relevant system information.

import re
import os
import psutil
import platform
import subprocess
import sys
import json

def _windows_safe_cwd() -> str:
    """
    Return a directory on the Windows C: drive that certainly exists.
    Falls back to C:\\ if nothing else is found.
    """
    for p in ["/mnt/c/Temp", "/mnt/c/Windows", "/mnt/c"]:
        if os.path.isdir(p):
            return p
    return "/mnt/c"

def get_windows_host_info():
    """Collect Windows host OS info from inside WSL2."""
    info = {}
    cwd = _windows_safe_cwd()         
    try:
        # --- Windows build & edition via PowerShell (modern and WMIC-free) ---
        ps_script = (
            "(Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption, BuildNumber, Version | "
            "ConvertTo-Json -Compress)"
        )
        raw = subprocess.check_output(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", ps_script],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        # PowerShell emits UTF-16LE by default; decode if needed
        if raw and raw[0] == '\x00':
            raw = raw.encode("latin1").decode("utf-16le")
        cim = json.loads(raw)
        info["windows_edition"] = cim.get("Caption", "").strip()
        info["windows_build"]   = cim.get("BuildNumber", "").strip()
        info["windows_ver"]     = cim.get("Version", "").strip()

    except Exception as e:
        # Fallback to cmd -- still force a CWD on C: \
        try:
            ver_output = subprocess.check_output(
                ["cmd.exe", "/c", "ver"], cwd=cwd, text=True
            ).strip()
            info["windows_ver"] = ver_output
        except Exception:
            pass
        info.setdefault("windows_info_error", str(e))

    return info

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

def is_wsl_environment():
    """Detects if the script is running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "Microsoft" in f.read() or "WSL" in platform.release()
    except FileNotFoundError:
        return False

def get_linux_distro_info():
    """Gets the Linux distribution name and version (e.g., Ubuntu 20.04)."""
    try:
        output = subprocess.check_output(["lsb_release", "-a"], stderr=subprocess.DEVNULL, text=True)
        info = {}
        for line in output.splitlines():
            if "Distributor ID" in line:
                info["distro"] = line.split(":")[1].strip()
            elif "Description" in line:
                info["distro_description"] = line.split(":")[1].strip()
            elif "Release" in line:
                info["distro_version"] = line.split(":")[1].strip()
        return info
    except Exception as e:
        return {"distro": "Unknown", "distro_version": "Unknown", "distro_description": "Unavailable"}
    
def get_cpu_model_linux() -> str:
    """Return the first 'model name' found in /proc/cpuinfo or lscpu."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass

    # fall back to lscpu
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        m = re.search(r"Model name:\s+(.*)", out)
        if m:
            return m.group(1).strip()
    except Exception:
        pass

    # last-ditch fallback
    return platform.processor() or "Unknown CPU"

def add_cpu_details(info: dict):
    """Enrich the info dict with CPU model and core counts."""
    info["cpu_model"]            = get_cpu_model_linux()
    info["cpu_cores_physical"]   = psutil.cpu_count(logical=False)
    info["cpu_cores_logical"]    = psutil.cpu_count(logical=True)

# Modify this function as needed for full functionality
def collect_system_info_nvidia(venv_python, output_file="system_info_nvidia.json"):
    info = {}
    print(f"Collecting system info using Python: {venv_python}")

    # --- Detect Environment ---
    info["running_in_wsl"] = is_wsl_environment()

    # --- OS Info ---
    info["os_system"] = platform.system()
    info["os_release"] = platform.release()
    info["os_version"] = platform.version()
    info["os_platform"] = platform.platform()

    if info["running_in_wsl"]:
        info.update(get_linux_distro_info())
        info.update(get_windows_host_info())

    # --- CPU Info ---
    add_cpu_details(info)
    info["cpu_architecture"] = platform.machine()

    try:
        import psutil
        ram = psutil.virtual_memory()
        info["system_ram_total_gb"] = round(ram.total / (1024**3), 2)

        # In Linux, check "/" mount for disk
        disk_info = psutil.disk_partitions()
        for partition in disk_info:
            if partition.mountpoint == "/":
                info["disk_mount_device"] = partition.device
                info["disk_type_guess"] = "Likely SSD or virtual disk"
    except ImportError:
        info["system_ram_total_gb"] = "psutil not installed"
        print("Warning: psutil not found. System RAM and Disk Type info unavailable.")

    # --- Python Environment ---
    info["benchmark_python_executable"] = venv_python
    info["benchmark_python_version"] = sys.version

    # --- Nvidia Specific Info ---
    fields = get_nvidia_driver_version()
    info['nvidia_driver_version'] = fields[0] if len(fields) > 0 else None
    info['gpu_compute_mode'] = fields[1] if len(fields) > 1 else None
    info['thermal_throttling_detected'] = fields[2].strip().lower() == 'enabled' if len(fields) > 2 else None

    # --- PyTorch & CUDA Info ---
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            info["gpu_cuda_capability"] = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
        else:
            info["gpu_name"] = "CUDA not detected by PyTorch"
            info["gpu_vram_total_gb"] = None
            info["gpu_cuda_capability"] = None
            print("ERROR: PyTorch reports CUDA is not available!")
    except ImportError:
        info["gpu_name"] = "PyTorch not installed"
        info["gpu_vram_total_gb"] = None
        info["gpu_cuda_capability"] = None
        print("ERROR: PyTorch not found. Cannot get GPU/CUDA details via Torch.")

    # --- CUDA Toolkit Version ---
    info["system_cuda_version_nvcc"] = get_cuda_version_nvcc()

    # --- Save to JSON ---
    try:
        with open(output_file, "w") as f:
            json.dump(info, f, indent=4, sort_keys=True)
        print(f"System information collected and saved to {output_file}")
    except Exception as e:
        print(f"ERROR saving system info to JSON: {e}")

if __name__ == "__main__":
    collect_system_info_nvidia(sys.executable)