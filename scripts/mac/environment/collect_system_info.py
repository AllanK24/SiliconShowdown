import platform
import subprocess
import os

def collect_system_info(output_file="system_info.txt"):
    info = {}

    # macOS version
    info["OS"] = platform.system()
    info["OS Version"] = platform.mac_ver()[0]

    # CPU model
    try:
        cpu_model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        info["CPU"] = cpu_model
    except Exception:
        info["CPU"] = "Unknown"

    # Total RAM
    try:
        ram_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        ram_gb = ram_bytes / (1024**3)
        info["RAM"] = f"{ram_gb:.2f} GB"
    except Exception:
        info["RAM"] = "Unknown"

    # Number of CPU cores
    try:
        cores = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
        info["CPU Cores"] = cores
    except Exception:
        info["CPU Cores"] = "Unknown"

    # Make sure output folder exists
    output_dir = os.path.join(os.getcwd(), "system_info")
    os.makedirs(output_dir, exist_ok=True)

    # Save file inside system_info/
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

    print(f"✅ System information saved to {output_file}")

if __name__ == "__main__":
    collect_system_info()
