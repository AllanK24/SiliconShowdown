#!/usr/bin/env python3
import platform
import subprocess
import os
import json
import sys

def collect_system_info(output_path):
    info = {}

    # macOS version
    info["OS"] = platform.system()
    info["OS Version"] = platform.mac_ver()[0]

    # CPU model
    try:
        cpu_model = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        info["CPU"] = cpu_model
    except Exception:
        info["CPU"] = "Unknown"

    # Total RAM
    try:
        ram_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip())
        ram_gb = ram_bytes / (1024 ** 3)
        info["RAM"] = f"{ram_gb:.2f} GB"
    except Exception:
        info["RAM"] = "Unknown"

    # Number of CPU cores
    try:
        cores = subprocess.check_output(
            ["sysctl", "-n", "hw.ncpu"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        info["CPU Cores"] = cores
    except Exception:
        info["CPU Cores"] = "Unknown"

    # Python executable and version
    info["Python Executable"] = sys.executable
    info["Python Version"] = platform.python_version()

    # Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"✅ System information saved to {output_path}")


if __name__ == "__main__":

    # 1. Determine a fixed output location under project-root/benchmark/
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    bench_dir = os.path.join(project_root, "benchmark")
    output_file = os.path.join(bench_dir, "system_info.json")

    # 2. Collect info and write it
    collect_system_info(output_file)