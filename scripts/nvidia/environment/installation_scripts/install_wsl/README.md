# 🧪 AI Benchmarking Setup Guide (WSL + Windows)

This guide will walk you through installing WSL, configuring your system, and running the AI benchmarking scripts required for the **Silicon Showdown** project.

> 📝 **Important:** These instructions assume you are running **Windows 10/11** with administrator privileges.

---

## ✅ Step-by-Step Instructions

### 🧭 1. Open PowerShell as Administrator

#### 🔹 How to open it:
1. Click on the **Start** menu (Windows logo).
2. Type `powershell`.
3. **Right-click** on **Windows PowerShell** (or **Terminal** on Windows 11).
4. Choose **"Run as administrator"**.

> ⚠️ You **must** run it as Administrator to install WSL and make system-level changes.

---

### 2. Install WSL2 with Ubuntu (if not already installed)

> If you already have Ubuntu installed in WSL2, you can skip this step.

```powershell
wsl --install
```

> This will install WSL

---

### Restart Computer
```powershell
Restart-Computer -Force
```

### Open PowerShell as administrator again, run the following command to install Ubuntu 24.04:
```powershell
wsl --install Ubuntu-24.04
```

### If you have WSL installed, run:
```powershell
wsl --update
```
to update WSL

---

### 3. Allow WSL to use max resources
```powershell
# This script automatically creates or updates the .wslconfig file to maximize performance.

Write-Host "`n🛠️  Step 1: Automatically detecting your system's RAM and CPU..." -ForegroundColor Cyan

# Get the total RAM in Gigabytes (GB) and round it to a whole number.
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)

# Get the number of CPU logical processors (cores/threads).
$cpuCount = [Environment]::ProcessorCount

Write-Host " - Detected: $ramGB GB of RAM"
Write-Host " - Detected: $cpuCount CPU processors"

# Define the full path to the .wslconfig file in your user profile.
$wslConfigPath = Join-Path $HOME '.wslconfig'

# Create the content for the .wslconfig file using the detected values.
$wslConfigContent = @"
[wsl2]
memory=${ramGB}GB
processors=$cpuCount
swap=0
"@

Write-Host "`n💾  Step 2: Saving the configuration to the file at '$wslConfigPath'..." -ForegroundColor Cyan

# Save the content to the file. -Force overwrites it if it already exists.
try {
    Set-Content -Path $wslConfigPath -Value $wslConfigContent -Encoding UTF8 -Force
    Write-Host "✅ Successfully saved the .wslconfig file." -ForegroundColor Green
}
catch {
    Write-Host "❌ An error occurred while trying to save the file. Please check permissions." -ForegroundColor Red
    exit 1
}

# Shutdown WSL to apply changes
wsl --shutdown
```

### 4. Install required dependencies for benchmarking
```bash
sudo apt update
sudo apt upgrade

sudo apt install -y build-essential cmake git wget curl unzip \
    python3 python3-pip python3-venv libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev libssl-dev libyaml-cpp-dev git-lfs libopenmpi-dev \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libreadline-dev libffi-dev libbz2-dev libsqlite3-dev liblzma-dev 

git lfs install
```

#### CUDA
```bash
sudo apt-key del 7fa2af80

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb # we are here with Emir
sudo dpkg -i cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
sudo cp /usr/lib/wsl/lib/nvidia-smi /usr/bin/nvidia-smi
sudo chmod +x /usr/bin/nvidia-smi
```
#### Check if nvidia-smi was installed:
```bash
nvidia-smi
```
*If successfull, and you can see the info about your GPU, continue with cuDNN installation, else try all the steps to install CUDA again*

#### cuDNN

```bash

### If you have Ubuntu 20.04, run the following to install cuDNN
wget https://developer.download.nvidia.com/compute/cudnn/9.10.1/local_installers/cudnn-local-repo-ubuntu2004-9.10.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.10.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.10.1/cudnn-*-keyring.gpg /usr/share/keyrings/

### If you have Ubuntu 22.04, run the following to install cuDNN
wget https://developer.download.nvidia.com/compute/cudnn/9.10.1/local_installers/cudnn-local-repo-ubuntu2204-9.10.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.10.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.10.1/cudnn-*-keyring.gpg /usr/share/keyrings/

### If you have Ubuntu 24.04, run the following to install cuDNN
wget https://developer.download.nvidia.com/compute/cudnn/9.10.1/local_installers/cudnn-local-repo-ubuntu2404-9.10.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.10.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.10.1/cudnn-*-keyring.gpg /usr/share/keyrings/

### Install the cuDNN package
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12
sudo apt install nvidia-cuda-toolkit

### Verify cuDNN installation
nvcc --version

### If you see something like this:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
### it means it was installed successfully.
```

#### Download and setup Python:
```bash
mkdir -p ~/benchmark_env/python
cd ~/benchmark_env/python

curl -O https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz
tar -xzf Python-3.11.0.tgz
cd Python-3.11.0

./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall  # installs as python3.11 without overwriting system python

### Check if python is installed
python3.11 --version
```

---

### 5. Setup the environment for development
