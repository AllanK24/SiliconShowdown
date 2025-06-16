# setup_dev_env_wsl.ps1
# This script sets up a development environment to run the benchmarking in WSL (Windows Subsystem for Linux) with Ubuntu.

# -------------------- USER‑TUNABLE OPTIONS --------------------
$Distro   = "Ubuntu-24.04"   # Change only if you installed a different Ubuntu variant
$WSLUser  = "benchmark"      # Dedicated benchmarking account
$WSLPass  = "benchmark2025"  # Password to set / verify
# --------------------------------------------------------------

Write-Host "`n🖥️  Silicon Showdown WSL bootstrapper`n" -ForegroundColor Cyan

# 1️⃣  Ensure WSL exists ----------------------------------------------------
$wslStatus = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ WSL is *not* installed. Please run 'wsl --install -d $Distro' first." -ForegroundColor Red
    exit 1
}
Write-Host "✅ WSL detected – proceeding ..." -ForegroundColor Green

# 2️⃣  Ensure requested distro exists --------------------------------------
$installed = (wsl --list --quiet) -contains $Distro
if (-not $installed) {
    Write-Host "📥 Installing distro $Distro ..."
    wsl --install -d $Distro
    if ($LASTEXITCODE -ne 0) { Write-Host "❌ Failed to install $Distro" -ForegroundColor Red; exit 1 }
}

# 3️⃣  Create / validate benchmarking user ---------------------------------
Write-Host "👤 Checking user '$WSLUser' inside $Distro ..."
$idCmd = "id -u $WSLUser >/dev/null 2>&1 && echo exists || echo missing"
$exists = (wsl -d $Distro -- bash -c $idCmd).Trim()
if ($exists -eq "missing") {
    Write-Host "➕ Creating user '$WSLUser' ..."
    $addUser = @"bash -c "adduser --disabled-password --gecos '' $WSLUser && echo '$WSLUser:$WSLPass' | chpasswd && usermod -aG sudo $WSLUser""@
    wsl -d $Distro --user root -- $addUser
    if ($LASTEXITCODE -ne 0) { Write-Host "❌ Failed to create user" -ForegroundColor Red; exit 1 }
    Write-Host "✅ User created with sudo rights."
} else {
    Write-Host "✅ User already present – resetting password just in case ..."
    wsl -d $Distro --user root -- bash -c "echo '$WSLUser:$WSLPass' | chpasswd"
}

# 4️⃣  Optimise WSL resource limits ----------------------------------------
Write-Host "🛠️  Updating .wslconfig (max CPU / RAM, GPU‑offload) ..."
$wslConfigPath = Join-Path $HOME '.wslconfig'
$ramGB    = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$cpuCount = [Environment]::ProcessorCount
$wslConfig = @"
[wsl2]
memory=${ramGB}GB  # use all RAM
processors=$cpuCount
swap=0
"@
Set-Content -Path $wslConfigPath -Value $wslConfig -Encoding ASCII -Force
Write-Host "✅ Saved $wslConfigPath"

Write-Host "🔄 Restarting WSL to apply the config ..."
wsl --shutdown
Start-Sleep -Seconds 2

# Define the setup script to be run in WSL
$SetupScript = @'
#!/bin/bash

set -e

sudo apt update && sudo apt upgrade -y
# Install essential packages (change Python version here)
sudo apt install -y build-essential cmake git wget curl unzip \
    python3 python3-pip python3-venv libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev libssl-dev libyaml-cpp-dev git-lfs libopenmpi-dev \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libreadline-dev libffi-dev libbz2-dev libsqlite3-dev liblzma-dev 

### Install git-lfs
echo "Installing Git LFS..."
git lfs install

### CUDA Toolkit Installation
echo "Installing CUDA Toolkit 12.9 for WSL2..."

sudo apt-key del 7fa2af80 || true

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-wsl-ubuntu-12-9-local_12.9.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-9-local_12.9.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

### Install cuDNN
echo "Installing cuDNN for CUDA 12.9..."

## Install the cuDNN based on the Ubuntu version
# Safely extract version from /etc/os-release
version=$(grep '^VERSION_ID=' /etc/os-release | cut -d '"' -f 2)
# Print the version
echo "Detected Ubuntu version: $version"

# Compare against supported versions
if [[ "$version" == "20.04" ]]; then
    wget https://developer.download.nvidia.com/compute/cudnn/9.10.1/local_installers/cudnn-local-repo-ubuntu2004-9.10.1_1.0-1_amd64.deb
    sudo dpkg -i cudnn-local-repo-ubuntu2004-9.10.1_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-ubuntu2004-9.10.1/cudnn-*-keyring.gpg /usr/share/keyrings/
elif [[ "$version" == "22.04" ]]; then
    wget https://developer.download.nvidia.com/compute/cudnn/9.10.1/local_installers/cudnn-local-repo-ubuntu2204-9.10.1_1.0-1_amd64.deb
    sudo dpkg -i cudnn-local-repo-ubuntu2204-9.10.1_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-ubuntu2204-9.10.1/cudnn-*-keyring.gpg /usr/share/keyrings/
elif [[ "$version" == "24.04" ]]; then
    wget https://developer.download.nvidia.com/compute/cudnn/9.10.1/local_installers/cudnn-local-repo-ubuntu2404-9.10.1_1.0-1_amd64.deb
    sudo dpkg -i cudnn-local-repo-ubuntu2404-9.10.1_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-ubuntu2404-9.10.1/cudnn-*-keyring.gpg /usr/share/keyrings/
else
    echo "Unknown or unsupported Ubuntu version: $version"
    echo "Please install cuDNN manually for your version of Ubuntu."
    exit 1
fi

sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12

# Check if CUDA and cuDNN are installed correctly
if command -v nvcc --version &>/dev/null && command -v nvidia-smi &>/dev/null; then
    echo "✅ CUDA and cuDNN installed successfully."
else
    echo "❌ CUDA or cuDNN installation failed. Please check the logs."
    exit 1
fi

### Install Python packages and create virtual environment ###
echo "Setting up Python environment..."

# Create a folder to install Python and envs
echo "Creating Python environment directory..."
mkdir -p ~/benchmark_env/python
cd ~/benchmark_env/python

# Download and build Python 3.11.0 from source
echo "Downloading and building Python 3.11.0..."
PYTHON_VERSION=3.11.0
curl -O https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar -xzf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION

./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall  # installs as python3.11 without overwriting system python

# Create virtual environment
echo "Creating virtual environment..."
cd ~/benchmark_env
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install required Python packages
echo "Installing Python packages..."
pip install --upgrade pip

# Define the list of Python packages in a variable
PYTHON_PACKAGES="torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 huggingface_hub transformers pynvml matplotlib --extra-index-url https://pypi.nvidia.com/ tensorrt-llm psutil datasets evaluate rouge_score transformers-stream-generator sentencepiece tiktoken einops h5py safetensors flax pyyaml"

pip install $PYTHON_PACKAGES

# Clone TensorRT LLM repository
echo "Cloning TensorRT LLM repository..."
git clone https://github.com/NVIDIA/TensorRT-LLM.git

echo "✅ WSL environment setup complete."
'@

# Write setup script to a temp file
$TempScriptPath = "$env:TEMP\wsl_setup.sh"
$SetupScript | Out-File -Encoding ASCII -FilePath $TempScriptPath

Write-Host "Copying setup script to WSL..."
wsl -d $Distro -- bash -c "mkdir -p /home/benchmark/setup"
wsl -d $Distro -- bash -c "rm -f /home/benchmark/setup/setup.sh"
Get-Content -Path $TempScriptPath -Raw | wsl -d $Distro -- bash -c "cat > /home/benchmark/setup/setup.sh"
wsl -d $Distro -- bash -c "chmod +x /home/benchmark/setup/setup.sh"

Write-Host "Running setup inside WSL..."
wsl -d $Distro --user root -- bash -c "/home/benchmark/setup/setup.sh"

# Check venv exists
$CheckVenv = wsl -d $Distro --user $WSLUser -- bash -c "test -d /home/benchmark/benchmark_env/.venv && echo 'venv_exists'"
if (-not ($CheckVenv -eq "venv_exists")) {
    Write-Host "❌ Virtual environment not found. Something went wrong during setup."
    exit 1
}else {
    Write-Host "✅ Virtual environment is set up successfully."
}

# Copy the run_benchmark.py file into WSL
$BenchmarkScriptLocalPath = ".\run_benchmark.py"  # Change path if needed
$BenchmarkScriptWSLPath = "/home/benchmark/benchmark_env/run_benchmark.py"

if (Test-Path $BenchmarkScriptLocalPath) {
    Write-Host "📋 Copying run_benchmark.py into WSL benchmarking folder..."
    wsl -d $Distro --user $WSLUser -- bash -c "rm -f $BenchmarkScriptWSLPath" # Remove any existing file
    Get-Content -Path $BenchmarkScriptLocalPath -Raw | wsl -d $Distro --user $WSLUser -- bash -c "cat > $BenchmarkScriptWSLPath"
    wsl -d $Distro --user $WSLUser -- bash -c "chmod +x $BenchmarkScriptWSLPath" # Make it executable
    Write-Host "✅ run_benchmark.py copied successfully."
} else {
    Write-Host "⚠️ Could not find run_benchmark.py in current directory. Skipping copy."
}

# Copy run_benchmark_tensorrt_llm.py into WSL
$TensorRTScriptLocalPath = ".\run_benchmark_tensorrt_llm.py"
$TensorRTScriptWSLPath = "/home/benchmark/benchmark_env/run_benchmark_tensorrt_llm.py"

if (Test-Path $TensorRTScriptLocalPath) {
    Write-Host "📋 Copying run_benchmark_tensorrt_llm.py into WSL benchmarking folder..."
    wsl -d $Distro --user $WSLUser -- bash -c "rm -f $TensorRTScriptWSLPath"
    Get-Content -Path $TensorRTScriptLocalPath -Raw | wsl -d $Distro --user $WSLUser -- bash -c "cat > $TensorRTScriptWSLPath"
    wsl -d $Distro --user $WSLUser -- bash -c "chmod +x $TensorRTScriptWSLPath"
    Write-Host "✅ run_benchmark_tensorrt_llm.py copied successfully."
} else {
    Write-Host "⚠️ Could not find run_benchmark_tensorrt_llm.py in current directory. Skipping copy."
}

# Copy config.yaml into WSL
$ConfigYamlLocalPath = ".\config.yaml"
$ConfigYamlWSLPath = "/home/benchmark/benchmark_env/config.yaml"

if (Test-Path $ConfigYamlLocalPath) {
    Write-Host "📋 Copying config.yaml into WSL benchmarking folder..."
    wsl -d $Distro --user $WSLUser -- bash -c "rm -f $ConfigYamlWSLPath"
    Get-Content -Path $ConfigYamlLocalPath -Raw | wsl -d $Distro --user $WSLUser -- bash -c "cat > $ConfigYamlWSLPath"
    Write-Host "✅ config.yaml copied successfully."
} else {
    Write-Host "⚠️ Could not find config.yaml in current directory. Skipping copy."
}

# Copy collect_system_info.py into WSL
$SystemInfoScriptLocalPath = ".\collect_system_info.py"  # Adjust path if needed
$SystemInfoScriptWSLPath = "/home/benchmark/benchmark_env/collect_system_info.py"

if (Test-Path $SystemInfoScriptLocalPath) {
    Write-Host "📋 Copying collect_system_info.py into WSL benchmarking folder..."
    wsl -d $Distro --user $WSLUser -- bash -c "rm -f $SystemInfoScriptWSLPath"
    Get-Content -Path $SystemInfoScriptLocalPath -Raw | wsl -d $Distro --user $WSLUser -- bash -c "cat > $SystemInfoScriptWSLPath"
    wsl -d $Distro --user $WSLUser -- bash -c "chmod +x $SystemInfoScriptWSLPath"
    Write-Host "✅ collect_system_info.py copied successfully."
} else {
    Write-Host "⚠️ Could not find collect_system_info.py. Skipping."
}

Write-Host "🚀 Running collect_system_info.py inside WSL..."

# Define WSL command to run the script and write output to known path
$SystemInfoOutputWSL = "/home/benchmark/benchmark_env/system_info_nvidia.json"
$SystemInfoOutputWindows = "$env:USERPROFILE\Desktop\system_info_nvidia.json"

wsl -d $Distro --user $WSLUser -- bash -c "
    source /home/benchmark/benchmark_env/.venv/bin/activate && \
    python3 /home/benchmark/benchmark_env/collect_system_info.py > /dev/null
"

# Copy result from WSL to a results/ folder in the current script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ResultsDir = Join-Path $ScriptDir "results"
$SystemInfoOutputWindows = Join-Path $ResultsDir "system_info_nvidia.json"

# Create results folder if it doesn't exist
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

Write-Host "💾 Copying system_info_nvidia.json to $ResultsDir..."
wsl -d $Distro --user $WSLUser -- bash -c "
    cat $SystemInfoOutputWSL
" > $SystemInfoOutputWindows

if (Test-Path $SystemInfoOutputWindows) {
    Write-Host "✅ system_info_nvidia.json saved to results/ folder."
} else {
    Write-Host "❌ Failed to save system_info_nvidia.json. Please check script output."
}

Write-Host "`n🎉 All done! Your WSL environment is ready for benchmarking."