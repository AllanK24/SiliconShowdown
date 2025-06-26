# setup_dev_env_wsl.ps1
# This script sets up a development environment to run the benchmarking in WSL (Windows Subsystem for Linux) with Ubuntu.

# -------------------- USER‑TUNABLE OPTIONS --------------------
$Distro   = "Ubuntu-24.04"   # Change only if you installed a different Ubuntu variant
$WSLUser  = "benchmark"      # Dedicated benchmarking account
$WSLPass  = "benchmark2025"  # Password to set / verify
# --------------------------------------------------------------

Write-Host "`n🖥️  Silicon Showdown WSL bootstrapper`n" -ForegroundColor Cyan

# 1. Ensure WSL exists ----------------------------------------------------
$wslStatus = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ WSL is *not* installed. Please run 'wsl --install -d $Distro' first." -ForegroundColor Red
    exit 1
}
Write-Host "✅ WSL detected – proceeding ..." -ForegroundColor Green

# 2. Ensure requested distro exists --------------------------------------
$installed = (wsl --list --quiet) -contains $Distro
if (-not $installed) {
    Write-Host "📥 Installing distro $Distro ..."
    wsl --install -d $Distro --no-launch # Prevents interactive setup
    # *** SYNTAX FIX IS HERE ***
    # This 'if' statement is now a proper block, and the extra brace is removed.
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install $Distro" -ForegroundColor Red
        exit 1
    }
}

# 3.  Optimise WSL resource limits ----------------------------------------
Write-Host "🛠️  Updating .wslconfig (max CPU / RAM)"
$wslConfigPath = Join-Path $HOME '.wslconfig'
$ramGB    = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$cpuCount = [Environment]::ProcessorCount
$wslConfig = @"
[wsl2]
memory=${ramGB}GB  # use all RAM
processors=$cpuCount
swap=0
"@

Set-Content -Path $wslConfigPath -Value $wslConfig -Encoding UTF8 -Force
Write-Host "✅ Saved $wslConfigPath"

Write-Host "🔄 Restarting WSL to apply the config ..."
wsl --shutdown
Start-Sleep -Seconds 3

# 4. Main Environment Setup ---------------------------------------------
Write-Host "`n🛠️  Beginning main environment setup inside WSL. This will take a significant amount of time." -ForegroundColor Cyan

# Define the setup script to be run in WSL
$SetupScript = @'
#!/bin/bash

set -e

# --- 1. System Preparation ---
echo "--- Updating system packages ---"
# Use DEBIAN_FRONTEND to prevent interactive prompts during upgrades
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
# Use 'upgrade' instead of 'full-upgrade' for safer automation.
# The interactive 'do-release-upgrade' has been removed as it's unsuitable for scripts.
sudo apt-get upgrade -y

# Install essential packages (change Python version here)
echo "--- Installing essential build and dev packages ---"
sudo apt install -y build-essential cmake git wget curl unzip \
    python3 python3-pip python3-venv libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev libssl-dev libyaml-cpp-dev git-lfs libopenmpi-dev \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libreadline-dev libffi-dev libbz2-dev libsqlite3-dev liblzma-dev 

# --- 2. GPU Environment (CUDA & cuDNN) ---
echo "--- Installing Git LFS ---"
git lfs install

echo "--- Installing CUDA Toolkit for WSL ---"

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

# --- PowerShell Execution Logic ---
$TempScriptPath = Join-Path $env:TEMP "wsl_setup.sh"
$SetupScript | Out-File -FilePath $TempScriptPath -Encoding UTF8

Write-Host "Copying setup script to WSL..."
Get-Content -Path $TempScriptPath -Raw | wsl -d $Distro -u $WSLUser -- bash -c "cat > /tmp/setup.sh"
wsl -d $Distro -u $WSLUser -- bash -c "chmod +x /tmp/setup.sh"

Write-Host "Running setup script inside WSL as user '$WSLUser'..."
$WslCommand = "echo '$WSLPass' | sudo -S /tmp/setup.sh"
wsl -d $Distro -u $WSLUser -- /bin/bash -c "$WslCommand"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ The WSL setup script failed. Please review the output above for errors." -ForegroundColor Red
    exit 1
}
Write-Host "✅ WSL setup script completed." -ForegroundColor Green


# 5. Verify setup and copy project files ------------------------------
$BaseWSLPath = "/home/$WSLUser/benchmark_env"

# Check if venv exists
$CheckVenv = wsl -d $Distro -u $WSLUser -- bash -c "test -d $BaseWSLPath/.venv && echo 'exists'"
if ($CheckVenv -ne "exists") {
    Write-Host "❌ Virtual environment not found at $BaseWSLPath/.venv. Setup failed." -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ Virtual environment verified." -ForegroundColor Green
}

# Function to copy files to WSL to reduce code duplication
function Copy-FileToWSL {
    param (
        [string]$LocalPath,
        [string]$WslPath,
        [bool]$Executable = $false
    )
    if (Test-Path $LocalPath) {
        Write-Host "📋 Copying $(Split-Path $LocalPath -Leaf) to WSL..."
        Get-Content -Path $LocalPath -Raw | wsl -d $Distro -u $WSLUser -- bash -c "cat > $WslPath"
        if ($Executable) {
            wsl -d $Distro -u $WSLUser -- bash -c "chmod +x $WslPath"
        }
        Write-Host "✅ Copied successfully."
    } else {
        Write-Host "⚠️ Could not find $(Split-Path $LocalPath -Leaf). Skipping copy." -ForegroundColor Yellow
    }
}

Copy-FileToWSL -LocalPath ".\run_benchmark.py" -WslPath "$BaseWSLPath/run_benchmark.py" -Executable $true
Copy-FileToWSL -LocalPath ".\run_benchmark_tensorrt_llm.py" -WslPath "$BaseWSLPath/run_benchmark_tensorrt_llm.py" -Executable $true
Copy-FileToWSL -LocalPath ".\config.yaml" -WslPath "$BaseWSLPath/config.yaml"
Copy-FileToWSL -LocalPath ".\collect_system_info.py" -WslPath "$BaseWSLPath/collect_system_info.py" -Executable $true


# 6. Run info collection and retrieve results -----------------------------
Write-Host "🚀 Running final system info collection..."
$SystemInfoScriptWSLPath = "$BaseWSLPath/collect_system_info.py"
$SystemInfoOutputWSL = "$BaseWSLPath/system_info_nvidia.json"

$WslExecCommand = "source $BaseWSLPath/.venv/bin/activate; python3 $SystemInfoScriptWSLPath"
wsl -d $Distro -u $WSLUser -- /bin/bash -c "$WslExecCommand"

# Create local 'results' directory and copy the output file
$ResultsDir = Join-Path (Get-Location) "results"
New-Item -ItemType Directory -Path $ResultsDir -ErrorAction SilentlyContinue
$SystemInfoOutputWindows = Join-Path $ResultsDir "system_info_nvidia.json"

wsl -d $Distro -u $WSLUser -- cat $SystemInfoOutputWSL > $SystemInfoOutputWindows

if (Test-Path $SystemInfoOutputWindows -and (Get-Item $SystemInfoOutputWindows).Length -gt 0) {
    Write-Host "✅ System info saved to results\system_info_nvidia.json" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to retrieve system info file from WSL." -ForegroundColor Red
}

Write-Host "`n🎉 All done! Your WSL environment is ready for benchmarking." -ForegroundColor Cyan