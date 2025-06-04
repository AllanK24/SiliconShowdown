# setup_dev_env_wsl.ps1
# This script sets up a development environment to run the benchmarking in WSL (Windows Subsystem for Linux) with Ubuntu.

# Setup the distribution and user for WSL
$Distro = "Ubuntu"
$WSLUser = "benchmark"

# Check if WSL is installed
$wslStatus = wsl --status 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ WSL is installed, continuing with setup..."
} else {
    Write-Host "❌ WSL not detected. Please install WSL first before running this script."
    Read-Host "Press Enter to exit..."
    exit 1
}

# Configure WSL to use maximum resources
Write-Host "🛠️ Configuring WSL to use maximum CPU, RAM, and enable GPU support..."

$wslConfigPath = Join-Path -Path $HOME -ChildPath '.wslconfig'

$ramGB    = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$cpuCount = [Environment]::ProcessorCount   # integer

$wslConfig = @"
[wsl2]
memory=${ramGB}GB        # cap = all physical RAM
processors=$cpuCount     # all logical CPUs
swap=0                   # no swap file – optional
"@

# Write (or overwrite) the config; -Force creates the file if absent
Set-Content -Path $wslConfigPath -Value $wslConfig -Encoding ASCII -NoNewline -Force

Write-Host "✅ WSL configuration saved to $WSLConfigPath"

# Shutdown WSL to apply the config (it will restart when setup continues)
Write-Host "🔄 Shutting down WSL to apply new settings..."
wsl --shutdown
Start-Sleep -Seconds 2

# Define the setup script to be run in WSL
$SetupScript = @'
#!/bin/bash

set -e

### Create user if needed
if ! id "benchmark" &>/dev/null; then
    sudo adduser --disabled-password --gecos "" benchmark
    echo "benchmark ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/benchmark
fi

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

# Write setup script to a temp file on Windows
$TempSetupScriptWindowsPath = "$env:TEMP\wsl_initial_setup.sh" # More descriptive temp name
$SetupScript | Out-File -Encoding UTF8 -FilePath $TempSetupScriptWindowsPath

# --- Steps to get setup.sh into WSL and run it ---
Write-Host "Copying initial setup script to WSL and making it executable..."
$WSLTempSetupScriptPath = "/tmp/initial_setup_for_benchmark.sh" # Temp path inside WSL

# Ensure clean state and copy as root
wsl -d $Distro --user root -- bash -c "rm -f $WSLTempSetupScriptPath"
Get-Content -Path $TempSetupScriptWindowsPath -Raw | wsl -d $Distro --user root -- bash -c "cat > $WSLTempSetupScriptPath"
wsl -d $Distro --user root -- bash -c "chmod +x $WSLTempSetupScriptPath"

Write-Host "Running initial setup script inside WSL (as root)..."
# This script creates the 'benchmark' user, its home, sudoers, benchmark_env dir, venv, etc.
wsl -d $Distro --user root -- bash -c "$WSLTempSetupScriptPath"

# Clean up the temporary setup script from /tmp in WSL
wsl -d $Distro --user root -- bash -c "rm -f $WSLTempSetupScriptPath"
# Clean up the temporary setup script from Windows %TEMP%
Remove-Item -Path $TempSetupScriptWindowsPath -ErrorAction SilentlyContinue

Write-Host "✅ Initial WSL setup script execution finished."

# --- Verify 'benchmark' user and venv (now operating as $WSLUser = "benchmark") ---
$VenvPathWSL = "/home/$WSLUser/benchmark_env/.venv"
$CheckVenv = wsl -d $Distro --user $WSLUser -- bash -c "test -d $VenvPathWSL && echo 'venv_exists'"

if (-not ($CheckVenv -eq "venv_exists")) {
    Write-Host "❌ Virtual environment not found at $VenvPathWSL for user '$WSLUser'."
    Write-Host "   This indicates a problem with the initial setup script execution."
    Write-Host "   Diagnosing by listing relevant directories as '$WSLUser':"
    Write-Host "   Listing /home/$WSLUser/:"
    wsl -d $Distro --user $WSLUser -- bash -c "ls -la /home/$WSLUser/"
    Write-Host "   Listing /home/$WSLUser/benchmark_env/:"
    wsl -d $Distro --user $WSLUser -- bash -c "ls -la /home/$WSLUser/benchmark_env/"
    exit 1
} else {
    Write-Host "✅ Virtual environment verified at $VenvPathWSL for user '$WSLUser'."
}

# --- Copy Benchmark-Specific Files into the 'benchmark' user's environment ---
# Base path for scripts inside WSL, within the 'benchmark' user's home
$BaseWSLPathForScripts = "/home/$WSLUser/benchmark_env"

# Define local and WSL paths for each script/config
$ScriptMappings = @(
    @{ Local = ".\run_benchmark.py";                  WSL = "$BaseWSLPathForScripts/run_benchmark.py";                  Executable = $true }
    @{ Local = ".\run_benchmark_tensorrt_llm.py";     WSL = "$BaseWSLPathForScripts/run_benchmark_tensorrt_llm.py";     Executable = $true }
    @{ Local = ".\config.yaml";                       WSL = "$BaseWSLPathForScripts/config.yaml";                       Executable = $false }
    @{ Local = ".\collect_system_info.py";            WSL = "$BaseWSLPathForScripts/collect_system_info.py";            Executable = $true }
)

foreach ($mapping in $ScriptMappings) {
    $LocalPath = $mapping.Local
    $WSLPath = $mapping.WSL
    $IsExecutable = $mapping.Executable

    if (Test-Path $LocalPath) {
        Write-Host "📋 Copying '$LocalPath' to '$WSLPath' for user '$WSLUser'..."
        wsl -d $Distro --user $WSLUser -- bash -c "rm -f '$WSLPath'" # Use single quotes for WSL path
        Get-Content -Path $LocalPath -Raw | wsl -d $Distro --user $WSLUser -- bash -c "cat > '$WSLPath'"
        if ($IsExecutable) {
            wsl -d $Distro --user $WSLUser -- bash -c "chmod +x '$WSLPath'"
        }
        Write-Host "✅ '$LocalPath' copied successfully."
    } else {
        Write-Host "⚠️ Could not find '$LocalPath'. Skipping copy."
    }
}

# --- Running collect_system_info.py inside WSL as $WSLUser ---
Write-Host "🚀 Running collect_system_info.py inside WSL as user '$WSLUser'..."
$SystemInfoScriptWSLPath = "$BaseWSLPathForScripts/collect_system_info.py"
$SystemInfoOutputWSL = "$BaseWSLPathForScripts/system_info_nvidia.json"

# Command to run system info script within the venv
# For bash -c, the entire command string needs to be correctly passed.
# Using single quotes around the whole bash command helps, and escape internal single quotes if bash needs them.
# OR, ensure PowerShell variables are expanded correctly into a string that bash will interpret.
$BashCommandToRun = "source '$VenvPathWSL/bin/activate' && python3 '$SystemInfoScriptWSLPath' '$SystemInfoOutputWSL'"

Write-Host "Executing in WSL: wsl -d $Distro --user $WSLUser -- bash -c ""$BashCommandToRun""" # Note double quotes around $BashCommandToRun
wsl -d $Distro --user $WSLUser -- bash -c "$BashCommandToRun" # This passes the string $BashCommandToRun to bash -c

# --- Copy system_info_nvidia.json result from WSL to Windows ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ResultsDir = Join-Path $ScriptDir "results"
$SystemInfoOutputWindows = Join-Path $ResultsDir "system_info_nvidia.json"

if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

Write-Host "💾 Copying '$SystemInfoOutputWSL' to '$SystemInfoOutputWindows'..."
$SystemInfoContentFromWSL = wsl -d $Distro --user $WSLUser -- bash -c "cat '$SystemInfoOutputWSL'"

if ($LASTEXITCODE -ne 0 -or -not $SystemInfoContentFromWSL) {
    Write-Host "❌ Failed to read '$SystemInfoOutputWSL' from WSL. Content was empty or command failed."
    # You can add more diagnostics here, like checking if the file exists in WSL
    $FileExistsCheck = wsl -d $Distro --user $WSLUser -- bash -c "test -f '$SystemInfoOutputWSL' && echo 'exists' || echo 'not_exists'"
    Write-Host "   File '$SystemInfoOutputWSL' in WSL: $FileExistsCheck"
} else {
    try {
        # Ensure $SystemInfoContentFromWSL is treated as a single multi-line string if needed
        # For JSON, it should be fine.
        Set-Content -Path $SystemInfoOutputWindows -Value $SystemInfoContentFromWSL -Encoding UTF8
        if (Test-Path $SystemInfoOutputWindows) {
            # CORRECTED Write-Host line:
            Write-Host "✅ system_info_nvidia.json saved to '$ResultsDir'."
        } else {
            Write-Host "❌ Failed to save system_info_nvidia.json to Windows. File not found after Set-Content."
        }
    } catch {
        Write-Host "❌ Error saving system_info_nvidia.json to Windows: $($_.Exception.Message)"
    }
}

Write-Host "`n🎉 All done! Your WSL environment should be ready for benchmarking."