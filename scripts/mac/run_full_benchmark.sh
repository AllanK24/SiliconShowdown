#!/bin/zsh
set -e

echo "Benchmark Process Started..."
echo "-------------------------------------------"
SCRIPT_DIR="$( cd "$( dirname "${ZSH_SOURCE[0]:-${(%):-%x}}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"
echo "-------------------------------------------"

echo "IMPORTANT PREREQUISITE (Manual Step):"
echo "1. Open System Settings -> Privacy & Security -> Developer Tools -> Enable Terminal."
echo "2. Ensure you have an active internet connection for Homebrew and component downloads."
echo "Press ENTER to continue if you've completed these, or Ctrl+C to quit."
read -r USER_CONFIRMATION_DEV_TOOLS
echo "-------------------------------------------"

INSTALL_SCRIPT_NAME="install_mac.sh"
PYTHON_INSTALL_HELPER="environment/install_python.py"
PYTHON_VENV_HELPER="environment/create_venv.py"
COLLECT_INFO_SCRIPT="environment/collect_system_info.py"
BENCHMARK_SCRIPT="benchmark/benchmark.py"

# Function to remove quarantine attribute
remove_quarantine() {
    local file_path="$1"
    if [ -f "$file_path" ]; then
        echo "Attempting to remove quarantine attribute from $file_path..."
        xattr -c "$file_path" || echo "Warning: xattr -c for $file_path failed, but continuing."
    else
        echo "Warning: $file_path not found for xattr removal."
    fi
}

# Remove quarantine from all key scripts
remove_quarantine "$INSTALL_SCRIPT_NAME"
remove_quarantine "$PYTHON_INSTALL_HELPER"
remove_quarantine "$PYTHON_VENV_HELPER"
remove_quarantine "$COLLECT_INFO_SCRIPT"
remove_quarantine "$BENCHMARK_SCRIPT"
# If smctemp involves other scripts that are *part of your ZIP* and get executed, add them too.
# However, smctemp is cloned, so its own scripts won't be quarantined by your ZIP.
echo "-------------------------------------------"


echo "Making installer ($INSTALL_SCRIPT_NAME) executable..."
if [ ! -f "$INSTALL_SCRIPT_NAME" ]; then
    echo "ERROR: $INSTALL_SCRIPT_NAME not found in $(pwd)"
    read -p "Press Enter to exit."
    exit 1
fi
chmod +x "$INSTALL_SCRIPT_NAME"
echo "Done."
echo "-------------------------------------------"

echo "Executing installer script (./$INSTALL_SCRIPT_NAME)..."
echo "This script will attempt to:"
echo "  1. Install/update Homebrew (if not found)."
echo "  2. Install Python 3.11 using Homebrew."
echo "  3. Compile and install 'smctemp' (temperature sensor utility)."
echo ""
echo "‼️ YOU WILL LIKELY BE PROMPTED FOR YOUR ADMINISTRATOR PASSWORD ‼️"
echo "   This is required for Homebrew (first time) and 'smctemp' installation."
echo "   Please enter it when prompted."
echo ""
./"$INSTALL_SCRIPT_NAME"
echo "Installer script finished."
echo "-------------------------------------------"

VENV_ACTIVATE_PATH="environment/ai_benchmark_env/bin/activate"
echo "Activating virtual environment: $VENV_ACTIVATE_PATH"
if [ ! -f "$VENV_ACTIVATE_PATH" ]; then
    echo "ERROR: Virtual environment activation script not found at $VENV_ACTIVATE_PATH"
    echo "This script should have been created by $INSTALL_SCRIPT_NAME."
    read -p "Press Enter to exit."
    exit 1
fi
source "$VENV_ACTIVATE_PATH"
echo "Virtual environment activated."
echo "Python in use: $(which python)"
echo "Python version: $(python --version)"
echo "-------------------------------------------"

echo "Collecting system information (python $COLLECT_INFO_SCRIPT)..."
if [ ! -f "$COLLECT_INFO_SCRIPT" ]; then
    echo "ERROR: System info script not found at $COLLECT_INFO_SCRIPT"
    read -p "Press Enter to exit."
    exit 1
fi
python "$COLLECT_INFO_SCRIPT"
echo "System information collection finished."
echo "-------------------------------------------"

echo "Running the benchmark (python $BENCHMARK_SCRIPT)..."
echo "This process can take a while to complete."
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "ERROR: Benchmark script not found at $BENCHMARK_SCRIPT"
    read -p "Press Enter to exit."
    exit 1
fi
python "$BENCHMARK_SCRIPT"
echo "-------------------------------------------"
echo "Benchmark run complete…"
echo ""
echo "PROCESS FINISHED!"
echo "-------------------------------------------"
echo "Please find your results in this folder ($SCRIPT_DIR):"
echo "  - system_info.json"
echo "  - benchmark_result_mps.json (or similar)"
echo ""
echo "Please send these files to us."
echo "-------------------------------------------"
echo ""
read -p "Press Enter to close this Terminal window."

exit 0