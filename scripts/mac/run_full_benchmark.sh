#!/bin/zsh
# Exit immediately if a command exits with a non-zero status.
set -e

echo "Benchmark Process Started..."
echo "This script will run both MPS and MLX benchmarks."
echo "-------------------------------------------"

# Get the directory where this script is located.
SCRIPT_DIR="$(cd "$(dirname "${ZSH_SOURCE[0]:-${(%):-%x}}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR" # Change to the 'mac' directory

echo "Working directory: $(pwd)"
echo "(This should be your '.../Desktop/mac' folder)"
echo "-------------------------------------------"

echo "IMPORTANT PREREQUISITES (Manual Steps - if not done already):"
echo "1. Open System Settings -> Privacy & Security -> Developer Tools -> Enable Terminal."
echo "2. Open System Settings -> Privacy & Security -> Full Disk Access -> Add and Enable Terminal."
echo "3. Ensure you have an active internet connection for initial downloads."
echo "Press ENTER to continue if you've completed/verified these, or Ctrl+C to quit."
read -r USER_CONFIRMATION_PREREQS
echo "-------------------------------------------"

INSTALL_SCRIPT_NAME="install_mac.sh"
COLLECT_INFO_SCRIPT="environment/collect_system_info.py" # Assuming you have this
MPS_BENCHMARK_SCRIPT="benchmark/benchmark_mps.py"
MLX_BENCHMARK_SCRIPT="benchmark/benchmark_mlx.py"

# Function to remove quarantine attribute
remove_quarantine() {
    local file_path="$1"
    if [ -f "$file_path" ]; then
        echo "Attempting to remove quarantine attribute from $file_path..."
        xattr -c "$file_path" || echo "Warning: xattr -c for $file_path failed (file might not have it), but continuing."
    else
        echo "Warning: $file_path not found for xattr removal check."
    fi
}

# Remove quarantine from the installer
remove_quarantine "$INSTALL_SCRIPT_NAME"
# Python scripts are run by the python interpreter, so xattr on them is less critical
# if the interpreter itself is trusted and the .app wrapper was allowed.
# However, no harm in being thorough if issues arise.
# remove_quarantine "$COLLECT_INFO_SCRIPT"
# remove_quarantine "$MPS_BENCHMARK_SCRIPT"
# remove_quarantine "$MLX_BENCHMARK_SCRIPT"
echo "-------------------------------------------"

# Step 4 from original list: Make the installer executable
echo "Making installer ($INSTALL_SCRIPT_NAME) executable..."
if [ ! -f "$INSTALL_SCRIPT_NAME" ]; then
    echo "ERROR: $INSTALL_SCRIPT_NAME not found in $(pwd)"
    echo "Please ensure all files from the ZIP are correctly extracted here."
    read "?Press Enter to exit."
    exit 1
fi
chmod +x "$INSTALL_SCRIPT_NAME"
echo "Done."
echo "-------------------------------------------"

# Step 5: Execute the installer script
echo "Executing installer script (./$INSTALL_SCRIPT_NAME)..."
echo "This script will attempt to install/update Homebrew, Python, smctemp, and Python packages."
echo ""
echo "‼️ YOU WILL LIKELY BE PROMPTED FOR YOUR ADMINISTRATOR PASSWORD ‼️"
echo "   This is required for Homebrew (first time) and 'smctemp' installation."
echo "   Please enter it when prompted."
echo ""
./"$INSTALL_SCRIPT_NAME"
# set -e will cause script to exit if install_mac.sh fails
echo "Installer script finished."
echo "-------------------------------------------"

# Step 6: Activate the virtual environment
VENV_ACTIVATE_PATH="environment/ai_benchmark_env/bin/activate"
echo "Activating virtual environment: $VENV_ACTIVATE_PATH"
if [ ! -f "$VENV_ACTIVATE_PATH" ]; then
    echo "ERROR: Virtual environment activation script not found at $VENV_ACTIVATE_PATH"
    echo "This script should have been created by $INSTALL_SCRIPT_NAME."
    echo "Please check the output of the installer script for errors."
    read "?Press Enter to exit."
    exit 1
fi
source "$VENV_ACTIVATE_PATH"
echo "Virtual environment activated."
echo "Python in use: $(which python)"
echo "Python version: $(python --version)"
echo "-------------------------------------------"

# Step 7: Collect system information (if you have this script)
if [ -f "$COLLECT_INFO_SCRIPT" ]; then
    echo "Collecting system information (python $COLLECT_INFO_SCRIPT)..."
    python "$COLLECT_INFO_SCRIPT" # Assuming it saves to system_info.json in 'mac' or 'mac/benchmark/results'
    echo "System information collection finished."
    echo "-------------------------------------------"
else
    echo "Warning: System info script ($COLLECT_INFO_SCRIPT) not found. Skipping."
    echo "Please ensure 'system_info.json' is generated if required."
    echo "-------------------------------------------"
fi

# Run the MPS benchmark
echo "Running the PyTorch MPS benchmark (python $MPS_BENCHMARK_SCRIPT)..."
echo "This process can take a while to complete."
if [ ! -f "$MPS_BENCHMARK_SCRIPT" ]; then
    echo "ERROR: MPS Benchmark script not found at $MPS_BENCHMARK_SCRIPT. Skipping."
else
    # Assuming benchmark_mps.py handles its own output file naming with timestamp
    python "$MPS_BENCHMARK_SCRIPT"
fi
echo "PyTorch MPS benchmark run complete (or skipped)."
echo "-------------------------------------------"

# Run the MLX benchmark
echo "Running the MLX benchmark (python $MLX_BENCHMARK_SCRIPT)..."
echo "This process can also take a while, especially the first time (model download)."
if [ ! -f "$MLX_BENCHMARK_SCRIPT" ]; then
    echo "ERROR: MLX Benchmark script not found at $MLX_BENCHMARK_SCRIPT. Skipping."
else
    # Assuming benchmark_mlx.py handles its own output file naming with timestamp
    python "$MLX_BENCHMARK_SCRIPT"
fi
echo "MLX benchmark run complete (or skipped)."
echo "-------------------------------------------"

# Deactivation of venv is optional here as the script will end,
# and the Terminal session launched by AppleScript will close.
# If you had subsequent commands *outside* the venv, you'd use 'deactivate'.

echo ""
echo "ALL BENCHMARK PROCESSES ATTEMPTED."
echo "-------------------------------------------"
echo "Please find your results. They are typically located in:"
echo "  $SCRIPT_DIR/benchmark/results/"
echo "Look for files like:"
echo "  - system_info.json (if generated by a separate script)"
echo "  - benchmark_results_mps_YYYYMMDD-HHMMSS.json"
echo "  - benchmark_results_mlx_YYYYMMDD-HHMMSS.json"
echo ""
echo "Please verify the files exist and then send them to us."
echo "-------------------------------------------"
echo ""
# Keep the terminal window open until the user presses Enter
read "?Press Enter to close this Terminal window."

exit 0