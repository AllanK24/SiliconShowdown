#!/bin/bash

# install_mac.sh
# Seamless automatic setup script for Mac

set -e  # Exit immediately if a command fails

echo ""
echo "Checking for python3..."

# Check if python3 exists
if ! command -v python3 &> /dev/null
then
    echo "❌ python3 not found."
    echo ""
    echo "🍺 Checking for Homebrew..."

    # Check if brew exists
    if ! command -v brew &> /dev/null
    then
        echo "❌ Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        echo "✅ Homebrew installation complete."
    else
        echo "✅ Homebrew found."
    fi

    echo ""
    echo "Installing Python 3.11 via Homebrew..."
    brew install python@3.11
    echo "✅ Python 3.11 installed."
else
    echo "✅ python3 found: $(python3 --version)"
fi

#
# After basic Python is guaranteed:
echo ""
echo "Running Python install script..."
python3 environment/install_python.py

# Then create the virtual environment
echo ""
echo "🛠️ Creating Python virtual environment..."
python3 environment/create_venv.py

# Collect system information
echo ""
echo "Collecting system information..."
python3 environment/collect_system_info.py

# user will still need to manually "source" unless extra scripting is added

echo ""
echo "Setup complete! You are ready to benchmark."
echo ""
echo "✅ Setup complete!"
echo "To start working inside your environment, run:"
echo "          source ai_benchmark_env/bin/activate"
echo ""
echo "Then you can run your benchmark scripts normally."