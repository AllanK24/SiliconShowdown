#!/bin/bash
set -e

echo ""
echo "🍺 Checking for Homebrew..."

if ! command -v brew &> /dev/null
then
    echo "❌ Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo "✅ Homebrew installation complete."
else
    echo "✅ Homebrew found."
    echo ""
fi

echo "🐍 Checking for python3..."

if ! command -v python3 &> /dev/null
then
    echo "❌ python3 not found. Installing Python 3.11.9 via Homebrew..."
    brew install python@3.11
    echo "✅ Python 3.11.9 installed."
    echo ""
else
    python_version=$(python3 --version | awk '{print $2}')
    echo "✅ python3 found: $python_version"

    if [[ "$python_version" != "3.11.9" ]]; then
        echo "⚠️  python3 version is not 3.11.9. Installing correct version..."
        brew install python@3.11
        echo "✅ Python 3.11.9 installed."
        echo ""
    else
        echo "✅ python3 version is already 3.11.9."
        echo ""
    fi
fi

echo ""
echo "🌡️ Installing osx-cpu-temp for temperature measurements..."
brew install osx-cpu-temp
echo "✅ osx-cpu-temp installed."
echo ""

# Now that brew, python, and osx-cpu-temp are installed, continue:

echo "🛠️ Creating Python virtual environment (ai_benchmark_env)..."
python3 environment/create_venv.py

echo ""
echo "📋 Collecting system and hardware information..."
python3 environment/collect_system_info.py

echo ""
echo "✅ Setup complete!"
echo "To start working inside your environment, run:"
echo "          source ai_benchmark_env/bin/activate"
echo ""
echo "Then you can run your benchmark scripts normally."