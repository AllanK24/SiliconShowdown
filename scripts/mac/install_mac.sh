set -e  # Exit immediately if a command fails

# Ensure Homebrew’s path is in PATH
if [ -x "/opt/homebrew/bin/brew" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
elif [ -x "/usr/local/bin/brew" ]; then
    export PATH="/usr/local/bin:$PATH"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "🍺 Checking for Homebrew..."

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
export PATH="/opt/homebrew/bin:$PATH"
brew install python@3.11

echo "✅ Python 3.11 installed."

echo ""
echo "🔧 Installing smctemp via git clone..."
git clone https://github.com/narugit/smctemp.git /tmp/smctemp
cd /tmp/smctemp
make
sudo make install
echo "✅ smctemp installed."
cd "$SCRIPT_DIR"

# Force shell to use Homebrew’s Python 3.11
export PATH="$(brew --prefix)/opt/python@3.11/bin:$PATH"

#
# After basic Python is guaranteed:
echo ""
echo "Running Python install script..."
"$(brew --prefix)/opt/python@3.11/bin/python3.11" "$SCRIPT_DIR/environment/install_python.py"

# Then create the virtual environment
echo ""
echo "🛠️ Creating Python virtual environment..."
"$(brew --prefix)/opt/python@3.11/bin/python3.11" "$SCRIPT_DIR/environment/create_venv.py"

# user will still need to manually "source" unless extra scripting is added

echo ""
echo "Setup complete! You are ready to benchmark."
echo ""
echo "✅ Setup complete!"
echo "To start working inside your environment, run:"
echo "          source environment/ai_benchmark_env/bin/activate"
echo ""
echo "Then you can run your benchmark scripts normally."