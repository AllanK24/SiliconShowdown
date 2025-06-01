import subprocess


def install_python_3119_mac():
    python_version = "3.11.12"

    # Check if already installed
    try:
        output = subprocess.check_output(["python3", "--version"], text=True).strip()
        if python_version in output:
            print(f"Python {python_version} is already installed.")
            return
        else:
            print(f"Existing Python version: {output} — installing {python_version} now.")
    except Exception:
        print("Python not found, installing...")

    # Check if Homebrew is installed
    try:
        subprocess.check_output(["brew", "--version"], text=True)
    except Exception:
        print("Homebrew is not installed. Installing Homebrew...")
        subprocess.run(
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
            shell=True,
            check=True
        )
        print("Homebrew installation complete.")

    # Install Python 3.11.0 using Homebrew
    print(f"Installing Python {python_version} via Homebrew...")
    subprocess.run(["brew", "install", "python@3.11"], check=True)

    print(f"Python {python_version} installation via Homebrew complete.")
if __name__ == "__main__":
    install_python_3119_mac()