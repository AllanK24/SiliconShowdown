import subprocess
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))

def create_virtual_environment(env_name="ai_benchmark_env"):
    venv_path = os.path.join(script_dir, env_name)
    venv_python = os.path.join(venv_path, "bin", "python3")
    venv_pip = os.path.join(venv_path, "bin", "pip3")

    # Create venv if it doesn't exist
    if not os.path.exists(venv_path):
        print(f"🛠️ Creating virtual environment '{env_name}'...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print(f"✅ Virtual environment '{env_name}' created successfully.")
    else:
        print(f"✅ Virtual environment '{env_name}' already exists.")

    # Install required packages if requirements.txt exists
    requirements_file = os.path.join(script_dir, "requirements.txt")
    if os.path.exists(requirements_file):
        print(f"📦 Installing dependencies from {requirements_file}...")
        subprocess.run([venv_pip, "install", "-r", requirements_file], check=True)
        print("✅ Dependencies installed successfully.")
    else:
        print(f"⚠️ Requirements file {requirements_file} not found. Skipping package installation.")

    print("\n🎉 Virtual environment is ready to use!")

if __name__ == "__main__":
    create_virtual_environment()