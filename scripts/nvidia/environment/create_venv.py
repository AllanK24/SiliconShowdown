import subprocess
import os
import sys

def create_and_activate_virtualenv(env_name="ai_benchmark_env"):
    python_exe = "C:\\Python3119\\python.exe"  # Installed Python 3.11.9
    venv_path = os.path.join(os.getcwd(), env_name)
    venv_python = os.path.join(venv_path, "Scripts", "python.exe")
    venv_pip = os.path.join(venv_path, "Scripts", "pip.exe")

    if not os.path.exists(python_exe):
        print("Python 3.11.9 executable not found. Please install Python first.")
        sys.exit(1)

    # Create venv if it doesn't exist
    if not os.path.exists(venv_path):
        print(f"Creating virtual environment '{env_name}'...")
        subprocess.run([python_exe, "-m", "venv", venv_path], check=True)
        print(f"Virtual environment '{env_name}' created successfully!")
    else:
        print(f"Virtual environment '{env_name}' already exists.")

    # Confirm venv Python is there
    if not os.path.exists(venv_python):
        print("Something went wrong creating the virtual environment.")
        sys.exit(1)
    
    print(f"Using venv Python: {venv_python}")
    return venv_python, venv_pip

def install_requirements(venv_pip, requirements_file="requirements.txt"):
    # Install requirements
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.run([venv_pip, "install", "-r", requirements_file], check=True)
        print("Dependencies installed successfully.")
    else:
        print(f"Requirements file {requirements_file} not found. Skipping package installation.")

    print("\nVirtual environment is ready!")