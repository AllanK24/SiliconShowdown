import sys
from scripts.nvidia.environment.collect_system_info import collect_system_info_nvidia
from scripts.nvidia.environment.install_python import install_python_3119_windows
from scripts.nvidia.environment.create_venv import create_and_activate_virtualenv, install_requirements

def main():
    try:
        install_python_3119_windows()
        venv_python, venv_pip = create_and_activate_virtualenv()
        install_requirements(venv_pip, "requirements.txt")
        collect_system_info_nvidia(venv_python)
        print("\n✅ Setup completed successfully! You are ready to run benchmarks.")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
