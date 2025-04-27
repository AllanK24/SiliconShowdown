from scripts.nvidia.environment.collect_system_info import collect_system_info
from scripts.nvidia.environment.install_python import install_python_3119_windows
from scripts.nvidia.environment.create_venv import create_and_activate_virtualenv, install_requirements

def main():
    install_python_3119_windows()
    venv_python, venv_pip = create_and_activate_virtualenv()
    install_requirements(venv_pip, "requirements.txt")
    collect_system_info(venv_python, "system_info.json")
    
if __name__ == "__main__":
    main()