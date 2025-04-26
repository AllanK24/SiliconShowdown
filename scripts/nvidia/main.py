from scripts.nvidia.environment.install_python import install_python_3119_windows
from scripts.nvidia.environment.create_venv import create_and_activate_virtualenv, install_requirements

def main():
    install_python_3119_windows()
    venv_pip = create_and_activate_virtualenv()
    install_requirements(venv_pip, "requirements.txt")
    
if __name__ == "__main__":
    main()