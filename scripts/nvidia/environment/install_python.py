import subprocess
import os
import urllib.request

def install_python_3119_windows():
    python_version = "3.11.9"
    installer_url = f"https://www.python.org/ftp/python/{python_version}/python-{python_version}-amd64.exe"
    installer_filename = f"python-{python_version}-installer.exe"
    installer_path = os.path.join(os.getcwd(), installer_filename)

    # Check if already installed
    try:
        output = subprocess.check_output(["python", "--version"], text=True).strip()
        if python_version in output:
            print(f"Python {python_version} is already installed.")
            return
        else:
            print(f"Existing Python version: {output} — installing {python_version} now.")
    except Exception:
        print("Python not found, installing...")

    # Download installer
    print(f"Downloading Python {python_version} installer...")
    urllib.request.urlretrieve(installer_url, installer_path)
    print(f"Downloaded installer to {installer_path}")

    # Run installer silently
    print("Running Python installer silently...")
    subprocess.run([
        installer_path,
        "/quiet",
        "InstallAllUsers=1",
        "PrependPath=1",
        f"TargetDir=C:\\Python{python_version.replace('.', '')}"
    ], check=True)

    print(f"Python {python_version} installed at C:\\Python{python_version.replace('.', '')}")

    # Cleanup
    if os.path.exists(installer_path):
        os.remove(installer_path)
        print("Installer cleaned up.")