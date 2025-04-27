import subprocess
import sys
import ctypes

# --- Configuration ---
# IMPORTANT: The Package ID for CUDA might change.
# You can verify the current ID by opening PowerShell or Command Prompt and running:
# winget search CUDA
# Then update the CUDA_PACKAGE_ID below if necessary.
CUDA_PACKAGE_ID = "Nvidia.CUDA" # Example ID, verify with 'winget search'

# --- Helper Function to Check/Request Admin Rights ---
def run_as_admin():
    """ Checks for admin rights and relaunches the script if needed. """
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception as e:
        print(f"Error checking admin status: {e}")
        is_admin = False # Assume not admin if check fails

    if not is_admin:
        print("Administrator privileges required. Attempting to relaunch...")
        try:
            # Relaunch the script with admin rights
            result = ctypes.windll.shell32.ShellExecuteW(
                None,           # Hwnd
                "runas",        # Operation: run as admin
                sys.executable, # Path to Python executable
                " ".join(sys.argv), # Script path and arguments
                None,           # Working directory
                1               # Show window normally
            )
            if result <= 32: # Error codes are <= 32
                 print(f"Failed to relaunch as admin (Error code: {result}). Please run the script manually as Administrator.")
                 sys.exit(1)
            else:
                 print("Relaunch successful. Exiting current non-admin instance.")
                 sys.exit(0) # Exit the original non-admin process

        except Exception as e:
            print(f"Error trying to relaunch as admin: {e}")
            print("Please run the script manually as Administrator.")
            sys.exit(1)
    else:
        print("Running with Administrator privileges.")

# --- Main Function to Install CUDA ---
def install_cuda_with_winget(package_id: str):
    """
    Uses winget to install the specified package ID.
    Reboots system automatically after successful installation.
    """
    print(f"\nAttempting to install CUDA Toolkit (Package ID: {package_id}) using winget...")
    
    
    # Construct the winget command
    # Flags used:
    # --id : Specifies the package ID
    # -s / --silent : Attempts a silent installation (behavior depends on the installer)
    # --accept-package-agreements : Automatically accept license agreements if required
    # --accept-source-agreements : Automatically accept source repository agreements
    # --force: Can be useful if winget thinks it's already installed or to force reinstall/upgrade
    command = [
        "winget", "install",
        "--id", package_id,
        "--silent",  # fixed flag (not "-s", it should be "--silent")
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
    ]

    print(f"Executing command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, shell=True)

        print("\n--- Winget Output ---")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("--- Winget Errors/Warnings ---")
            print(result.stderr)
        print("--- End Winget Output ---")

        if result.returncode == 0:
            print(f"\nWinget command executed successfully.")
            print("CUDA Toolkit installation likely started or completed.")
            print("System will reboot in 10 seconds to finalize installation...")

            # Reboot system
            subprocess.run(["shutdown", "/r", "/t", "10"], shell=True)

        else:
            print(f"\nWinget command failed with return code: {result.returncode}")
            if "0x80070005" in (result.stderr or ""):
                print("Error suggests permission issues. Ensure you are running as Administrator.")
            elif "No applicable installer found" in (result.stderr or ""):
                print(f"Error suggests the package ID '{package_id}' might be incorrect or unavailable.")
                print("Please run 'winget search CUDA' to find the correct ID.")
            else:
                print("Review the Winget output above for specific error details.")

    except FileNotFoundError:
        print("\nError: 'winget' command not found.")
        print("Please ensure winget is installed and in your system's PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred while running winget: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_as_admin()  # <<< Make sure we are admin FIRST
    install_cuda_with_winget(CUDA_PACKAGE_ID)