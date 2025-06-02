#!/bin/zsh
# Or #!/bin/bash
set -e # Exit immediately if a command fails

echo "Benchmark Cleanup Script"
echo "-------------------------------------------"
echo "This script will attempt to:"
echo "1. Uninstall 'smctemp' (if found and installed from /tmp/smctemp)."
echo "2. Remove the temporary 'smctemp' build directory (/tmp/smctemp)."
echo ""
echo "It will NOT uninstall Homebrew or Python."
echo ""
echo "‼️ YOU MAY BE PROMPTED FOR YOUR ADMINISTRATOR PASSWORD ‼️"
echo "   This is required to uninstall 'smctemp'."
echo ""
read -p "Press Enter to continue with cleanup, or Ctrl+C to cancel." USER_CONFIRMATION

SMCTEMP_BUILD_DIR="/tmp/smctemp"

if [ -d "$SMCTEMP_BUILD_DIR" ]; then
    echo "Found smctemp build directory: $SMCTEMP_BUILD_DIR"
    cd "$SMCTEMP_BUILD_DIR"
    if [ -f "Makefile" ]; then
        echo "Attempting to run 'sudo make uninstall' for smctemp..."
        sudo make uninstall # This requires the Makefile to have an uninstall target
        echo "✅ 'sudo make uninstall' command issued."
    else
        echo "⚠️ Makefile not found in $SMCTEMP_BUILD_DIR. Cannot run 'make uninstall'."
        echo "   If smctemp was installed, you might need to remove it manually."
        echo "   Commonly installed to /usr/local/bin/smctemp or similar."
    fi
    echo "Removing temporary smctemp build directory: $SMCTEMP_BUILD_DIR..."
    cd "$HOME" # Go to a safe directory before removing
    rm -rf "$SMCTEMP_BUILD_DIR"
    echo "✅ Temporary smctemp build directory removed."
else
    echo "Temporary smctemp build directory ($SMCTEMP_BUILD_DIR) not found."
    echo "If smctemp was installed from a different session, you might need to remove it manually."
    echo "Commonly installed to /usr/local/bin/smctemp or similar."
fi

echo "-------------------------------------------"
echo "Cleanup script finished."
echo ""
echo "MANUAL STEP REMAINING:"
echo "Please drag the entire benchmark folder (e.g., 'mac' from your Desktop)"
echo "to the Trash to remove all other benchmark-specific files, scripts, and the"
echo "virtual environment."
echo ""
read -p "Press Enter to close this Terminal window."

exit 0