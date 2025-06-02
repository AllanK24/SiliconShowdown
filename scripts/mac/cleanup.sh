#!/bin/zsh
set -e # Exit immediately if a command fails

echo "Benchmark Cleanup Script"
echo "-------------------------------------------"
echo "This script will attempt to:"
echo "1. Remove the 'smctemp' binary (typically from /usr/local/bin/smctemp)."
echo "2. Remove the temporary 'smctemp' build directory (checking both /private/tmp/smctemp and /tmp/smctemp) if it exists."
echo ""
echo "It will NOT uninstall Homebrew or Python."
echo ""
echo "‼️ YOU MAY BE PROMPTED FOR YOUR ADMINISTRATOR PASSWORD ‼️"
echo "   This is required to remove 'smctemp' if it was installed system-wide."
echo ""
read "USER_CONFIRMATION?Press Enter to continue with cleanup, or Ctrl+C to cancel. "

SMCTEMP_INSTALL_PATH="/usr/local/bin/smctemp" # The typical installation path for the binary
uninstalled_smctemp_binary=false

# Attempt to remove the installed binary directly
if [ -f "$SMCTEMP_INSTALL_PATH" ]; then
    echo "Found smctemp binary at $SMCTEMP_INSTALL_PATH."
    echo "Attempting to remove it with sudo..."
    if sudo rm -f "$SMCTEMP_INSTALL_PATH"; then
        echo "✅ Successfully removed $SMCTEMP_INSTALL_PATH."
        uninstalled_smctemp_binary=true
    else
        echo "❌ ERROR: Failed to remove $SMCTEMP_INSTALL_PATH even with sudo."
        echo "   You may need to remove it manually or check permissions."
    fi
else
    echo "smctemp binary not found at $SMCTEMP_INSTALL_PATH. Assuming it's not installed or already removed."
    uninstalled_smctemp_binary=true # Effectively, it's not there to be removed
fi

if [ "$uninstalled_smctemp_binary" = true ]; then
    echo "✅ smctemp binary appears to be uninstalled or was not found."
else
    # This case should ideally not be reached if sudo rm works or file doesn't exist.
    echo "⚠️ smctemp binary at $SMCTEMP_INSTALL_PATH might still be present if removal failed."
fi

# Determine and remove the actual smctemp build directory
SMCTEMP_BUILD_DIR_PRIMARY="/private/tmp/smctemp"
SMCTEMP_BUILD_DIR_FALLBACK="/tmp/smctemp"
ACTUAL_SMCTEMP_BUILD_DIR=""

if [ -d "$SMCTEMP_BUILD_DIR_PRIMARY" ]; then
    ACTUAL_SMCTEMP_BUILD_DIR="$SMCTEMP_BUILD_DIR_PRIMARY"
elif [ -d "$SMCTEMP_BUILD_DIR_FALLBACK" ]; then
    ACTUAL_SMCTEMP_BUILD_DIR="$SMCTEMP_BUILD_DIR_FALLBACK"
fi

if [ -n "$ACTUAL_SMCTEMP_BUILD_DIR" ]; then
    echo "Found temporary smctemp build directory: $ACTUAL_SMCTEMP_BUILD_DIR"
    echo "Removing it..."
    if rm -rf "$ACTUAL_SMCTEMP_BUILD_DIR"; then
        echo "✅ Temporary smctemp build directory removed."
    else
        echo "⚠️ Could not remove $ACTUAL_SMCTEMP_BUILD_DIR. It might require manual deletion or was already gone."
    fi
else
    echo "Temporary smctemp build directory was not found (checked $SMCTEMP_BUILD_DIR_PRIMARY and $SMCTEMP_BUILD_DIR_FALLBACK)."
fi

echo "-------------------------------------------"
echo "Cleanup script finished."
echo ""
echo "MANUAL STEP REMAINING:"
echo "Please drag the entire benchmark folder (e.g., 'mac' from your Desktop)"
echo "to the Trash to remove all other benchmark-specific files, scripts, and the"
echo "virtual environment."
echo ""
read "?Press Enter to close this Terminal window. "

exit 0