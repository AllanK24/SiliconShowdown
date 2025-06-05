# install_wsl.ps1
#Requires -RunAsAdministrator # Ensures the script requests admin rights

# --- Configuration ---
$DesiredDistroName = "Ubuntu"
$BenchmarkUserName = "benchmark"
$BenchmarkUserPassword = "benchmark" # Be cautious with hardcoded passwords

# --- Script Start ---
Write-Host "---------------------------------------------------------------------"
Write-Host " WSL & $DesiredDistroName Installation/Setup Script "
Write-Host "---------------------------------------------------------------------"
Write-Host "INFO: This script will check for and install/update WSL and $DesiredDistroName."
Write-Host "INFO: It will attempt to create a user '$BenchmarkUserName'."
Write-Host "IMPORTANT: A system restart may be required and initiated by this script."
Write-Host "Please save all your work before proceeding."
$confirmation = Read-Host "Do you want to continue? (Y/N)"
if ($confirmation -ne 'Y' -and $confirmation -ne 'y') {
    Write-Host "Exiting script."
    exit 0
}

# --- Helper Function to Check if a Feature is Enabled ---
function Test-WindowsFeatureEnabled {
    param(
        [string]$FeatureName
    )
    $feature = Get-WindowsOptionalFeature -Online -FeatureName $FeatureName
    return $feature.State -eq 'Enabled'
}

# --- 1. Ensure Core WSL Features are Enabled ---
Write-Host "`n--- Checking Core Windows Features for WSL ---"
$wslFeature = "Microsoft-Windows-Subsystem-Linux"
$vmPlatformFeature = "VirtualMachinePlatform"

if (-not (Test-WindowsFeatureEnabled -FeatureName $wslFeature)) {
    Write-Host "INFO: Enabling '$wslFeature'..."
    try {
        Enable-WindowsOptionalFeature -Online -FeatureName $wslFeature -NoRestart -ErrorAction Stop
        Write-Host "✅ '$wslFeature' enabled. A restart will be needed later."
    } catch {
        Write-Error "❌ FAILED to enable '$wslFeature': $($_.Exception.Message)"
        Read-Host "Press Enter to exit."
        exit 1
    }
} else {
    Write-Host "✅ '$wslFeature' is already enabled."
}

if (-not (Test-WindowsFeatureEnabled -FeatureName $vmPlatformFeature)) {
    Write-Host "INFO: Enabling '$vmPlatformFeature'..."
    try {
        Enable-WindowsOptionalFeature -Online -FeatureName $vmPlatformFeature -NoRestart -ErrorAction Stop
        Write-Host "✅ '$vmPlatformFeature' enabled. A restart will be needed later."
    } catch {
        Write-Error "❌ FAILED to enable '$vmPlatformFeature': $($_.Exception.Message)"
        Read-Host "Press Enter to exit."
        exit 1
    }
} else {
    Write-Host "✅ '$vmPlatformFeature' is already enabled."
}

# --- 2. Install/Update WSL and the Desired Distribution ---
Write-Host "`n--- Checking/Installing WSL Distribution: $DesiredDistroName ---"
$installedDistros = try { wsl --list --quiet --verbose } catch { @() } # Get installed distros, handle error if wsl command not found yet
$distroInstanceFound = $false
foreach ($line in ($installedDistros | Out-String -Stream)) {
    if ($line -match "\*?\s*$([regex]::Escape($DesiredDistroName))\s+") { # Check for distro name, ignoring default asterisk
        $distroInstanceFound = $true
        break
    }
}

$needsRestartForFeatures = (Get-WindowsOptionalFeature -Online -FeatureName $wslFeature).RestartNeeded -or (Get-WindowsOptionalFeature -Online -FeatureName $vmPlatformFeature).RestartNeeded

if ($distroInstanceFound) {
    Write-Host "✅ WSL distribution '$DesiredDistroName' is already installed."
    Write-Host "INFO: Checking for WSL kernel updates..."
    try {
        wsl --update --no-launch # --no-launch prevents distro from starting if not needed
        Write-Host "✅ WSL kernel update check complete."
    } catch {
        Write-Warning "⚠️  Could not run 'wsl --update'. Error: $($_.Exception.Message)"
    }
    # Updating packages within the distro can be done in the setup_dev_env script
    # to ensure it runs after any potential restart.
} else {
    Write-Host "INFO: '$DesiredDistroName' not found. Attempting to install..."
    if ($needsRestartForFeatures) {
        Write-Warning "⚠️  A system restart is required to enable core WSL features before the distribution can be fully installed."
        Write-Host "The system will restart. After restart, please re-run this script to continue with '$DesiredDistroName' installation and user setup."
        Read-Host "Press Enter to restart the computer."
        Restart-Computer -Force
        exit 0 # Exit so user can re-run after restart
    }
    try {
        Write-Host "Executing: wsl --install -d $DesiredDistroName --no-launch"
        # --no-launch prevents the distro from opening its console immediately, allowing script to continue.
        wsl --install -d $DesiredDistroName --no-launch
        $LASTEXITCODE = 0 # Assume success if no exception
    } catch {
        $LASTEXITCODE = 1 # Indicate failure
        Write-Error "❌ FAILED during 'wsl --install -d $DesiredDistroName': $($_.Exception.Message)"
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ WSL and '$DesiredDistroName' installation command initiated."
        Write-Host "INFO: The distribution will complete its setup on first proper launch or if a restart is needed."
        # Check if a restart is now flagged by 'wsl --install' itself (it might not update the feature status immediately)
        # This is harder to detect without parsing `wsl --status` which is also tricky.
        # Often `wsl --install` handles the restart prompt if needed.
        $needsRestartForFeatures = $true # Assume restart is good after install command
    } else {
        Write-Error "❌ '$DesiredDistroName' installation failed. Please check errors above."
        Write-Host "You might need to install '$DesiredDistroName' from the Microsoft Store manually or check WSL documentation."
        Read-Host "Press Enter to exit."
        exit 1
    }
}

# --- 3. Set WSL2 as Default (Recommended) ---
Write-Host "`n--- Setting WSL default version to 2 (if applicable) ---"
try {
    wsl --set-default-version 2
    Write-Host "✅ WSL default version set to 2."
} catch {
    Write-Warning "⚠️  Could not set WSL default version to 2. Error: $($_.Exception.Message). This might be fine if WSL1 is not installed or if already default."
}

# --- 4. Verify Distro and Create Benchmark User ---
# It's crucial the distro is fully initialized before we try to add a user.
# `wsl --install` with `--no-launch` might mean the distro hasn't had its first-run setup.
# A robust way is to try running a simple command. If it fails, initial setup might still be pending.

Write-Host "`n--- Verifying '$DesiredDistroName' and Creating User '$BenchmarkUserName' ---"
Start-Sleep -Seconds 5 # Give a moment for WSL state to settle

function Test-WslDistroReady {
    param([string]$DistroName)
    try {
        wsl -d $DistroName --exec whoami | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-WslDistroReady -DistroName $DesiredDistroName)) {
    Write-Warning "⚠️ '$DesiredDistroName' does not seem to be fully ready or operational yet."
    Write-Host "This can happen if it's a fresh install and requires its initial setup (which normally prompts for a default user)."
    Write-Host "If you see a '$DesiredDistroName' console window open asking for a new UNIX username, please complete that setup first."
    Write-Host "Alternatively, if a restart was indicated earlier for WSL features, that restart is essential."
    if ($needsRestartForFeatures) {
         Write-Host "A restart is pending. The script will now initiate it. Please re-run after restart."
         Read-Host "Press Enter to restart."
         Restart-Computer -Force
         exit 0
    }
    Read-Host "Press Enter to try user creation (may fail if distro not set up), or Ctrl+C to exit and set up '$DesiredDistroName' manually."
}

Write-Host "Attempting to create user '$BenchmarkUserName' in '$DesiredDistroName'..."
# Commands to create user if not exists and set password.
# These must run as root in the WSL distro.
$userCreateCommand = @"
if id '$BenchmarkUserName' &>/dev/null; then
    echo "User '$BenchmarkUserName' already exists."
else
    useradd -m -s /bin/bash '$BenchmarkUserName' && echo "User '$BenchmarkUserName' created." || echo "Failed to create user '$BenchmarkUserName'."
fi
echo '$BenchmarkUserName:$BenchmarkUserPassword' | chpasswd && echo "Password set for '$BenchmarkUserName'." || echo "Failed to set password for '$BenchmarkUserName'."
usermod -aG sudo '$BenchmarkUserName' && echo "User '$BenchmarkUserName' added to sudo group." || echo "Failed to add user to sudo group."
"@

try {
    # Execute the user creation commands as root in the WSL distro
    wsl -d $DesiredDistroName -u root -- bash -c $userCreateCommand
    Write-Host "✅ User creation/password commands sent to '$DesiredDistroName'."
    # Verify user by trying to run a command as that user
    wsl -d $DesiredDistroName -u $BenchmarkUserName --exec whoami
    Write-Host "✅ Successfully executed a command as user '$BenchmarkUserName'."
} catch {
    Write-Error "❌ FAILED to execute user creation commands or verify user in '$DesiredDistroName': $($_.Exception.Message)"
    Write-Host "Please ensure '$DesiredDistroName' is running and accessible."
    Write-Host "You may need to manually create the user '$BenchmarkUserName' with password '$BenchmarkUserPassword' and add them to the sudo group."
    # Don't exit here, let the restart proceed if needed for features.
}


# --- 5. Final Status and Restart ---
Write-Host "`n--- Final WSL Status ---"
try {
    wsl --status
} catch {
    Write-Warning "⚠️  Could not get 'wsl --status'."
}

if ($needsRestartForFeatures) {
    Write-Host "`nINFO: A system restart is required to finalize WSL feature setup or installation."
    Write-Host "The system will restart in 20 seconds."
    Write-Host "After restarting, please proceed with the 'setup_dev_env_wsl.ps1' script."
    Start-Sleep -Seconds 20
    Restart-Computer -Force
} else {
    Write-Host "`nINFO: WSL and '$DesiredDistroName' setup appears complete (or was already setup)."
    Write-Host "If any errors occurred, please review them. No immediate restart initiated by this script for feature enablement."
    Write-Host "You can now proceed with 'setup_dev_env_wsl.ps1'."
}

Write-Host "--- Script Finished ---"
Read-Host "Press Enter to exit if no restart was initiated."