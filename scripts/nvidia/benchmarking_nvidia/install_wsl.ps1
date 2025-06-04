# install_wsl.ps1

Write-Host "Please make sure you close all the important applications, as this script will restart your computer."
Write-Host "Starting WSL installation and setup..."

# Define the desired distribution name
$DesiredDistroName = "Ubuntu"

# Check if WSL feature is enabled and what version is default
Write-Host "Checking WSL feature status..."
dism.exe /online /get-featureinfo /featurename:Microsoft-Windows-Subsystem-Linux | Out-String
dism.exe /online /get-featureinfo /featurename:VirtualMachinePlatform | Out-String

# Check if the desired distribution is already installed
$installedDistros = wsl --list --quiet
$distroFound = $false
foreach ($distro in $installedDistros) {
    if ($distro -eq $DesiredDistroName) {
        $distroFound = $true
        break
    }
}

if ($distroFound) {
    Write-Host "✅ WSL distribution '$DesiredDistroName' is already installed."
    Write-Host "Checking for WSL updates..."
    wsl --update
    # Optionally update the specific distro too
    # Write-Host "Updating '$DesiredDistroName' packages..."
    # wsl -d $DesiredDistroName -- bash -c "sudo apt update && sudo apt upgrade -y"
} else {
    Write-Host "❌ WSL distribution '$DesiredDistroName' not found."
    Write-Host "Attempting to install WSL and '$DesiredDistroName'..."

    # Attempt to install WSL (enables the feature if not already) and the specific distribution
    # The -d flag with --install explicitly requests a distribution.
    # If WSL is already enabled, this will primarily focus on installing the distro.
    try {
        Write-Host "Running: wsl --install -d $DesiredDistroName"
        wsl --install -d $DesiredDistroName
        Write-Host "✅ WSL and '$DesiredDistroName' installation command initiated."
        Write-Host "The distribution will be set up on first launch if needed."
    } catch {
        Write-Host "Error during 'wsl --install -d $DesiredDistroName': $($_.Exception.Message)"
        Write-Host "Please ensure your Windows version supports this command and you have internet access."
        Write-Host "You might need to install '$DesiredDistroName' from the Microsoft Store manually if this fails."
        Read-Host "Press Enter to exit..."
        exit 1
    }
}

Write-Host "`nVerifying installation..."
# Give WSL a moment if a new install was just initiated
Start-Sleep -Seconds 5
$installedDistrosAfter = wsl --list --quiet
$distroFoundAfter = $false
foreach ($distro in $installedDistrosAfter) {
    if ($distro -eq $DesiredDistroName) {
        $distroFoundAfter = $true
        break
    }
}

if ($distroFoundAfter) {
    Write-Host "✅ WSL distribution '$DesiredDistroName' confirmed to be installed or installation is in progress."
    Write-Host "The system will now restart to complete any pending WSL setup."
    Write-Host "`nFinal WSL status:"
    wsl --status
    Write-Host "`nRestarting in 15 seconds..."
    Start-Sleep -Seconds 15
    Restart-Computer
} else {
    Write-Host "❌ WSL distribution '$DesiredDistroName' could not be confirmed after installation attempt."
    Write-Host "Please check the Microsoft Store for '$DesiredDistroName' or use 'wsl --list --online' to see available distributions."
    Read-Host "Press Enter to exit without restarting..."
    exit 1
}