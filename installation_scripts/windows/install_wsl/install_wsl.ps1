# install_wsl.ps1

Write-Host "Please make sure you close all the important applications, as this script will restart your computer."

Write-Host "Starting WSL installation and setup..."

# Check if WSL is installed
$wslStatus = wsl --status 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ WSL is already installed. Checking for updates..."
    wsl --update
} else {
    Write-Host "❌ WSL not detected. Installing..."
    wsl --install
}

Write-Host "`nFinal WSL status:"
wsl --status

Write-Host "`nSetup complete. Restarting in 15 seconds..."
Start-Sleep -Seconds 15
Restart-Computer