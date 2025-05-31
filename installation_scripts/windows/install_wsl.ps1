# install_wsl.ps1

# Check if script is running as administrator
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
    [Security.Principal.WindowsBuiltInRole] "Administrator"))
{
    Write-Warning "This script must be run as Administrator. Right-click and select 'Run with PowerShell as Administrator'."
    Exit
}

# Check and Install WSL (Windows Subsystem for Linux)
try {
    $wslStatus = wsl --status 2>&1
    if ($wslStatus -match "Default Version") {
        Write-Host "WSL is already installed. Updating to latest version..."
        wsl --update
    } else {
        Write-Host "Installing WSL..."
        wsl --install
    }
} catch {
    Write-Host "WSL not detected. Installing..."
    wsl --install
}

# Show final WSL status
Write-Host "`nFinal WSL status:"
wsl --status

Write-Host "`nWSL setup complete."
Write-Host "The system will restart in 15 seconds to complete setup..."

Start-Sleep -Seconds 15
Restart-Computer