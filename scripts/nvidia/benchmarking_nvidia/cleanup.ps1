# cleanup.ps1
# This script is designed to clean up the WSL environment used for benchmarking by either removing all WSL distributions or just the 'benchmark' user.

Write-Host "=== Benchmark Environment Cleanup Utility ===" -ForegroundColor Cyan
Write-Host "This script can either:"
Write-Host "1. Completely remove WSL (all distributions, including the benchmarking environment)"
Write-Host "2. OR, only remove the 'benchmark' user inside the WSL benchmarking distro"
Write-Host " "

$deleteWSL = Read-Host "Do you want to completely remove WSL? (Y/N)"
if ($deleteWSL -eq "Y" -or $deleteWSL -eq "y") {
    Write-Host "Unregistering all WSL distributions..." -ForegroundColor Yellow
    wsl --unregister Ubuntu
    Write-Host "WSL distributions removed."
    Write-Host "You can also uninstall 'Windows Subsystem for Linux' and 'Ubuntu' from Windows Settings > Apps manually if needed."
    exit
}

$deleteUser = Read-Host "Do you want to remove the 'benchmark' user and all their data from the WSL benchmarking environment? (Y/N)"
if ($deleteUser -eq "Y" -or $deleteUser -eq "y") {
    Write-Host "Attempting to remove 'benchmark' user and its files..." -ForegroundColor Yellow

    # Run a WSL command to delete the user and their home directory
    $command = @"
sudo deluser --remove-home benchmark && echo 'User deleted successfully.' || echo 'Failed to delete user. Try manually or check if WSL is running.'
"@
    wsl bash -c $command
    Write-Host "Cleanup complete."
} else {
    Write-Host "No changes were made." -ForegroundColor Green
}