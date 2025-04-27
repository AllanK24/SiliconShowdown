# PowerShell Script: Check and Install Python 3.11.9

# Function to check if python is installed
function Check-Python {
    try {
        $python_version = python --version 2>$null
        if ($python_version -and $python_version -match "3\.11\.9") {
            Write-Host "Python 3.11.9 is already installed: $python_version"
            return $true
        } else {
            return $false
        }
    } catch {
        return $false
    }
}

# Function to download and install Python 3.11.9
function Install-Python {
    Write-Host "Downloading Python 3.11.9 installer..."
    $installer_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    $installer_path = "$env:TEMP\python-3.11.9-installer.exe"

    Invoke-WebRequest -Uri $installer_url -OutFile $installer_path

    Write-Host "Installing Python 3.11.9..."
    Start-Process -FilePath $installer_path -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait

    Remove-Item $installer_path

    Write-Host "Python 3.11.9 installation complete!"
}

# Main execution
if (Check-Python) {
    Write-Host "No action needed."
} else {
    Write-Host "Python 3.11.9 not found. Installing Python 3.11.9..."
    Install-Python
}
