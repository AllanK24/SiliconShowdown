# 🧪 AI Benchmarking Setup Guide (WSL + Windows)

This guide will walk you through installing WSL, configuring your system, and running the AI benchmarking scripts required for the **Silicon Showdown** project.

> 📝 **Important:** These instructions assume you are running **Windows 10/11** with administrator privileges.

---

## ✅ Step-by-Step Instructions

### 🧭 1. Open PowerShell as Administrator

#### 🔹 How to open it:
1. Click on the **Start** menu (Windows logo).
2. Type `powershell`.
3. **Right-click** on **Windows PowerShell** (or **Terminal** on Windows 11).
4. Choose **"Run as administrator"**.

> ⚠️ You **must** run it as Administrator to install WSL and make system-level changes.

---

### 2. Install WSL2 with Ubuntu (if not already installed)

> If you already have Ubuntu installed in WSL2, you can skip this step.

```powershell
wsl --install
```

> This will install WSL

---

### Restart Computer
```powershell
Restart-Computer -Force
```

### Open PowerShell as administrator again, run the following command to install Ubuntu 24.04:
```powershell
wsl --install Ubuntu-24.04
```

### If you have WSL installed, run:
``powershell
wsl --update
```
to update WSL

### 👤 3. Create a Dedicated Linux User for Benchmarking (will appear after you installed Ubuntu):

Regardless of whether you already have Ubuntu installed in WSL or are installing it now:

#### ➤ You **must** create a new Linux user named:

- **Username:** `benchmark`  
- **Password:** `benchmark2025`

#### ✅ If You're Launching Ubuntu for the First Time:
It will prompt you to create a new user — simply enter the values above.

#### 🔁 If Ubuntu is Already Installed:
Do the following inside your existing Ubuntu shell:

```bash
sudo adduser benchmark
```

Then set the password when prompted:
```
Enter new UNIX password: benchmark2025
```

Add this user to `sudo` group:

```bash
sudo usermod -aG sudo benchmark
```

Choose the newly created user:
```bash
su - benchmark
```

---

### 4. Install required dependencies for benchmarking

1) Run the following command to allow custom powershell script execution:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

2) Get into a directory with benchmarking files shared to you:
```powershell
cd full_path_to_dir
```

3) Run the script to setup benchmarking environment

```powershell
.\setup_dev_env_wsl.ps1
```

---
