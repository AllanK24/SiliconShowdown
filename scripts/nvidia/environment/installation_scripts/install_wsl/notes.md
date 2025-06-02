## 📝 Notes: Converting a PowerShell Script to `.exe` using `ps2exe`

### 🎯 Goal

Turn your PowerShell script (`install_wsl.ps1`) into a standalone `.exe` file for easy distribution and execution.

---

### ✅ Step-by-Step Instructions

#### 1. **Install `ps2exe` module**

First, open PowerShell **as Administrator** and run:

```powershell
Install-Module -Name ps2exe -Scope CurrentUser
```

> 🔒 You may be prompted to trust the repository — choose `Yes`.

---

#### 2. **Allow Script Execution Temporarily**

Set the execution policy just for this session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

#### 3. **Import the `ps2exe` Module**

```powershell
Import-Module ps2exe
```

---

#### 4. **Convert the Script to EXE**

Now run the conversion:

```powershell
Invoke-ps2exe "install_wsl.ps1" "install_wsl.exe"
```

* The first argument is your `.ps1` script file.
* The second is the name of the `.exe` you want to generate.

---

### 📁 Output

The `install_wsl.exe` file will be created in the current working directory. You can now:

* Run it by double-clicking
* Right-click → **Run as administrator** to ensure elevation

---

### 🛠 Tip: Always Test the `.exe`

Make sure to test the `.exe`:

* With and without elevation
* On a clean system
* After a reboot