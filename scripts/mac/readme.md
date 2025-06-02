# macOS Benchmark Suite

Thank you for helping us by running this benchmark! Please follow the steps below carefully.

## ⚠️ Important Prerequisites (One-Time Setup)

Before running the benchmark for the first time, please ensure the following:

1.  **Enable Developer Tools for Terminal:**
    *   Open **System Settings**.
    *   Navigate to **Privacy & Security** → **Developer Tools**.
    *   Find **Terminal** in the list and toggle it **ON**.
    *   *(If you do not see the "Developer Tools" option, or Terminal is not in the list after opening it at least once, please contact us.)*

2.  **Grant Full Disk Access to Terminal (Optional but Recommended):**
    *   Open **System Settings**.
    *   Navigate to **Privacy & Security** → **Full Disk Access**.
    *   Click the `+` button, navigate to `/Applications/Utilities/`, select `Terminal.app`, and add it.
    *   Ensure the toggle next to **Terminal** is **ON**.
    *   *(This step helps prevent permission issues with tools and scripts that access various system locations and is highly recommended for a smooth installation.)*

3.  **Active Internet Connection:**
    *   An internet connection is required during the installation phase to download necessary components like Homebrew and other software dependencies.

---

## Running the Benchmark

1.  **Extract the Archive:**
    *   Extract the provided ZIP archive (e.g., `benchmark_suite.zip`) directly onto your **Desktop**.
    *   This should create a folder named `mac` (i.e., `~/Desktop/mac/`).

2.  **Run the Benchmark Application:**
    *   Open the `mac` folder located on your Desktop.
    *   Double-click the `Run Benchmark.app` application.
    *   A Terminal window will open and guide you through the process.
    *   **Important:**
        *   The script will install necessary tools like Homebrew (if not already present), Python 3.11, and a utility called `smctemp`.
        *   You will likely be **prompted to enter your macOS administrator password** during this installation (e.g., for `smctemp`). Please enter it when requested.
        *   The entire benchmark process, especially the final benchmark run, can take a significant amount of time to complete. Please be patient.

3.  **Monitor the Process:**
    *   The Terminal window will display progress messages.
    *   Wait until you see the following message:
        ```
        Benchmark run complete…
        ...
        PROCESS FINISHED!
        ...
        Please send these files to us.
        Press Enter to close this Terminal window.
        ```

4.  **Collect and Send Results:**
    *   Once the process is finished, press `Enter` in the Terminal window to close it.
    *   Inside the `mac` folder on your Desktop, you will find two JSON files:
        *   `system_info.json`
        *   `benchmark_result_mps.json` (or a similar name for the benchmark results)
    *   Please **send us both of these `.json` files**.

---

## Cleaning Up After the Benchmark (Optional but Recommended)

After you have successfully run the benchmark and sent us the results, you can clean up the components installed on your system.

1.  **Run the Cleanup Application:**
    *   Open the `mac` folder on your Desktop.
    *   Double-click the `Run Cleanup.app` application.
    *   A Terminal window will open. This script will:
        *   Attempt to uninstall the `smctemp` utility from your system (you may be prompted for your administrator password).
        *   Remove the temporary `smctemp` build directory.
    *   **Note:** This cleanup script will *not* uninstall Homebrew or Python, as you might use these for other purposes.
    *   Press `Enter` in the Terminal window to close it when the cleanup is complete.

2.  **Remove the Benchmark Folder:**
    *   After running the cleanup script, **drag the entire `mac` folder** (from your Desktop) to the **Trash**. This will remove all the benchmark scripts, the Python virtual environment, and any remaining files.

---

## Troubleshooting & Contact

*   If you encounter "Operation not permitted" errors, ensure you have completed the **Developer Tools** and **Full Disk Access** prerequisites for Terminal.
*   If `Run Benchmark.app` or `Run Cleanup.app` fails to find its associated `.sh` script, make sure both the `.app` file and the `.sh` file (e.g., `run_full_benchmark.sh` or `cleanup.sh`) are directly inside the `mac` folder and have not been moved or renamed.

If you have any issues or questions, please do not hesitate to contact us.

Thank you again for your participation!