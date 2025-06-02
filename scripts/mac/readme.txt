MACOS BENCHMARK SUITE - README

Thank you for helping us by running this benchmark! Please follow the steps below carefully.

============================================
IMPORTANT PREREQUISITES (One-Time Setup)
============================================

Before running the benchmark for the first time, please ensure the following:

1.  ENABLE DEVELOPER TOOLS FOR TERMINAL:
    *   Open System Settings.
    *   Navigate to Privacy & Security -> Developer Tools.
    *   Find "Terminal" in the list and toggle it ON.
    *   (If you do not see the "Developer Tools" option, or Terminal is not in the list after opening it at least once, please contact us.)

2.  GRANT FULL DISK ACCESS TO TERMINAL (Highly Recommended):
    *   Open System Settings.
    *   Navigate to Privacy & Security -> Full Disk Access.
    *   Click the '+' button, navigate to /Applications/Utilities/, select Terminal.app, and add it.
    *   Ensure the toggle next to "Terminal" is ON.
    *   (This step helps prevent permission issues and is highly recommended for a smooth installation.)

3.  ACTIVE INTERNET CONNECTION:
    *   An internet connection is required during the installation phase to download necessary components like Homebrew, Python packages, and benchmark models (on their first run).

============================================
RUNNING THE BENCHMARKS
============================================

1.  EXTRACT THE ARCHIVE:
    *   Extract the provided ZIP archive (e.g., benchmark_suite.zip) directly onto your Desktop.
    *   This should create a folder named "mac" (i.e., ~/Desktop/mac/).

2.  RUN THE BENCHMARK APPLICATION:
    *   Open the "mac" folder located on your Desktop.
    *   Double-click the "Run Benchmark.app" application.
    *   A Terminal window will open and guide you through the process.
    *   This application will execute a series of steps, including:
        *   Installing necessary tools (Homebrew, Python, smctemp) if not already present.
        *   Setting up a Python virtual environment and installing required packages.
        *   Running a benchmark using PyTorch MPS.
        *   Running a benchmark using MLX for Apple Silicon.
    *   IMPORTANT:
        *   You will likely be PROMPTED TO ENTER YOUR MACOS ADMINISTRATOR PASSWORD during the initial setup (e.g., for "smctemp" or Homebrew). Please enter it when requested.
        *   The first time you run the benchmarks, models will be downloaded, which can take some time depending on your internet speed.
        *   The entire benchmark process can take a significant amount of time to complete. Please be patient and let it run until the end.
        *   Make sure all other applications are closed for best performance.

3.  MONITOR THE PROCESS:
    *   The Terminal window will display progress messages for each stage.
    *   Wait until you see a message similar to:
        "ALL BENCHMARK PROCESSES ATTEMPTED.
        ...
        Please verify the files exist and then send them to us.
        Press Enter to close this Terminal window."

4.  COLLECT AND SEND RESULTS:
    *   Once the process is finished, press Enter in the Terminal window to close it.
    *   Navigate to the "mac/benchmark/results/" folder on your Desktop.
    *   Inside this "results" folder, you should find several JSON files, including:
        *   system_info.json (if a system collection script is part of the suite)
        *   generation_config.json
        *   A file starting with "benchmark_results_mps_" followed by a timestamp (e.g., benchmark_results_mps_20250601-103000.json)
        *   A file starting with "benchmark_results_mlx_" followed by a timestamp (e.g., benchmark_results_mlx_20250601-103500.json)
    *   Please send us ALL of these relevant .json files from the "results" directory. You can send us the whole "results" folder too.

=========================================================
CLEANING UP AFTER THE BENCHMARK (Optional but Recommended)
=========================================================

After you have successfully run the benchmarks and sent us the results, you can clean up some components.

1.  RUN THE CLEANUP APPLICATION:
    *   Open the "mac" folder on your Desktop.
    *   Double-click the "Run Cleanup.app" application.
    *   A Terminal window will open. This script attempts to:
        *   Uninstall the "smctemp" utility (you may be prompted for your administrator password).
        *   Remove the temporary "smctemp" build directory.
    *   NOTE: This cleanup script will NOT uninstall Homebrew, Python, or downloaded benchmark models/packages.
    *   Press Enter in the Terminal window to close it when the cleanup is complete.

2.  REMOVE THE BENCHMARK FOLDER:
    *   After running the cleanup script, drag the entire "mac" folder (from your Desktop) to the Trash. This will remove all benchmark scripts, the Python virtual environment, and any remaining files related to this benchmark suite.

============================================
TROUBLESHOOTING & CONTACT
============================================

*   If you encounter "Operation not permitted" errors, ensure you have completed the "Developer Tools" and "Full Disk Access" prerequisites for Terminal.
*   If "Run Benchmark.app" or "Run Cleanup.app" fails to find its associated .sh script (e.g., run_full_benchmark.sh or cleanup.sh), make sure both the .app file and the .sh file are directly inside the "mac" folder and have not been moved or renamed.

If you have any issues or questions, please do not hesitate to contact us.

Thank you again for your participation!