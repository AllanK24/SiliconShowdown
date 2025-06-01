Follow these steps to run the benchmark:

1.  Open System Settings, go to Privacy & Security → Developer Tools, and enable Terminal. (If you do not see this option, please contact us.)

2.  Extract the provided ZIP archive onto your Desktop.

3.  In a new Terminal window, run:

```bash
cd ~/Desktop/mac
```

4.  Make the installer executable by running:

```bash
chmod +x install_mac.sh
```

5.  Execute the installer script (you may be prompted to enter your password):

```bash
./install_mac.sh
```

6.  Activate the virtual environment:

```bash
source environment/ai_benchmark_env/bin/activate
```

7.  Collect system information:

```bash
python environment/collect_system_info.py
```

8.  Run the benchmark:

```bash
python benchmark/benchmark.py
```

- This process can take a while to complete.

9.  Once you see:

```
Benchmark run complete…
```

the process is finished. Please send us both `benchmark_result_mps.json` and `system_info.json` from the benchmark folder.

Thank you!