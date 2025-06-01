# run_benchmark.ps1
# This script launches WSL, activates the virtual environment, and runs the benchmark.

$Distro = "Ubuntu"
$WSLUser = "benchmark"
$WSLBenchmarkPath = "/home/benchmark/benchmark_env"

Write-Host "🚀 Launching benchmark script inside WSL..."

# Build the WSL command
$Command = @"
cd $WSLBenchmarkPath
source .venv/bin/activate
python3 run_benchmark.py
"@

# Run it in WSL
wsl -d $Distro --user $WSLUser -- bash -c "$Command"

Write-Host "`n✅ Benchmark completed."
