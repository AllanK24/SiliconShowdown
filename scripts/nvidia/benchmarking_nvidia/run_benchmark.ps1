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

# Define base paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ResultsDir = Join-Path $ScriptDir "results"
$WSLResultsDir = "/home/benchmark/benchmark_env"

# Ensure results/ directory exists
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

Write-Host "📦 Fetching benchmark results from WSL..."

# Get list of JSON files to copy (benchmark result + generation config)
$JsonFiles = @(
    "generation_config.json"
)

# Fetch the benchmark file name dynamically from WSL using timestamp prefix
$BenchmarkFileName = wsl -d $Distro --user $WSLUser -- bash -c "
    ls -t $WSLResultsDir/llm_benchmark_results_cuda_*.json 2>/dev/null | head -n 1
" | ForEach-Object { $_.Trim() }

if ($BenchmarkFileName) {
    $BenchmarkFileName = [System.IO.Path]::GetFileName($BenchmarkFileName)
    $JsonFiles += $BenchmarkFileName
}

foreach ($JsonFile in $JsonFiles) {
    $WSLPath = "$WSLResultsDir/$JsonFile"
    $WindowsPath = Join-Path $ResultsDir $JsonFile

    Write-Host "💾 Copying $JsonFile to results/ folder..."
    wsl -d $Distro --user $WSLUser -- bash -c "cat $WSLPath" > $WindowsPath

    if (Test-Path $WindowsPath) {
        Write-Host "✅ $JsonFile copied successfully."
    } else {
        Write-Host "❌ Failed to copy $JsonFile."
    }
}
