# run_benchmark_tensorrt_llm.ps1
# This script launches WSL, activates the virtual environment, runs the benchmark, and retrieves results.

$Distro = "Ubuntu"
$WSLUser = "benchmark"
$WSLBenchmarkPath = "/home/benchmark/benchmark_env"

Write-Host "🚀 Launching benchmark script inside WSL..."

# Build the WSL bash commands
$Command = @"
cd $WSLBenchmarkPath
source .venv/bin/activate

mkdir -p models/qwen models/gemma models/llama

# Qwen
echo "Cloning Qwen model..."
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct models/qwen/Qwen2.5-1.5B-Instruct

echo "Converting Qwen..."
python3 TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py --model_dir models/qwen/Qwen2.5-1.5B-Instruct \
    --output_dir models/qwen/Qwen2.5-1.5B-Instruct_TensorRT-LLM_Checkpoint --dtype float16

trtllm-build --checkpoint_dir models/qwen/Qwen2.5-1.5B-Instruct_TensorRT-LLM_Checkpoint \
    --output_dir models/qwen/Qwen2.5-1.5B-Instruct_TensorRT-LLM \
    --gemm_plugin auto --max_batch_size 1 --max_input_len 512 \
    --max_num_tokens 512 --max_seq_len 4096 --logits_dtype float16

# Gemma
echo "Cloning Gemma model..."
git clone https://huggingface.co/google/gemma-3-1b-it models/gemma/gemma-3-1b-it

echo "Converting Gemma..."
python3 TensorRT-LLM/examples/models/core/gemma/convert_checkpoint.py --model_dir models/gemma/gemma-3-1b-it \
    --output_dir models/gemma/gemma-3-1b-it_TensorRT-LLM_Checkpoint --dtype float16

trtllm-build --checkpoint_dir models/gemma/gemma-3-1b-it_TensorRT-LLM_Checkpoint \
    --output_dir models/gemma/gemma-3-1b-it_TensorRT-LLM \
    --gemm_plugin auto --max_batch_size 1 --max_input_len 512 \
    --max_num_tokens 512 --max_seq_len 4096 --logits_dtype float16

# Llama
echo "Cloning Llama model..."
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct models/llama/Llama-3.2-1B-Instruct

echo "Converting Llama..."
python3 TensorRT-LLM/examples/models/core/llama/convert_checkpoint.py --model_dir models/llama/Llama-3.2-1B-Instruct \
    --output_dir models/llama/Llama-3.2-1B-Instruct_TensorRT-LLM_Checkpoint --dtype float16

trtllm-build --checkpoint_dir models/llama/Llama-3.2-1B-Instruct_TensorRT-LLM_Checkpoint \
    --output_dir models/llama/Llama-3.2-1B-Instruct_TensorRT-LLM \
    --gemm_plugin auto --max_batch_size 1 --max_input_len 512 \
    --max_num_tokens 512 --max_seq_len 4096 --logits_dtype float16

echo "✅ All models converted."

# Run the benchmark
python3 run_benchmark_tensorrt_llm.py
"@

# Execute the commands inside WSL
wsl -d $Distro --user $WSLUser -- bash -c "$Command"

# Fetch benchmark result file to Windows
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ResultsDir = Join-Path $ScriptDir "results"
$WSLResultsDir = "/home/benchmark/benchmark_env"

if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

# Locate the output file in WSL
$BenchmarkFileName = wsl -d $Distro --user $WSLUser -- bash -c "
    ls -t $WSLResultsDir/llm_benchmark_results_tensorrt_llm_*.json 2>/dev/null | head -n 1
" | ForEach-Object { $_.Trim() }

if ($BenchmarkFileName) {
    $BenchmarkFileName = [System.IO.Path]::GetFileName($BenchmarkFileName)
    $WSLFilePath = "$WSLResultsDir/$BenchmarkFileName"
    $LocalFilePath = Join-Path $ResultsDir $BenchmarkFileName

    Write-Host "💾 Copying $BenchmarkFileName to results/ folder..."
    wsl -d $Distro --user $WSLUser -- bash -c "cat $WSLFilePath" > $LocalFilePath

    if (Test-Path $LocalFilePath) {
        Write-Host "✅ $BenchmarkFileName copied successfully to results/."
    } else {
        Write-Host "❌ Failed to copy $BenchmarkFileName."
    }
} else {
    Write-Host "⚠️ No benchmark result file found in WSL."
}

Write-Host "`n✅ TensorRT-LLM Benchmarking completed."
