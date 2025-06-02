# run_benchmark_tensorrt_llm.ps1
# This script launches WSL, activates the virtual environment, and runs the benchmark.

$Distro = "Ubuntu"
$WSLUser = "benchmark"
$WSLBenchmarkPath = "/home/benchmark/benchmark_env"

Write-Host "🚀 Launching benchmark script inside WSL..."

# Build the WSL command
$Command = @"
cd $WSLBenchmarkPath
source .venv/bin/activate

### Clone model repositories (change paths, git clone the models into "models" directory)
# Qwen/Qwen2.5-1.5B-Instruct
echo "Cloning Qwen2.5-1.5B-Instruct model..."
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

# google/gemma-3-1b-it
echo "Cloning Gemma-3-1B-IT model..."
git clone https://huggingface.co/google/gemma-3-1b-it

# meta-llama/Llama-3.2-1B-Instruct
echo "Cloning Llama-3.2-1B-Instruct model..."
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

echo "✅ Model repositories cloned."

### Convert models to TensorRT-LLM format
echo "Converting models to TensorRT-LLM format..."

# Convert Qwen2.5-1.5B-Instruct
python3 TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py --model_dir Qwen2.5-1.5B-Instruct \
                              --output_dir ./Qwen2.5-1.5B-Instruct_TensorRT-LLM_Checkpoint \
                              --dtype float16

trtllm-build --checkpoint_dir ./Qwen2.5-1.5B-Instruct_TensorRT-LLM_Checkpoint \
            --output_dir ./Qwen2.5-1.5B-Instruct_TensorRT-LLM \
            --gemm_plugin float16

# Convert Gemma-3-1B-IT
python3 TensorRT-LLM/examples/models/core/gemma/convert_checkpoint.py --model_dir gemma-3-1b-it \
                              --output_dir ./gemma-3-1b-it_TensorRT-LLM_Checkpoint \
                              --dtype float16
trtllm-build --checkpoint_dir ./gemma-3-1b-it_TensorRT-LLM_Checkpoint \
            --output_dir ./gemma-3-1b-it_TensorRT-LLM \
            --gemm_plugin float16

# Convert Llama-3.2-1B-Instruct
python3 TensorRT-LLM/examples/models/core/llama/convert_checkpoint.py --model_dir Llama-3.2-1B-Instruct \
                              --output_dir ./Llama-3.2-1B-Instruct_TensorRT-LLM_Checkpoint \
                              --dtype float16

trtllm-build --checkpoint_dir ./Llama-3.2-1B-Instruct_TensorRT-LLM_Checkpoint \
            --output_dir ./Llama-3.2-1B-Instruct_TensorRT-LLM \
            --gemm_plugin float16

echo "✅ Models converted to TensorRT-LLM format."

python3 run_benchmark_tensorrt_llm.py
"@

# Run it in WSL
wsl -d $Distro --user $WSLUser -- bash -c "$Command"

Write-Host "`n✅ TensorRT-LLM Benchmarking completed."
