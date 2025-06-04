## ----- General Notes -----
* Drivers for Nvidia are needed to be installed manually

* Replace os.environ.get("HF_TOKEN"), set up the token directly, otherwise import of models won't work

* **We need to agree on the specific version of Python to use, probably it will depend on the Mac dependencies - 3.11.0**

* When making the review of our project, we need to explicitly mention that we conduct the tests of Nvidia GPUs on Windows in WSL, not on the native Windows OS, due to TensorRT LLM not being supported on Windows natively. So the actual performance on the native Linux OS may differ.

* ### ` **Before sharing the project for benchmarking, hardcode token into the code, otherwise an error will be thrown** ### ` 

## ------ WSL Notes ------
* **Before making inference, allow WSL to use maximum memory** ✅
* **We need to ship the python benchmarking files in the same folder as the installation and setup scripts, because the benchmarking python files will be copied to the WSL benchmarking directory during the installation process in `setup_dev_env_wsl.ps1` script.** ✅
* Add run_benchmark_tensorrt_llm.py to the list of files to be copied to the WSL benchmarking directory in `setup_dev_env_wsl.ps1` script. ✅
* **Write a script that asks user if the cleaning up needed after benchmarking, and if yes, removes the benchmarking files including the WSL itself.** ✅
* **Change setup_dev_env_wsl.ps1 script to also copy the TensorRT LLM benchmarking script, the `configs` dir and the `generation_config.json` to the WSL benchmarking directory.** ✅
* **Double check the paths everywhere to avoid errors with files not found.** ✅
* Adjust the requirements.txt file and requirements in setup_dev_env_wsl.ps1 script. ✅

## ------ Questions ------
1. Temperature and hyperparams of LLM generation process
2. TTFT (alr done for mac)
3. Acceleration: MLX and Nvidia for Windows (https://github.com/sgl-project/sglang, https://github.com/vllm-project/vllm)
4. Change CUDA download script, set up the .exe to install via command line code
5. Match benchmark functions
8. Come up with requirements file
9. Remove redundant records from benchmark.py
10. Come up with a script to install any python version if python doesn't exist (alr done for mac)
11. Update the model list, add the diffusion LM if feasable
