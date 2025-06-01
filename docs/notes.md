* Drivers for Nvidia are needed to be installed manually

* Replace os.environ.get("HF_TOKEN"), set up the token directly, otherwise import of models won't work

* We need to agree on the specific version of Python to use, probably it will depend on the Mac dependencies

* When making the review of our project, we need to explicitly mention that we conduct the tests of Nvidia GPUs on Windows in WSL, not on the native Windows OS, due to TensorRT LLM not being supported on Windows natively. So the actual performance on the native Linux OS may differ.

------ WSL Notes ------
* Before making inference, allow WSL to use maximum memory

------ Questions ------
1. Temperature and hyperparams of LLM generation process
2. TTFT (alr done for mac)
3. Acceleration: MLX and Nvidia for Windows (https://github.com/sgl-project/sglang, https://github.com/vllm-project/vllm)
4. Change CUDA download script, set up the .exe to install via command line code
5. Match benchmark functions
8. Come up with requirements file
9. Remove redundant records from benchmark.py
10. Come up with a script to install any python version if python doesn't exist (alr done for mac)
11. Update the model list, add the diffusion LM if feasable
