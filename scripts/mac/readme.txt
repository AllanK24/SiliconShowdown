Hello! Here are the steps needed for you to successfully run the benchmark:
1. open system settings, navigate to privacy & security -> Developer Tools and turn on Terminal. (if this step is not available, please contact us)
2. Extract the zip file we provided to the desktop
3. Open a new Terminal window, type "cd Desktop/mac" then enter
4. type "chmod +x install_mac.sh" then enter
5. type "./install_mac.sh" and enter for it to run(you will be asked to enter your password in this step)
6. type "source environment/ai_benchmark_env/bin/activate" and enter
7. type "python environment/collect_system_info.py" and enter
8. type "python benchmark/benchmark.py" and enter(this will run the benchmark and may take a while)
9. when you see "benchmark run complete ..." it means everything is done and we need you to send us the benchmark_result_mps.json file and the system_info.json file

Thank you for your help! 