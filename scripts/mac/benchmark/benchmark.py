import os
import json
import gc
import threading
import time
import subprocess

import torch
import psutil
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS backend not available. Please install a compatible PyTorch 2.x build.")


def get_mps_usage_mb():
    """
    Returns current MPS (Metal Performance Shaders) GPU memory allocated and reserved in MB.
    """
    alloc = torch.mps.current_allocated_memory() / (1024 ** 2)
    resv = torch.mps.driver_allocated_memory() / (1024 ** 2)
    return round(alloc, 2), round(resv, 2)


def get_rss_usage_mb():
    """
    Returns current process resident set size (RSS) in MB.
    """
    proc = psutil.Process(os.getpid())
    return round(proc.memory_info().rss / (1024 ** 2), 2)


def start_rss_sampler(proc, interval=0.05):
    """
    Launch a background thread that samples proc.memory_info().rss every `interval` seconds.
    Returns a dict containing 'peak_rss' and a threading.Event to stop sampling.
    """
    stop_event = threading.Event()
    sampler = {"peak_rss": 0, "stop_event": stop_event}

    def _run():
        while not stop_event.is_set():
            rss = proc.memory_info().rss
            if rss > sampler["peak_rss"]:
                sampler["peak_rss"] = rss
            time.sleep(interval)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return sampler


def get_gpu_temperature(samples=3, delay=0.1):
    """
    Measure GPU temperature multiple times using 'smctemp -g' CLI.
    Returns the average temperature in Celsius, or None if unavailable.
    """
    readings = []
    try:
        for _ in range(samples):
            output = subprocess.check_output(["smctemp", "-g"], text=True)
            readings.append(float(output.strip()))
            time.sleep(delay)
        return round(sum(readings) / len(readings), 2)
    except Exception:
        return None


def get_cpu_temperature(samples=3, delay=0.1):
    """
    Measure CPU temperature multiple times using 'smctemp -c' CLI.
    Returns the average temperature in Celsius, or None if unavailable.
    """
    readings = []
    try:
        for _ in range(samples):
            output = subprocess.check_output(["smctemp", "-c"], text=True)
            readings.append(float(output.strip()))
            time.sleep(delay)
        return round(sum(readings) / len(readings), 2)
    except Exception:
        return None


# ---------- Model List ----------
model_list = [
    "google/gemma-3-1b-it",
]

# ---------- Warm-up Prompts ----------
warm_prompts = [
    "hi, i'd like you to tell me the general weather in moscow",
    (
        "You are a professional tutor explaining the fundamentals of machine learning to a beginner. "
        "This is a very important task that shapes the order of the new world. Start by introducing the "
        "concept of supervised learning. Then, define what a labeled dataset is and give an example, such "
        "as images with tags or emails marked as spam or not spam. Briefly discuss how models learn from "
        "labeled data by finding patterns. Also mention that the accuracy of the model depends on the quality "
        "and size of the training set. Keep the language clear and simple. Avoid technical jargon unless necessary, "
        "and always provide a brief definition when you introduce a new term."
    ),
    (
        "You are a scientific researcher preparing an educational article about renewable energy technologies "
        "for a general audience. Begin by briefly explaining the environmental problems caused by fossil fuels, "
        "including greenhouse gas emissions, air pollution, water contamination, and resource depletion. Emphasize "
        "how these impacts contribute to climate change, biodiversity loss, and public health crises. Then introduce "
        "renewable energy sources such as solar, wind, hydroelectric, geothermal, and biomass. For each energy source, "
        "provide a concise but informative explanation of how it works: for example, describe how photovoltaic cells in "
        "solar panels convert sunlight into electricity through the photovoltaic effect, or how wind turbines transform "
        "kinetic energy from moving air into mechanical energy and subsequently into electrical power via generators. "
        "Include real-world examples, like major solar farms or offshore wind projects, to make the information more tangible. "
        "Discuss the advantages of renewables, including sustainability, reduced operational costs over time, scalability, "
        "and the positive impact on energy independence. Also, acknowledge the limitations and challenges associated with "
        "renewable sources, such as intermittency, the need for energy storage solutions, material sourcing for battery "
        "production, and geographic and economic disparities in access. Conclude by emphasizing the critical importance of "
        "continued investment in energy research, advancements in smart grid technologies, breakthroughs in energy storage "
        "systems, supportive public policies, and international collaboration to accelerate the global transition to a cleaner, "
        "more resilient energy future. Keep the language engaging and accessible without oversimplifying essential technical details. "
        "Aim to leave the reader feeling informed, empowered, and optimistic about the future of renewable energy."
    ),
]

# ---------- Benchmark Prompts ----------
prompt_list = [
    "Translate the following sentence to German: 'The weather is beautiful today.'",
    (
        "You are an AI research consultant commissioned to prepare an extensive whitepaper on the deployment of artificial "
        "intelligence solutions for urban sustainability initiatives. Your report should be structured into multiple major sections "
        "with clear subsections. Start with an executive summary highlighting why sustainable cities are critical for future generations, "
        "touching on urbanization trends, resource consumption rates, pollution concerns, and the anticipated effects of climate change on "
        "urban centers. Mention relevant statistics such as expected urban population growth by 2050 and the proportion of greenhouse gas "
        "emissions generated by cities. Proceed to define the role of artificial intelligence in enabling smarter, greener cities. Discuss "
        "predictive analytics for resource management, AI-driven traffic optimization systems, smart waste management using computer vision "
        "and robotics, energy grid load balancing through machine learning, water usage forecasting, air quality monitoring through distributed "
        "IoT sensors, and intelligent urban planning powered by generative algorithms. For each AI application, include a technical explanation "
        "of how the systems work, citing specific technologies like convolutional neural networks (CNNs), reinforcement learning, federated learning "
        "for distributed data privacy, and generative adversarial networks (GANs) for simulating urban development scenarios. Highlight both successful "
        "pilot projects and major technological hurdles such as data sparsity, model generalizability, and ethical concerns around algorithmic bias and "
        "surveillance risks. Introduce a full section dedicated to energy systems in cities."
    ),
    (
        "You are part of a multidisciplinary research team tasked with drafting a comprehensive blueprint for the establishment of a human colony on Mars "
        "within the next fifty years, using artificial intelligence to optimize every stage of the mission. Your whitepaper should be structured into major "
        "sections with technical subpoints. Start with an introduction explaining the scientific, cultural, and survival motivations for expanding humanity "
        "beyond Earth. Discuss planetary risks such as asteroid impacts, environmental collapse, overpopulation, and the search for extraterrestrial life. Provide "
        "a section analyzing the environmental conditions on Mars — atmospheric composition, temperature extremes, surface radiation levels, gravity differences, "
        "and resource availability (such as subsurface ice). Use precise data where applicable. Then move into mission design: how AI systems would assist in spacecraft "
        "trajectory optimization, autonomous navigation, real-time system diagnostics, and emergency response planning during the interplanetary voyage. Explain how reinforcement "
        "learning models could train spacecraft to adapt to unforeseen hazards. Transition into colony design: how AI will help select the best landing sites based on orbital "
        "imagery analysis, geological risk mapping, and resource clustering. Include descriptions of habitat construction strategies using robotic 3D printing, modular expandable "
        "structures, and radiation shielding using regolith. Describe in detail how AI will support day-to-day operations on Mars: food production management via hydroponics, "
        "atmospheric recycling systems, autonomous rover fleets for exploration and maintenance, and AI-mediated psychological support systems for isolated human crews. Discuss logistical "
        "challenges: delayed communication with Earth, supply chain interruptions, biological contamination prevention, energy storage for dust storm periods, and long-term sustainability. "
        "Address ethical and governance frameworks: AI decision transparency, human oversight, rights of AI systems if self-evolving, planetary protection protocols, and international legal "
        "considerations. Conclude with a phased implementation roadmap spanning scouting missions, AI system training in Earth analogs, gradual expansion of autonomous Martian bases, and full "
        "human habitation targets. Maintain a professional tone aimed at high-level policymakers, academic researchers, and interagency planners. Include inline citations to simulated studies where "
        "appropriate."
    ),
    (
        "You are leading a global task force responsible for designing an international AI-powered pandemic early warning, prevention, and response system intended to mitigate future biological "
        "threats. Your deliverable is a detailed technical and strategic document for public health agencies, governments, and the United Nations. Start by outlining the weaknesses exposed during "
        "recent global pandemics: delayed pathogen detection, disjointed international data sharing, inadequate resource allocation, misinformation propagation, and unequal access to vaccines and "
        "treatments. Provide examples and statistics where possible. Describe the architecture of the proposed system: global sensor networks (biological, atmospheric, wastewater), IoT-enabled monitoring "
        "devices, genomic sequencing hubs, and decentralized data fusion centers. Explain how machine learning algorithms would detect anomalies in public health data streams, wastewater viral load signals, "
        "hospital admission rates, and zoonotic spillover events. Describe methods for training predictive models: unsupervised anomaly detection, time-series forecasting, federated learning to maintain data "
        "sovereignty, and continual learning to adapt to novel pathogens. Discuss rapid response logistics: AI-optimized medical supply chain management, dynamic hospital resource reallocation, and targeted "
        "containment strategies. Explain how reinforcement learning could optimize movement restrictions to minimize socioeconomic disruption while maximizing pathogen containment. Dedicate a section to communication: "
        "using natural language generation (NLG) systems to generate clear, multi-language health advisories. Emphasize trust-building mechanisms like verifiable sourcing and bias mitigation audits. Analyze ethical "
        "risks: potential for surveillance overreach, data privacy erosion, and algorithmic discrimination. Propose transparent governance models, including independent algorithm audits, citizen data ownership rights, "
        "and multi-stakeholder oversight boards. Include a final section projecting the evolution of pathogen threats, considering synthetic biology, climate migration, and antimicrobial resistance. Recommend "
        "adaptive architectures that can defend against unforeseen threat classes. Conclude with a strategic roadmap prioritizing pilot deployments, international treaty development, cross-border simulation exercises, "
        "and public engagement initiatives to build societal resilience. The writing style should balance rigorous technical depth with accessibility for high-level decision-makers across health, policy, technology, "
        "and humanitarian sectors."
    ),
    "What are the main differences between renewable and non-renewable energy sources?",
]

# ---------- Generation Config ----------
MAX_NEW_TOKENS = 256
generation_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)


def benchmark_model_on_prompt_mps(model, tokenizer, prompt, dtype, num_runs=3):
    """
    Runs benchmark for a single prompt on MPS/CPU, returns metrics.
    """
    results = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]

        process = psutil.Process(os.getpid())
        peak_mem_mb = 0
        peak_mps_mb = 0

        # Measure TTFT runs
        ttft_runs = []
        for _ in range(num_runs):
            start_ttft = time.time()
            with torch.no_grad():
                _ = model.generate(max_new_tokens=1, do_sample=False, **inputs)
            end_ttft = time.time()
            ttft_runs.append((end_ttft - start_ttft) * 1000)
        ttft_ms_avg = round(sum(ttft_runs) / len(ttft_runs), 2)

        # Measure full-generation runs
        full_runs = []
        output_tokens = 0
        generated_text = ""
        mem_before_mb = mem_after_mb = memory_delta_mb = None
        cpu_temp_before = cpu_temp_after = None
        gpu_temp_before = gpu_temp_after = None
        mps_alloc_before = mps_resv_before = None
        mps_alloc_after = mps_resv_after = None
        peak_rss_mb = 0

        for i in range(num_runs):
            # a) “Before” snapshots
            mem_before_mb = process.memory_full_info().rss / (1024 * 1024)
            gpu_temp_before = get_gpu_temperature()
            cpu_temp_before = get_cpu_temperature()
            sampler = start_rss_sampler(process, interval=0.05)

            mps_alloc_before, mps_resv_before = get_mps_usage_mb()
            rss_before = get_rss_usage_mb()

            # b) Run inference
            start_full = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **inputs,
                )
            end_full = time.time()

            # c) “After” snapshots
            mps_alloc_after, mps_resv_after = get_mps_usage_mb()
            # Track peak MPS allocation in MB
            if mps_alloc_after > peak_mps_mb:
                peak_mps_mb = mps_alloc_after
            mem_after_mb = process.memory_full_info().rss / (1024 * 1024)
            sampler["stop_event"].set()
            time.sleep(0.05)
            peak_rss_mb = sampler["peak_rss"] / (1024 ** 2)
            gpu_temp_after = get_gpu_temperature()
            cpu_temp_after = get_cpu_temperature()

            memory_delta_mb = mem_after_mb - mem_before_mb
            current_peak = max(mem_before_mb, mem_after_mb)
            if current_peak > peak_mem_mb:
                peak_mem_mb = current_peak

            iter_time_ms = (end_full - start_full) * 1000
            full_runs.append(iter_time_ms)

            if i == 0:
                actual_output_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                generated_text = tokenizer.decode(actual_output_ids, skip_special_tokens=True)
                output_tokens = len(actual_output_ids)

            torch.mps.empty_cache()
            gc.collect()

        avg_time_ms = round(sum(full_runs) / len(full_runs), 3) if full_runs else 0
        tokens_per_sec = round(output_tokens / (avg_time_ms / 1000.0), 2) if full_runs else 0
        peak_memory_mb = round(peak_mem_mb, 2)
        cpu_temp_delta_c = (
            round(cpu_temp_after - cpu_temp_before, 2)
            if cpu_temp_before is not None and cpu_temp_after is not None
            else None
        )
        gpu_temp_delta_c = (
            round(gpu_temp_after - gpu_temp_before, 2)
            if gpu_temp_before is not None and gpu_temp_after is not None
            else None
        )

        results = {
            "prompt": prompt,
            "status": "success",
            "error_message": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "avg_time_ms": avg_time_ms,
            "tokens_per_sec": tokens_per_sec,
            "ttft_ms_avg": ttft_ms_avg,
            "output_text_preview": generated_text[:100] + "...",
            "cpu_temp_before_c": cpu_temp_before,
            "cpu_temp_after_c": cpu_temp_after,
            "cpu_temp_delta_c": cpu_temp_delta_c,
            "gpu_temp_before_c": gpu_temp_before,
            "gpu_temp_after_c": gpu_temp_after,
            "gpu_temp_delta_c": gpu_temp_delta_c,
            "mps_alloc_before_mb": mps_alloc_before,
            "mps_resv_before_mb": mps_resv_before,
            "rss_before_mb": rss_before,
            "mps_alloc_after_mb": mps_alloc_after,
            "mps_resv_after_mb": mps_resv_after,
            "rss_after_mb": mem_after_mb,
            "peak_rss_during_run_mb": peak_rss_mb,
            "peak_mps_mb": peak_mps_mb,
        }
    except Exception as e:
        results = {"prompt": prompt, "status": "failed", "error_message": str(e)}
    return results


def run_full_benchmark_mps(output_filename="benchmark_results_mps.json"):
    """
    Runs benchmarks for all models and prompts on MPS/CPU, saving results.
    """
    all_results = []
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    benchmark_dtype = torch.float16
    print(f"--- Running Benchmark on device: {device} with dtype: {benchmark_dtype} ---")

    try:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
            print("Logged in to Hugging Face Hub successfully.")
        else:
            print("HF_TOKEN not set. Skipping login.")
    except Exception as e:
        print(f"Warning: Hugging Face login failed or skipped: {e}")

    for model_id in model_list:
        print(f"\n{'='*20} Benchmarking Model: {model_id} {'='*20}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            print(f"Loading model {model_id} to CPU (dtype={benchmark_dtype})...")
            load_start = time.time()
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=benchmark_dtype)
            load_end = time.time()

            rss_after_load = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            print(f"RSS after loading onto CPU: {rss_after_load:.2f} MB")

            print(f"Moving model to {device} and emptying cache...")
            model.to(device)
            model.eval()
            torch.mps.empty_cache()
            gc.collect()

            rss_after_move = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            model_load_time = load_end - load_start
            print(f"Model on {device} in {model_load_time:.2f}s; RSS after move: {rss_after_move:.2f} MB")

            print("Running global warm-up...")
            for w_prompt in warm_prompts:
                w_inputs = tokenizer(w_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    _ = model.generate(**w_inputs, max_new_tokens=16, do_sample=False)
            print("Global warm-up complete.")

            for prompt_text in prompt_list:
                print(f"--- Prompt: '{prompt_text[:50]}...' ---")
                prompt_results = benchmark_model_on_prompt_mps(model, tokenizer, prompt_text, benchmark_dtype, num_runs=3)
                prompt_results.update(
                    {
                        "model_id": model_id,
                        "device": device,
                        "dtype": str(benchmark_dtype),
                        "model_load_time_s": round(model_load_time, 2),
                    }
                )
                all_results.append(prompt_results)
                with open(output_filename, "w") as f:
                    json.dump(all_results, f, indent=4)

        except Exception as e:
            print(f"FATAL ERROR for {model_id}: {e}")
            all_results.append({
                "model_id": model_id,
                "status": "load_or_setup_failed",
                "error_message": str(e),
                "device": device,
                "dtype": str(benchmark_dtype),
            })
        finally:
            print(f"Cleaning up resources for {model_id}...")
            try:
                del model
                del tokenizer
            except NameError:
                pass
            if device == "mps":
                torch.mps.empty_cache()
            print("Cleanup complete.")
            with open(output_filename, "w") as f:
                json.dump(all_results, f, indent=4)

    print(f"\nBenchmark run complete. Results saved to {output_filename}")


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"benchmark_results_mps_{timestamp}.json"
    run_full_benchmark_mps(output_filename=output_file)