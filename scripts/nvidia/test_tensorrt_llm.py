import os
from timeit import default_timer as timer
from tensorrt_llm import LLM, SamplingParams, BuildConfig
from transformers import AutoTokenizer
from huggingface_hub import login

def main():
    # prompts = [
    #     "Translate the following sentence to German: 'The weather is beautiful today.'",
    #     "Explain the concept of quantum entanglement in simple terms.",
    #     "Write a python function that calculates the factorial of a number.",
    #     "Summarize the main plot points of the movie 'Inception'.",
    #     "What are the main differences between renewable and non-renewable energy sources?"
    # ]
    login(token=os.environ.get("HF_TOKEN"))
    prompt = "Translate the following sentence to German: 'The weather is beautiful today.'"
    tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.2-1B-Instruct", trust_remote_code=True, use_fast=True)
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_new_tokens=256)
    llm = LLM(model="models/Llama-3.2-1B-Instruct_TensorRT-LLM_Engine", tokenizer="models/Llama-3.2-1B-Instruct", trust_remote_code=True)
    start_time = timer()
    outputs = llm.generate(
        inputs=[token_ids],
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    end_time = timer()
    print(f"Time taken for generation: {end_time - start_time:.2f} seconds")
    print("-" * 50)
    # Print the outputs.
    print(outputs)

# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()