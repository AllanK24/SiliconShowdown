from timeit import default_timer as timer
from tensorrt_llm import LLM, SamplingParams

def main():
    prompts = [
        "Translate the following sentence to German: 'The weather is beautiful today.'",
        "Explain the concept of quantum entanglement in simple terms.",
        "Write a python function that calculates the factorial of a number.",
        "Summarize the main plot points of the movie 'Inception'.",
        "What are the main differences between renewable and non-renewable energy sources?"
    ]
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_new_tokens=256)
    llm = LLM(model="google/gemma-3-1b-it")
    start_time = timer()
    outputs = llm.generate(prompts, sampling_params)
    end_time = timer()
    print(f"Time taken for generation: {end_time - start_time:.2f} seconds")
    print("-" * 50)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()