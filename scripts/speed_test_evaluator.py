# Speed test comparisons for prompt batches


import pandas as pd
import torch

# import sys
# sys.path.append("src")

# from modeler import Modeler
# from prompt_manager import PromptManager, Prompt
# from evaluator import evaluate_prompts
# from utils import generate_in_batches

# This is how your main script's imports should look
from src.modeler import Modeler
from src.prompt_manager import PromptManager, Prompt
from src.evaluator import evaluate_prompts
from src.utils import generate_in_batches

### Load data
from datasets import load_dataset

data_files = {
    "train": "https://huggingface.co/datasets/McGill-NLP/feedbackQA/resolve/main/data/feedback_train.json",
    "validation": "https://huggingface.co/datasets/McGill-NLP/feedbackQA/resolve/main/data/feedback_valid.json",
    "test": "https://huggingface.co/datasets/McGill-NLP/feedbackQA/resolve/main/data/feedback_test.json"
}

ds = load_dataset("json", data_files=data_files)
ratings = pd.DataFrame(ds["train"])

ratings["review_1"] = ratings["rating"].apply(lambda x: x[0])
ratings["explanation_1"] = ratings["feedback"].apply(lambda x: x[0])
ratings["review_2"] = ratings["rating"].apply(lambda x: x[1])
ratings["explanation_2"] = ratings["feedback"].apply(lambda x: x[1])

# # Map scores to numeric values
# conversion_dict = {"Excellent": 4, "Acceptable": 3, "Could be Improved": 2, "Bad": 1}
# ratings["score_1"] = ratings["review_1"].map(conversion_dict)
# ratings["score_2"] = ratings["review_2"].map(conversion_dict)


ratings['answer'] = ratings['passage'].apply(lambda x: x['reference']['section_content'])

ratings['id'] = ratings.index.values


import torch
from typing import Callable, Any, Dict

def benchmark_gpu_operation(
    operation: Callable[[], Any],
    n_items: int,
    num_repetitions: int = 20,
    warmup_runs: int = 3,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Accurately benchmarks a PyTorch GPU operation using CUDA events.

    Args:
        operation (Callable[[], Any]): A zero-argument function that executes the GPU work to be timed.
                                       Use a lambda to wrap your function call, e.g., 
                                       lambda: my_model(my_input).
        batch_size (int): The number of items in the batch, used for calculating throughput.
        num_repetitions (int): The number of times to run the operation for timing.
        warmup_runs (int): The number of warm-up runs to perform before timing.
        verbose (bool): If True, prints the results to the console.

    Returns:
        Dict[str, float]: A dictionary containing the average latency and throughput.
    """
    # --- 1. Warm-up ---
    # This handles initial CUDA kernel compilations and setup overhead.
    if verbose:
        print("Warming up GPU...")
    for _ in range(warmup_runs):
        _ = operation()
        torch.cuda.synchronize()

    # --- 2. Timed Execution ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings_ms = []

    if verbose:
        print(f"Starting benchmark with {num_repetitions} repetitions...")
    
    for _ in range(num_repetitions):
        start_event.record()
        _ = operation() # Execute the provided function
        end_event.record()
        
        # VERY IMPORTANT: Wait for the GPU to finish the work before reading the time.
        torch.cuda.synchronize()
        
        timings_ms.append(start_event.elapsed_time(end_event))

    # --- 3. Calculate and Report Results ---
    avg_latency_ms = sum(timings_ms) / len(timings_ms)
    throughput_prompts_sec = n_items / (avg_latency_ms / 1000)

    if verbose:
        print("\n--- Benchmark Results ---")
        print(f"Batch size:               {batch_size}")
        print(f"Average time per batch:   {avg_latency_ms:.2f} ms")
        print(f"Throughput:               {throughput_prompts_sec:.2f} prompts/sec")
    
    return {
        "avg_latency_ms": avg_latency_ms,
        "throughput_prompts_sec": throughput_prompts_sec
    }




### Load prompts

pm = PromptManager(folder = "prompts")
prompts = pm.load_all()

### Instantiate models
model_name = "Qwen/Qwen3-0.6B"
prompt_id = '354005a603'
cols = ["question", "answer"]


modeler = Modeler(model_name)
modeler.set_token_constraints(list("12345"))


# # --- Example 1: Timing your get_evaluation_logits_with_cache function ---

### Prompt cache method


# print("--- Benchmarking Logits Evaluation ---")


## Prepare prompt prefix/dynamic template

# split_index_dict = {prompt_id: prompt.template.find('{') for prompt_id, prompt in prompts.items()}
# split_index = split_index_dict[prompt_id]
# static_prefix = prompts[prompt_id].template[:split_index]
# dynamic_template = prompts[prompt_id].template[split_index:]

# # Subset 
# test_prompts_dict = ratings.loc[:50, cols].to_dict(orient='records')
# test_prompts = [dynamic_template.format(**entry) for entry in test_prompts_dict][:batch_size]

# # 
# modeler.prepare_prompt_cache(static_prefix) 


# logits_results = benchmark_gpu_operation(
#     operation=lambda: modeler.get_evaluation_logits_with_cache(test_prompts),
#     batch_size=batch_size,
#     num_repetitions=num_reps # Use a higher number for stable results
# )


# --- Example 2: Timing a different function, like model.generate ---
num_reps = 5

n_batches = 8
batch_size = 4
n_items = batch_size*n_batches
test_prompts = prompts[prompt_id].render_many(ratings[:n_items])


res = generate_in_batches(modeler, test_prompts, batch_size)

generation_results = benchmark_gpu_operation(
    operation=lambda: generate_in_batches(modeler, test_prompts, batch_size),
    n_items=n_items,
    num_repetitions=num_reps # Generation is slower, so fewer reps are ok
)

print(f"Generation throughput: {generation_results['throughput_prompts_sec']:.2f}")



# generation_results = benchmark_gpu_operation(
#     operation=lambda: modeler.generate(test_prompts, max_new_tokens=1),
#     batch_size=batch_size,
#     num_repetitions=num_reps # Generation is slower, so fewer reps are ok
# )

# print("\n\n--- Benchmarking Full Generation ---")
# generation_results = benchmark_gpu_operation(
#     operation=lambda: modeler.generate(test_prompts, max_new_tokens=1),
#     batch_size=batch_size,
#     num_repetitions=num_reps # Generation is slower, so fewer reps are ok
# )

# print(f"Generation throughput: {generation_results['throughput_prompts_sec']:.2f}")





# You can now use the returned dictionaries for further analysis
# print(f"\nLogits throughput: {logits_results['throughput_prompts_sec']:.2f}")
# print(f"Generation throughput: {generation_results['throughput_prompts_sec']:.2f}")


















# # 1. Perform a warm-up run to handle initial CUDA overhead
# print("Warming up...")
# _ = modeler.get_evaluation_logits_with_cache(test_prompts)
# torch.cuda.synchronize() # Wait for the warm-up run to actually finish

# print("Timing...")

# # 2. Use torch.cuda.Event for accurate GPU timing
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# num_repetitions = 5 # Use a higher number for more stable results
# timings = []

# for _ in range(num_repetitions):
#     start_event.record() # Start the timer

#     # The operation you want to measure
#     eval_logits = modeler.get_evaluation_logits_with_cache(test_prompts)

#     end_event.record() # Stop the timer

#     # 3. Synchronize and calculate the elapsed time
#     torch.cuda.synchronize() # IMPORTANT: Wait for the GPU to finish the work
    
#     # elapsed_time_ms returns time in milliseconds
#     elapsed_time = start_event.elapsed_time(end_event)
#     timings.append(elapsed_time)

# # Calculate and print the results
# avg_time_ms = sum(timings) / len(timings)
# throughput = batch_size / (avg_time_ms / 1000) # prompts per second

# print(f"Batch size: {batch_size}")
# print(f"Average time per batch: {avg_time_ms:.2f} ms")
# print(f"Throughput: {throughput:.2f} prompts/sec")





# import torch
# import pandas as pd
# import time

# # --- Your existing setup code ---
# # ... (load modeler, prompts, data, etc.)
# # static_prefix = ...
# # dynamic_template = ...
# # modeler.prepare_prompt_cache(static_prefix)
# # ...

# # --- NEW BENCHMARKING SCRIPT ---

# # Define the batch sizes you want to test
# batch_sizes_to_test = [4, 8, 16, 24, 32, 36, 48, 64, 96, 128] 
# results = []
# num_repetitions = 20 # Use a decent number for stable timings

# # Prepare a large pool of prompts to draw from
# full_test_prompts_dict = ratings.loc[:200, cols].to_dict(orient='records')
# full_test_prompts = [dynamic_template.format(**entry) for entry in full_test_prompts_dict]


# print("Starting benchmark...")
# for batch_size in batch_sizes_to_test:
#     # Ensure we don't test a batch size larger than our sample data
#     if batch_size > len(full_test_prompts):
#         continue

#     # Prepare a batch of the current size
#     test_prompts_batch = full_test_prompts[:batch_size]
    
#     try:
#         # --- Warm-up run for the current batch size ---
#         _ = modeler.get_evaluation_logits_with_cache(test_prompts_batch)
#         torch.cuda.synchronize()

#         # --- Accurate GPU Timing ---
#         start_event = torch.cuda.Event(enable_timing=True)
#         end_event = torch.cuda.Event(enable_timing=True)
        
#         timings_ms = []
#         for _ in range(num_repetitions):
#             start_event.record()
#             _ = modeler.get_evaluation_logits_with_cache(test_prompts_batch)
#             end_event.record()
#             torch.cuda.synchronize()
#             timings_ms.append(start_event.elapsed_time(end_event))

#         avg_latency_ms = sum(timings_ms) / len(timings_ms)
#         throughput = batch_size / (avg_latency_ms / 1000)

#         # --- Measure VRAM ---
#         # Note: For accurate peak memory, it's best to run this in a clean script.
#         # This gives the memory used by the last run.
#         peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
#         print(f"Batch Size: {batch_size:<4} | Avg Latency: {avg_latency_ms:>8.2f} ms | Throughput: {throughput:>8.2f} prompts/sec | VRAM: {peak_vram_gb:.2f} GB")
#         results.append({
#             "batch_size": batch_size,
#             "latency_ms": avg_latency_ms,
#             "throughput_prompts_sec": throughput,
#             "peak_vram_gb": peak_vram_gb
#         })

#     except torch.cuda.OutOfMemoryError:
#         print(f"Batch Size: {batch_size:<4} | FAILED: CUDA Out of Memory")
#         break # No point in trying larger sizes

# # Display final results in a clean table
# print("\n--- Benchmark Summary ---")
# df = pd.DataFrame(results)
# print(df.to_string(index=False))









# # # ### Generate method


# # # test_prompts = prompts[prompt_id].render_many(ratings[:batch_size])
# # # res = modeler.generate(test_prompts)
