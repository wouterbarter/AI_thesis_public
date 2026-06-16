from src.modeler import Modeler
from src.utils import generate_in_batches, generate_stream
from src.prompt_manager import PromptManager, PromptTemplate, PromptSuite
import torch.nn.functional as F
from src.results import ResultsContainer  # Import the new class
from pathlib import Path
import pandas as pd
import torch
import hashlib
from tqdm import tqdm
from typing import Optional


def run_experiment(df: pd.DataFrame,
                   modeler: Modeler,
                   suite: PromptSuite,
                   output_dir: Path,
                   file_stem: str,
                   model_name: str,
                   batch_size: int,
                   id_col: str = 'input_id',
                   top_k: int = 1000,
                   assistant_prefix: str = "",
                   #    shards_per_save: int = 10,
                   items_per_shard: int = 100,
                   max_new_tokens: int = 1,
                   **kwargs):

    print('Running generation function...')
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'gemma' in model_name:
        print("Reducing batch size for Gemma to avoid garbage output")
        batch_size = 1

    # processed_keys: Tuple(input_id, dimension_name)
    shard_index, processed_keys = ResultsContainer.get_experiment_state(
        output_dir)

    # We do NOT filter df here anymore! Pass the whole thing.
    print(f"Experiment ID: {output_dir}")
    print(
        f"Streaming {len(df)} IDs to check against {len(processed_keys)} completed keys...")

    # --- 1. DYNAMIC BATCH OPTIMIZATION ---
    # Ask the suite which columns it uses to build its prompts
    suite_vars = list(suite.all_required_variables)

    # We create a working copy so we don't mutate the global df
    # and to avoid Pandas SettingWithCopyWarnings
    working_df = df.copy()

    # Find how many templates are in the suite (adjust attribute name if needed based on your PromptSuite class)
    num_templates = len(suite.templates)
    total_items = len(working_df) * num_templates - len(processed_keys)

    dimension_names = suite.dimensions

    # 2. Build the "Target State" (Everything we WANT to exist)
    # Using a set comprehension for blazing fast O(1) mathematical operations
    required_keys = {
        (str(row_id), dim)
        for row_id in working_df[id_col]
        for dim in dimension_names
    }

    # 3. Find exactly what is missing (Required - Completed)
    remaining_keys = required_keys - processed_keys
    total_items = len(remaining_keys)

    if total_items == 0:
        print("✅ All items have already been processed! Skipping generation.")
        return  # Exit the function cleanly

    # processed_ids = set(x[0] for x in processed_keys)
    # n_new_ids = processed_ids - set(working_df[id_col])
    # if len(n_new_ids) > 0:
    #     print(f"{n_new_ids} new IDs found.")
    # elif total_items <= 0:
    #     print("✅ All items have already been processed! Skipping generation.")
    #     return  # Exit the function cleanly

    if suite_vars:
        print(f"📉 Optimizing batch padding based on variables: {suite_vars}")

        # 1. Get the length of EACH variable separately
        # .applymap (or .map in newer pandas) gets the len of each cell
        len_df = working_df[suite_vars].astype(str).map(len)

        # 2. Find the MAX length across the variables for each row
        working_df['_max_len'] = len_df.max(axis=1)

        # 3. Sort by the heaviest variable in that row
        working_df = working_df.sort_values('_max_len').reset_index(drop=True)
        working_df = working_df.drop(columns=['_max_len'])

    # 1. Create the Stream
    prompt_stream = suite.stream_render(
        working_df, id_col, assistant_prefix, processed_keys)

    # 2. Create the Processor
    # generate_stream calls modeler on list[PreparedPrompt]
    result_stream = generate_stream(
        modeler=modeler,
        prompt_iterator=prompt_stream,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        **kwargs
    )

    buffer = []

    print(f"BATCH SIZE: {batch_size}")
    total_batches = (total_items + batch_size - 1) // batch_size

    print(f"Resuming at Shard {shard_index}. Starting streaming inference...")
    print(
        f"Total items to process: {total_items} across {total_batches} batches.")

    # 3. Consume the Stream
    print("Starting streaming inference...")
    for batch_results in tqdm(result_stream, total=total_batches, desc="Evaluating Suite", dynamic_ncols=True):
        buffer.extend(batch_results)

        if len(buffer) >= items_per_shard:
            filename = f"{file_stem}_part_{shard_index:04d}.pt"
            output_path = output_dir / filename

            rc = ResultsContainer.from_model_outputs(model_name, top_k, buffer)
            rc.save(output_path)

            tqdm.write(f"💾 Saved checkpoint: {filename} ({len(buffer)} items)")
            shard_index += 1
            buffer = []  # Clear memory immediately!

    # 5. Save remaining items
    if buffer:
        filename = f"{file_stem}_part_{shard_index:04d}.pt"
        output_path = output_dir/filename

        rc = ResultsContainer.from_model_outputs(model_name, top_k, buffer)
        rc.save(output_path)
        print(f"Saved final shard: {filename}")
