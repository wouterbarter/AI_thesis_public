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
                   shards_per_save: int = 10,
                   max_new_tokens: int = 1):

    print('running generation function...')

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_index, processed_ids = ResultsContainer.get_experiment_state(output_dir)

    df_new = df[~df[id_col].isin(processed_ids)]

    if len(df_new) == 0:
        print("Experiment complete. No new IDs to process.")
        return

    print(f"Experiment ID: {output_dir}")
    print(f"Processing {len(df_new)} new IDs...")

    # 1. Create the Stream
    prompt_stream = suite.stream_render(df_new, id_col, assistant_prefix)
    
    # 2. Create the Processor
    ## generate_stream calls modeler on list[PreparedPrompt]
    result_stream = generate_stream(
        modeler=modeler, 
        prompt_iterator=prompt_stream, 
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        top_k=top_k
    )
    
    buffer = []
    total_batches = (len(df_new) // batch_size) + 1

    print(f"Resuming at Shard {shard_index}. Starting streaming inference...")

    # 3. Consume the Stream
    print("Starting streaming inference...")
    for batch_results in tqdm(result_stream, total=total_batches):
        buffer.extend(batch_results)
        
        # 4. Checkpoint Strategy
        # Save every N batches (e.g., every 10 batches = 80 items)
        if len(buffer) >= (batch_size * shards_per_save):
            filename = f"{file_stem}_part_{shard_index:04d}.pt"
            output_path = output_dir/filename

            rc = ResultsContainer.from_model_outputs(model_name, top_k, buffer)
            rc.save(output_path)

            print(f" Saved checkpoint: {filename}") # Helpful logging
            shard_index += 1
            buffer = [] # Clear memory immediately!

    # 5. Save remaining items
    if buffer:
        filename = f"{file_stem}_part_{shard_index:04d}.pt"
        output_path = output_dir/filename
        
        rc = ResultsContainer.from_model_outputs(model_name, top_k, buffer)
        rc.save(output_path)
        print(f"Saved final shard: {filename}")




