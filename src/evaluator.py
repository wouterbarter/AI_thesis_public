import torch
import pandas as pd
from tqdm import trange

def evaluate_prompts(modeler, prompts: dict, ratings_df, cols, batch_size=2, start=0):
    """
    Run batch inference for multiple prompts over a dataset and return long-format DataFrame.
    
    Args:
        modeler: instance of Modeler (handles generate + get_relevant_logits)
        prompts: dict of {prompt_id: prompt_template}
        ratings_df: DataFrame containing the data
        cols: list of columns to render into the template
        batch_size: number of rows per batch
        start: index offset (useful for resuming)
    """
    all_logits = {pid: {} for pid in prompts}

    for i in trange(start, len(ratings_df), batch_size, desc="Evaluating prompts"):
        batch = ratings_df.iloc[i:i + batch_size]
        for prompt_id, prompt_template in prompts.items():
            rendered = prompt_template.render_many(batch[cols])
            modeler.generate(rendered)
            logits = modeler.get_relevant_logits()
            all_logits[prompt_id].update(
                {id_: logit for id_, logit in zip(batch['id'], logits)}
            )

    # Flatten to long format
    rows = []
    for prompt_id, row_dict in all_logits.items():
        for row_id, logits in row_dict.items():
            rows.append({
                "prompt_id": prompt_id,
                "id": row_id,
                "logits": logits,
                "rating": int(torch.argmax(logits))
            })
    return pd.DataFrame(rows)