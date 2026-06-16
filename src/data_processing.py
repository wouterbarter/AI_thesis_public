# from pathlib import Path
from typing import Dict
import pandas as pd
from src.data_manager import DataManager
import torch
from typing import List, Optional
from src import paths
from pandas.util import hash_pandas_object
import os
from pathlib import Path
from src.general_analysis.metrics import compute_entropy, calculate_reliability_gap, compute_validity_mass
import numpy as np
import yaml

# TODO: Make sure that the Suite Metadata is also included in the final_df
# TODO: I use hashing to generate input_id, but that might lead to duplicate hashes in downstream processing which could be removed in processing


# def load_input_data(path_to_parquet: str | Path):
#     return pd.read_parquet(path_to_parquet)


def load_llm_results_data(path_to_llm_results: Path, allowed_prompt_ids: Optional[list] = None, allowed_models: Optional[list] = None) -> pd.DataFrame:
    dm = DataManager(path_to_llm_results)
    dm.load_all(allowed_prompt_ids=allowed_prompt_ids,
                allowed_models=allowed_models)
    analysis_df = dm.create_analysis_dataframe()
    return analysis_df


def create_sorted_logits(df: pd.DataFrame, label_order: Optional[List[str]] = None) -> pd.DataFrame:
    # Get the raw list of tuples
    raw_results = df.apply(_reorder_logits_row, axis=1,
                           label_order=label_order).tolist()

    # Instantly convert the list of tuples into a DataFrame
    sorted_df = pd.DataFrame(raw_results, columns=[
                             'sorted_tokens', 'sorted_logits'], index=df.index)
    return sorted_df


def _reorder_logits_row(
    row: pd.Series,
    label_order: Optional[List[str]] = None
) -> tuple:
    """
    Helper function to sort token-logit pairs for a single row.
    """
    tokens = row['constrained_tokens']
    logits = row['constrained_logits']

    if hasattr(logits, 'tolist'):
        logits = logits.tolist()

    if not tokens or not logits:
        return pd.Series([[], []], index=['sorted_tokens', 'sorted_logits'])

    if label_order:
        # Custom order: need index-based approach
        order_map = {token: i for i, token in enumerate(label_order)}
        sort_indices = sorted(
            range(len(tokens)),
            key=lambda i: order_map.get(tokens[i], float('inf'))
        )
        sorted_tokens = [tokens[i] for i in sort_indices]
        sorted_logits = [[logit_list[i] for i in sort_indices]
                         for logit_list in logits]
    else:
        # Natural sort: optimize with direct sorting
        sort_indices = sorted(range(len(tokens)), key=lambda i: tokens[i])
        sorted_tokens = sorted(tokens)  # Faster than list comprehension
        sorted_logits = [[logit_list[i] for i in sort_indices]
                         for logit_list in logits]

    # return pd.Series(
    #     [sorted_tokens, sorted_logits],
    #     index=['sorted_tokens', 'sorted_logits']
    # )

    return sorted_tokens, sorted_logits


# Data cleaning

def remove_garbage_rows(df: pd.DataFrame,
                        input_seq_col: str = 'input_sequence',  # Your HF sequence column
                        data_col: str = 'top_k_tokens'):
    '''
    Identifies and removes "garbage" rows where the LLM outputs the exact same 
    generic distribution (priors) for completely different input sequences.
    '''
    helper = pd.DataFrame()

    # --- 1. Hash the Ground Truth Input ---
    # If your input sequence is saved as a list of Token IDs:
    if isinstance(df[input_seq_col].iloc[0], (list, np.ndarray)):
        helper['input_hash'] = df[input_seq_col].apply(tuple).apply(hash)
    # If your input sequence is saved as a decoded string:
    else:
        helper['input_hash'] = hash_pandas_object(
            df[input_seq_col], index=False)

    # --- 2. Hash the Output ---
    helper['hashable_output'] = df[data_col].apply(lambda x: tuple(x[0]))
    helper['model_name'] = df['model_name']

    # --- 3. Detect Garbage ---
    # Count how many DIFFERENT inputs resulted in the EXACT SAME output
    unique_inputs_per_output = helper.groupby(['model_name', 'hashable_output'])[
        'input_hash'].transform('nunique')

    is_garbage_mask = unique_inputs_per_output > 1

    if not is_garbage_mask.any():
        print("✅ No garbage detected. Skipping merge and returning original data.")
        # Return full clean_df, empty dirty_df
        return df.copy(), df.iloc[0:0].copy()

    # --- 4. Create the Blocklist & Filter ---
    trial_cols = ['model_name', 'input_id', 'prompt_id', 'assistant_prefix']
    tainted_trials = df.loc[is_garbage_mask, trial_cols].drop_duplicates()

    print(
        f"Found {len(tainted_trials)} experimental trials contaminated by garbage output.")

    merged = df.merge(
        tainted_trials,
        on=trial_cols,
        how='left',
        indicator=True
    )

    clean_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    dirty_df = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])

    return clean_df, dirty_df


def get_balanced_intersection(df: pd.DataFrame,
                              input_id_col: str,
                              experimental_groups: list[str],
                              model_col: str = 'model_name') -> pd.DataFrame:
    """
    Returns a subset of df where every experimental trial (defined by key_cols)
    is present for ALL models found in the dataset.
    """

    key_cols = [input_id_col] + \
        [col for col in experimental_groups if col != model_col]

    # 1. Identify all unique models in the current clean data
    required_models = df['model_name'].unique()
    n_models = len(required_models)

    print(f"Balancing data across {n_models} models: {required_models}")

    # 2. Count how many models successfully completed each trial
    # We group by the trial keys (input_id, prompt_id, etc) and count unique models
    trial_counts = df.groupby(key_cols)['model_name'].nunique()

    # 3. Identify trials that have a count equal to the total number of models
    # These are the "Complete" trials
    valid_trials = trial_counts[trial_counts == n_models].index

    # 4. Filter the original dataframe to keep only these complete trials
    # We use .isin() on the index if it's a single level, but for multi-col keys
    # it's often cleaner to merge or join.

    # Let's make the keys a proper index on the main df for fast joining
    df_indexed = df.set_index(key_cols)

    # Intersection
    balanced_df = df_indexed.loc[valid_trials].reset_index()

    print(f"Original rows: {len(df)} -> Balanced rows: {len(balanced_df)}")

    return balanced_df


def compute_ratings_from_logits(
    df: pd.DataFrame,
    weights: Optional[Dict[int, List[float]]] = None,
    weights_tensor: Optional[Dict[int, torch.Tensor]] = None,
    sequence_index: int = 0
) -> pd.DataFrame:
    """
    Calculates mean and mode ratings from a DataFrame column of logits tensors.
    Automatically handles mixed scale sizes by grouping and processing separately.
    Extracts logits from a specific position in the sequence.

    Args:
        df: The DataFrame containing the logit data.
        weights: A dictionary mapping scale_size -> weights list.
                 E.g., {4: [1, 2, 3, 4], 5: [1, 2, 3, 4, 5]}
                 If None, defaults to [1, 2, ..., N] for each scale size.
        weights_tensor: Alternative to weights - provide pre-computed tensors.
        sequence_index: Which position in the sequence to extract (default: 0, first position).

    Returns:
        A new DataFrame with 'mean_rating', 'mode_rating', and 'scale_size' columns,
        preserving the original index order.
    """

    # --- 1. Determine scale size for each row ---
    df = df.copy()
    df['_scale_size'] = df['sorted_logits'].apply(
        lambda x: len(x[sequence_index]))

    # Get unique scale sizes
    scale_sizes = df['_scale_size'].unique()

    print(
        f"Found {len(scale_sizes)} different scale sizes: {sorted(scale_sizes)}")

    # --- 2. Prepare weights for each scale size ---
    weights_dict = {}

    if weights_tensor is not None:
        # Use provided tensors
        weights_dict = weights_tensor
    elif weights is not None:
        # Convert provided weights to tensors
        for scale_size, weight_list in weights.items():
            if len(weight_list) != scale_size:
                raise ValueError(
                    f"Weights for scale size {scale_size} has {len(weight_list)} elements, "
                    f"expected {scale_size}"
                )
            weights_dict[scale_size] = torch.tensor(
                weight_list, dtype=torch.float32)
    else:
        # Generate default weights for each scale size
        for scale_size in scale_sizes:
            weights_dict[scale_size] = torch.arange(
                1, scale_size + 1, dtype=torch.float32)

    # --- 3. Process each scale size group ---
    results_list = []

    for scale_size in scale_sizes:
        # Get subset for this scale size
        mask = df['_scale_size'] == scale_size
        df_subset = df[mask]

        if len(df_subset) == 0:
            continue

        # Get weights for this scale size
        if scale_size not in weights_dict:
            # Generate default if not provided
            weights_dict[scale_size] = torch.arange(
                1, scale_size + 1, dtype=torch.float32)

        current_weights = weights_dict[scale_size]

        # Extract logits from the sequence position and stack into tensor
        # Each x is a tensor of shape [seq_len, num_classes]
        # We extract x[sequence_index] to get [num_classes]
        logits_list = df_subset['sorted_logits'].apply(
            lambda x: x[sequence_index].tolist() if torch.is_tensor(
                x) else x[sequence_index]
        ).tolist()

        logits_tensor = torch.tensor(logits_list, dtype=torch.float32)

        # Validate shape
        if logits_tensor.shape[1] != scale_size:
            raise ValueError(
                f"Logits shape mismatch for scale_size {scale_size}: "
                f"expected {scale_size}, got {logits_tensor.shape[1]}"
            )

        # Call the core logic function
        mean_rating_tensor, mode_rating_tensor = _compute_ratings_from_tensors(
            logits_tensor, current_weights
        )

        # Safety check
        batch_size = len(df_subset)
        if mean_rating_tensor.numel() != batch_size:
            raise ValueError(
                f"Shape Mismatch for scale_size {scale_size}! "
                f"DataFrame subset has {batch_size} rows, but "
                f"mean_rating_tensor has {mean_rating_tensor.numel()} elements "
                f"(Shape: {mean_rating_tensor.shape}). "
                "Did you forget to reduce a dimension?"
            )

        # Create results DataFrame for this group
        group_results = pd.DataFrame({
            'mean_rating': mean_rating_tensor.flatten().tolist(),
            'mode_rating': mode_rating_tensor.flatten().tolist(),
            'scale_size': scale_size
        }, index=df_subset.index)

        results_list.append(group_results)

        print(
            f"  Processed {len(df_subset)} rows with scale_size={scale_size}")

    # --- 4. Combine all results and restore original order ---
    if not results_list:
        raise ValueError("No valid data to process")

    results_df = pd.concat(results_list)

    # Restore original row order
    results_df = results_df.loc[df.index]

    return results_df


def _compute_ratings_from_tensors(
    logits_tensor: torch.Tensor,
    weights_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core logic: Calculates mean and mode ratings from tensors.

    Returns:
        A tuple of (mean_rating_tensor, mode_rating_tensor)
    """
    softmax_logits = torch.softmax(logits_tensor, dim=-1)

    # Calculate mean rating (weighted average)
    mean_rating = softmax_logits @ weights_tensor

    # Calculate mode rating
    mode_rating_index = softmax_logits.argmax(dim=-1)
    mode_rating_label = weights_tensor[mode_rating_index]

    return mean_rating, mode_rating_label


def compute_entropy_from_logits(
    df: pd.DataFrame,
    sequence_index: int = 0,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Computes entropy from logits, handling mixed scale sizes.

    Args:
        df: DataFrame with 'sorted_logits' column containing tensors
        sequence_index: Which position in the sequence to use
        normalize: Whether to normalize entropy by log(scale_size)

    Returns:
        DataFrame with 'entropy', 'normalized_entropy', and 'scale_size' columns
    """
    df = df.copy()

    # Determine scale size for each row
    df['_scale_size'] = df['sorted_logits'].apply(
        lambda x: len(x[sequence_index]))

    scale_sizes = df['_scale_size'].unique()
    print(
        f"Computing entropy for {len(scale_sizes)} different scale sizes: {sorted(scale_sizes)}")

    results_list = []

    for scale_size in scale_sizes:
        # Get subset for this scale size
        mask = df['_scale_size'] == scale_size
        df_subset = df[mask]

        if len(df_subset) == 0:
            continue

        # Extract logits from the sequence position
        logits_list = df_subset['sorted_logits'].apply(
            lambda x: x[sequence_index].tolist() if torch.is_tensor(
                x) else x[sequence_index]
        ).tolist()

        # Convert to tensor and apply softmax
        logits_tensor = torch.tensor(logits_list, dtype=torch.float32)
        softmax_probs = torch.softmax(logits_tensor, dim=-1)

        # Compute entropy
        entropy_values = compute_entropy(softmax_probs)

        # Create results for this group
        group_results = pd.DataFrame({
            'entropy': entropy_values,
            'scale_size': scale_size
        }, index=df_subset.index)

        # Add normalized entropy if requested
        if normalize:
            group_results['normalized_entropy'] = entropy_values / \
                np.log(scale_size)

        results_list.append(group_results)
        print(
            f"  Processed {len(df_subset)} rows with scale_size={scale_size}")

    # Combine and restore original order
    results_df = pd.concat(results_list)
    results_df = results_df.loc[df.index]

    return results_df


# def get_input_data_path(results_dir):
#     with open(results_dir / 'experiment_config.yaml') as f:
#         config = yaml.safe_load(f)
#     input_filename = config['analysis_config']['input_filenames']['processed_data_filename']
#     input_data_path = paths.PROCESSED_DATA_DIR / f"{input_filename}.parquet"
#     return input_data_path


def z_standardize_rating_cols(ratings_df):

    print("--- 🚀 STANDARDIZING MEASUREMENTS FOR ECONOMETRIC COMPARISON ---")

    # 1. Apply the Z-score transformation using groupby and transform
    # We use ddof=1 for the sample standard deviation
    ratings_df['z_mean_rating'] = ratings_df.groupby(
        ['model_name', 'prompt_id', 'dimension_name']
    )['mean_rating'].transform(lambda x: (x - x.mean()) / x.std(ddof=1))

    ratings_df['z_mode_rating'] = ratings_df.groupby(
        ['model_name', 'prompt_id', 'dimension_name']
    )['mode_rating'].transform(lambda x: (x - x.mean()) / x.std(ddof=1))

    return ratings_df


def get_analysis_ready_df(analysis_config: dict,
                          #   full_config: dict,
                          active_analysis: Optional[str] = None,
                          use_cache: bool = False,
                          force_refresh: bool = False,
                          return_dirty_df: bool = False,
                          balance_experimental_trials: bool = False
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:

    # active_analysis_name = active_analysis if active_analysis is not None else full_config[
    #     'active_analysis']

    # analysis_config = full_config['analyses'][active_analysis_name]

    print(f"Loading files for analysis {analysis_config.get('name')}")
    # analysis_name = active_analysis_name.upper()

    results_dir = analysis_config['metadata']['results_dir']

    # if not full_config.get('sandbox_mode'):
    #     results_dir = paths.RESULTS_DIR / analysis_name
    # else:
    #     results_dir = paths.RESULTS_DIR / 'sandbox' / analysis_name

    cache_filename = f"{results_dir.stem}_analysis_ready.pkl"
    cache_path = results_dir / cache_filename

    # 2. Check Cache
    if use_cache and not force_refresh and os.path.exists(cache_path):
        print(f"⚡ Loading cached DataFrame from {cache_path}...")
        try:
            final_df = pd.read_pickle(cache_path)
            input_data_path = analysis_config['metadata']['input_data_path']
            input_df = pd.read_parquet(input_data_path)
            return final_df, input_df
        except Exception as e:
            print(
                f"⚠️ Cache file corrupted or incompatible. Re-running pipeline. Error: {e}")

    print("🐢 Running full processing pipeline...")

    # analysis_config = full_config['analyses'][active_analysis_name]
    id_col = analysis_config['keys']['raw_id_col']
    experimental_groups = analysis_config['model_vars']['experimental_groups']
    evaluations_id_col = analysis_config['keys']['evaluations_id_col']
    # input_variable_names = analysis_config['variable_names']

    allowed_prompt_ids = analysis_config['data_loading']['prompts']['ids_to_include']
    allowed_models = analysis_config['data_loading']['prompts']['models_to_include']

    # ---- Specific config vars
    # TODO: Just sorts when label_order=None, but I will need to customize to test for positional bias
    # I will need to extract it from the prompt template if I want to implement this
    label_order = analysis_config['model_vars'].get('label_order', None)

    print(f"Results dir: {results_dir}")
    # input_data_path = get_input_data_path(results_dir)
    input_data_path = analysis_config['metadata']['input_data_path']

    print(f"input_data_path: {input_data_path}")
    input_df = pd.read_parquet(input_data_path)
    print("Loading evaluations...")
    evaluations_df = load_llm_results_data(
        results_dir, allowed_prompt_ids=allowed_prompt_ids, allowed_models=allowed_models)

    # Instead of trusting the config, ask the dataframe what top_k it actually used.
    if 'top_k' in evaluations_df.columns:
        unique_ks = evaluations_df['top_k'].unique()
        print(f"📊 Loaded evaluations with top_k values: {unique_ks}")

        # Optional: Warn if you accidentally mixed top_500 and top_1000 runs in the same folder
        if len(unique_ks) > 1:
            print(
                f"⚠️ WARNING: Multiple top_k settings detected in this folder: {unique_ks}")
    else:
        print("⚠️ WARNING: 'top_k' metadata column missing from evaluations_df.")

    print("Finished loading experiment data")

    # ----- Processing
    # Combine
    print("Merging input data and evaluations")
    merged_df = pd.merge(input_df, evaluations_df,
                         left_on=id_col, right_on=evaluations_id_col,
                         how='left',             # Keep all processed rows
                         indicator='_merge_status'  # Track success
                         )

    # Explicitly check for missing evaluations
    missing = merged_df[merged_df['_merge_status'] == 'left_only']
    if not missing.empty:
        print(
            f"⚠️ Warning: {len(missing)} rows from input data are missing LLM results.")
        # Optional: Drop them if you can't analyze them
        merged_df = merged_df[merged_df['_merge_status'] == 'both'].copy()
    print("Finished merging.")

    # Clean
    clean_df, dirty_df = remove_garbage_rows(
        df=merged_df,
        input_seq_col='formatted_prompts',
        data_col='top_k_tokens')

    if balance_experimental_trials:
        balanced_df = get_balanced_intersection(
            clean_df, id_col, experimental_groups, model_col='model_name')
    else:
        balanced_df = clean_df

    # Features
    print("Sorting logits...")
    sorted_logits_df = create_sorted_logits(balanced_df, label_order)
    print("Finished sorting logits.")
    print("Computing ratings from logits...")
    ratings_df = compute_ratings_from_logits(
        sorted_logits_df, sequence_index=0)
    print("Finished computing ratings.")

    # # Assemble
    # final_df = pd.concat([balanced_df, sorted_logits_df, ratings_df], axis=1)

    # Assign directly to avoid index mismatch risks
    balanced_df.loc[:, sorted_logits_df.columns] = sorted_logits_df
    balanced_df.loc[:, ratings_df.columns] = ratings_df
    final_df = balanced_df  # Rename for clarity

    # Features - Entropy (handles mixed scale sizes)
    entropy_df = compute_entropy_from_logits(
        final_df, sequence_index=0, normalize=True)
    final_df['entropy'] = entropy_df['entropy']
    final_df['normalized_entropy'] = entropy_df['normalized_entropy']
    # Note: scale_size is already added by compute_ratings_from_logits

    # Features - Apply z-standardization

    final_df = z_standardize_rating_cols(final_df)

    # TODO: MOVE INTO FEATURES! SHOULD BE PART OF THE PROCESSED FILE!!!
    # Dataset-specific pre-processing steps
    if active_analysis == 'mcgill_qa_feedback':
        # Mean score
        final_df['mean_human_rating'] = (
            final_df['score_1'] + final_df['score_2']) / 2

        # Disagreement
        final_df['human_disagreement'] = abs(
            final_df['score_1'] - final_df['score_2'])

    elif active_analysis == 'barter_deals':
        final_df = final_df[~final_df.content_types.isna()]
        final_df.loc[:, 'first_content_type'] = final_df.content_types.apply(
            lambda x: x[0]['name'])

    final_df = final_df.drop_duplicates(
        subset=experimental_groups + [evaluations_id_col], keep='first')

    if use_cache:
        # Ensure directory exists
        os.makedirs(cache_path.parent, exist_ok=True)
        print(f"💾 Saving result to {cache_path}...")
        final_df.to_pickle(cache_path, protocol=5)

    if return_dirty_df:
        return final_df, dirty_df

    return final_df, input_df
