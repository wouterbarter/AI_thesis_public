# from pathlib import Path
import pandas as pd
from src.data_manager import DataManager
import torch
from typing import List, Optional
from src import paths
from pandas.util import hash_pandas_object
import os
from pathlib import Path

#TODO: I use hashing to generate input_id, but that might lead to duplicate hashes in downstream processing which could be removed in processing

def load_input_data(path_to_parquet: str | Path):
    return pd.read_parquet(path_to_parquet)

def load_llm_results_data(path_to_llm_results: Path) -> pd.DataFrame:
    dm = DataManager(path_to_llm_results)
    dm.load_all()
    analysis_df = dm.create_analysis_dataframe()
    return analysis_df




def create_sorted_logits(
    df: pd.DataFrame, 
    label_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Sorts 'constrained_tokens' and 'constrained_logits' columns while maintaining pairing.
    
    Args:
        df: DataFrame with 'constrained_tokens' (flat list) and 
            'constrained_logits' (list of lists) columns
        label_order: Optional list defining desired sort order (e.g., ['1', '2', '3', '4'])
    
    Returns:
        DataFrame with added 'sorted_tokens' and 'sorted_logits' columns
    """
    sorted_df = df.apply(
        _reorder_logits_row,
        axis=1,
        label_order=label_order
    )
    return sorted_df

def _reorder_logits_row(
    row: pd.Series, 
    label_order: Optional[List[str]] = None
) -> pd.Series:
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
        sorted_logits = [[logit_list[i] for i in sort_indices] for logit_list in logits]
    else:
        # Natural sort: optimize with direct sorting
        sort_indices = sorted(range(len(tokens)), key=lambda i: tokens[i])
        sorted_tokens = sorted(tokens)  # Faster than list comprehension
        sorted_logits = [[logit_list[i] for i in sort_indices] for logit_list in logits]
    
    return pd.Series(
        [sorted_tokens, sorted_logits],
        index=['sorted_tokens', 'sorted_logits']
    )


# def create_sorted_logits(df: pd.DataFrame, label_order: Optional[list] = None) -> pd.DataFrame:
#     """
#     Applies logic to sort the 'constrained_tokens' and 'constrained_logits' columns.
    
#     label_order: An optional list defining the desired sort order 
#                  (e.g., ['1', '2', '3', '4', '5']).
#     """
#     print("Reordering logits...")
    
#     # The .apply() call *already* returns the new DataFrame
#     # with 'sorted_tokens' and 'sorted_logits'
#     sorted_df = df.apply(
#         _reorder_logits_row,  # (Your helper function)
#         axis=1, 
#         label_order=label_order
#     )
    
#     return sorted_df

# # --- Private Helper Function ---

# def _reorder_logits_row(row, label_order=None):
#     """
#     Helper function to sort logits for a single row.
#     """
#     tokens = row['constrained_tokens']
#     logits = row['constrained_logits']
#     if hasattr(logits, 'tolist'):
#         logits = logits.tolist()

#     zipped_pairs = list(zip(tokens, logits))

#     if not zipped_pairs:
#         return pd.Series([[], []], index=['sorted_tokens', 'sorted_logits'])

#     # Sorting logic
#     if label_order:
#         # Create a lookup map for the desired order
#         # e.g., {'1': 0, '2': 1, '3': 2, ...}
#         order_map = {token: i for i, token in enumerate(label_order)}
        
#         # Sort using the map. Use .get() for safety.
#         # Push any unknown tokens to the end.
#         sorted_pairs = sorted(
#             zipped_pairs, 
#             key=lambda pair: order_map.get(pair[0], float('inf'))
#         )
#     else:
#         # Default to alphabetical/numeric sort if no order is given 
#         sorted_pairs = sorted(zipped_pairs)

#     sorted_tokens, sorted_logits = zip(*sorted_pairs)
#     return pd.Series([list(sorted_tokens), list(sorted_logits)],
#                      index=['sorted_tokens', 'sorted_logits'])



# Data cleaning

def remove_garbage_rows(df: pd.DataFrame, input_vars: list[str], data_col: str = 'top_1000_tokens'):
    '''
    Due to some problem in Gemma (which I have found was due to dtype=float16, which I have fixed), the output is sometimes incorrect, and the output becomes the default token distribution.
    Since we do not expect the output to be the same ever (except for duplicate deals, which are infrequent), we remove the incorrect output by removing duplicates.
    TODO: this is not a clean solution, and I would have to look into the causes for the garbage output in Gemma
    TODO: Check whether deals that are true duplicates get removed as well
    TODO: Check why nr. of observations of each experimental group is not equal

    TODOs should all be fixed, but have not explicitly verified.

    1. Creates helper dataframe so original df is not modified
    2. creates hash of the input so we know which ones are equivalent
    3. Casts the data_col (top_1000_tokens) to tuple (easier duplication detection than list)
    4. groupby model_name and hashable_tokens, then count how many different texts the _same_ output was generated for
        it is impossible that the top_1000_tokens are exactly equivalent for two distinct texts. That means the model produced garbage output (priors)
    5. Remove the observations, as well as the observations for the same combination of experimental vars, so datasets across all combinations are equivalent
    '''

    helper = pd.DataFrame()

    helper['text_hash'] = hash_pandas_object(df[input_vars], index=False)
    # helper['text_hash'] = df['deal_text'].apply(hash)
    # helper['hashable_tokens'] = df[data_col].apply(tuple)
    helper['hashable_tokens'] = df[data_col].apply(lambda x: tuple(x[0])) # Only apply to first element of list, which is sufficient to detect output that projects Priors

    helper['model_name'] = df['model_name']

    unique_texts_per_output = helper.groupby(['model_name', 'hashable_tokens'])['text_hash'].transform('nunique')
    is_garbage_mask = unique_texts_per_output > 1

    # --- Step 2: Create the "blocklist" of problematic experimental keys ---
    #
    # Define the columns that uniquely identify an experimental trial
    trial_cols = ['input_id', 'prompt_id', 'assistant_prefix']
    
    # Get the keys of trials that failed at least once
    tainted_trials = df.loc[is_garbage_mask, trial_cols].drop_duplicates()
    
    print(f"Found {len(tainted_trials)} experimental trials contaminated by garbage output.")
    # --- Step 3: Perform the "Anti-Merge" to filter the main DataFrame ---
    #
    # We use pd.merge with how='left' and indicator=True.
    # This will join 'df' with our 'problematic_keys' blocklist.
    # - If a row in 'df' has a key on the blocklist, it gets marked as 'both'.
    # - If a row is clean, it gets marked as 'left_only'.
    #
    merged = df.merge(
        tainted_trials, 
        on=trial_cols, 
        how='left', 
        indicator=True # This creates the '_merge' column
    )

    # --- Step 4: Create the final, clean DataFrame ---
    #
    # We select *only* the rows that were *not* on the blocklist.
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

    key_cols = [input_id_col] + [col for col in experimental_groups if col != model_col]

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



from typing import List, Optional
import torch
import pandas as pd


def compute_ratings_from_logits(
    df: pd.DataFrame,
    weights: Optional[List[float]]| torch.Tensor = None
    ) -> pd.DataFrame:
    """
    Calculates mean and mode ratings from a DataFrame column of logits.

    This is a convenience wrapper around the core _compute_ratings_from_tensors.

    Args:
        df: The DataFrame containing the logit data.
        logits_col: The name of the column with the sorted logits.
        weights: A list of weights (e.g., [1, 2, 3, 4, 5]).
                 If None, defaults to this 1-5 scale.

    Returns:
        A new DataFrame with 'mean_rating' and 'mode_rating' columns.
    TODO: Fix for when len(constrained_tokens) is not equal for all data!!!
        - Will have to groupby len(constrained_tokens) (and maybe more)
    """
    # --- 1. Handle Defaults and Data Prep ---
    if weights is None:
        weights = [1.0, 2.0, 3.0, 4.0]
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # Extract data from pandas
    logits_tensor = torch.tensor(df['sorted_logits'].tolist(), dtype=torch.float32)

    # --- 2. Call the Core Logic Function ---
    mean_rating_tensor, mode_rating_tensor = _compute_ratings_from_tensors(
        logits_tensor, weights_tensor
    )

    # --- SAFETY CHECK: Ensure 1-to-1 mapping ---
    batch_size = len(df)
    
    # If the tensor is [Batch, 1], this check passes. 
    # If it is [Batch, N], this fails before you hit the confusing Pandas error.
    if mean_rating_tensor.numel() != batch_size:
        raise ValueError(
            f"Shape Mismatch! DataFrame has {batch_size} rows, but "
            f"mean_rating_tensor has {mean_rating_tensor.numel()} elements "
            f"(Shape: {mean_rating_tensor.shape}). "
            "Did you forget to reduce a dimension?"
        )

    # --- 3. Package Results for Pandas ---
    return pd.DataFrame({
        'mean_rating': mean_rating_tensor.flatten().tolist(),
        'mode_rating': mode_rating_tensor.flatten().tolist()
    }, index=df.index)



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
    # print(softmax_logits.shape)
    # print(weights_tensor.shape)

    mean_rating = softmax_logits @ weights_tensor


    # Calculate mode rating
    mode_rating_index = softmax_logits.argmax(dim=-1)
    mode_rating_label = weights_tensor[mode_rating_index]
    
    return mean_rating, mode_rating_label



# def _compute_ratings_from_tensors(
#     logits_tensor: torch.Tensor,
#     weights_tensor: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Core logic: Calculates mean and mode ratings from tensors.
    
#     Returns:
#         A tuple of (mean_rating_tensor, mode_rating_tensor)
#     """
#     softmax_logits = torch.softmax(logits_tensor, dim=1)
    
#     # Calculate mean rating (weighted average)
#     mean_rating = torch.matmul(softmax_logits, weights_tensor)

#     # Calculate mode rating
#     mode_rating_index = softmax_logits.argmax(dim=1)
#     mode_rating_label = weights_tensor[mode_rating_index]
    
#     return mean_rating, mode_rating_label

# def compute_ratings_from_logits(
#     df: pd.DataFrame,
#     weights: Optional[List[float]]| torch.Tensor = None
#     ) -> pd.DataFrame:
#     """
#     Calculates mean and mode ratings from a DataFrame column of logits.

#     This is a convenience wrapper around the core _compute_ratings_from_tensors.

#     Args:
#         df: The DataFrame containing the logit data.
#         logits_col: The name of the column with the sorted logits.
#         weights: A list of weights (e.g., [1, 2, 3, 4, 5]).
#                  If None, defaults to this 1-5 scale.

#     Returns:
#         A new DataFrame with 'mean_rating' and 'mode_rating' columns.
#     TODO: Fix for when len(constrained_tokens) is not equal for all data!!!
#         - Will have to groupby len(constrained_tokens) (and maybe more)
#     """
#     # --- 1. Handle Defaults and Data Prep ---
#     if weights is None:
#         weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    
#     weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
#     # Extract data from pandas
#     logits_tensor = torch.tensor(df['sorted_logits'].tolist(), dtype=torch.float32)

#     # --- 2. Call the Core Logic Function ---
#     mean_rating_tensor, mode_rating_tensor = _compute_ratings_from_tensors(
#         logits_tensor, weights_tensor
#     )

#     # --- 3. Package Results for Pandas ---
#     # We preserve the index here, as you wanted
#     return pd.DataFrame({
#         'mean_rating': mean_rating_tensor.numpy(),
#         'mode_rating': mode_rating_tensor.numpy()
#     }, index=df.index)



def get_analysis_ready_df(full_config: dict, 
                          active_analysis: Optional[str] = None, 
                          use_cache: bool = False, 
                          force_refresh: bool = False,
                          return_dirty_df: bool = False) -> pd.DataFrame:

    active_analysis_name = active_analysis if active_analysis is not None else full_config['active_analysis']
    print(f"Loading files for analysis {active_analysis_name}")
    # active_analysis_name = full_config['active_analysis']
    analysis_name = active_analysis_name.upper()
    raw_data_path = paths.RAW_DATA_DIR / f'{analysis_name}.parquet'
    results_dir = paths.RESULTS_DIR / analysis_name
    # results_dir = Path("../results_test")

    cache_filename = f"{active_analysis_name}_analysis_ready.pkl"
    cache_path = paths.RESULTS_DIR / active_analysis_name / cache_filename

    # 2. Check Cache
    if use_cache and not force_refresh and os.path.exists(cache_path):
        print(f"⚡ Loading cached DataFrame from {cache_path}...")
        try:
            return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"⚠️ Cache file corrupted or incompatible. Re-running pipeline. Error: {e}")

    print("🐢 Running full processing pipeline...")

    analysis_config = full_config['analyses'][active_analysis_name]
    id_col = analysis_config['keys']['raw_id_col']
    experimental_groups = analysis_config['model_vars']['experimental_groups']
    evaluations_id_col = analysis_config['keys']['evaluations_id_col']
    input_variable_names = analysis_config['variable_names']



    # ---- Specific config vars 
    ## Currently, metadata_df does not contain the top_k variable, but new datasets will have it #TODO implement, top_k from config
    data_col = 'top_k_tokens'
    # data_col = f'top_{top_k}_tokens'
    ## TODO: Just sorts when label_order=None, but I will need to customize to test for positional bias
    ### I will need to extract it from the prompt template if I want to implement this
    label_order = analysis_config['model_vars'].get('label_order', None) 

    input_df = load_input_data(raw_data_path)
    evaluations_df = load_llm_results_data(results_dir) 


    print("Finished loading experiment data")

    # ---- Variable dependent vars
    ## TODO: Now only works when entire dataset is constrained to 5 tokens, not for varying token constraints.
    n_labels = len(evaluations_df['constrained_tokens'].iloc[0])
    label_weights = torch.arange(1, n_labels + 1, dtype=torch.float32)

    # ----- Processing
    # Combine
    merged_df = pd.merge(input_df, evaluations_df, 
                         left_on = id_col, right_on = evaluations_id_col) 
    


    # Clean
    ## Clean first since it relies on the entire dataset to detect the garbage rows
    ## TODO: check todos in function def
    clean_df,dirty_df = remove_garbage_rows(merged_df,
                                   input_variable_names,
                                   data_col)
    
    BALANCE = False #TODO warning hardcoded
    if BALANCE:
        balanced_df = get_balanced_intersection(clean_df, id_col, experimental_groups, model_col = 'model_name')
    else:
        balanced_df = clean_df

    


    # Feature engineering
    sorted_logits_df = create_sorted_logits(balanced_df, label_order)
    # return sorted_logits_df
    ratings_df = compute_ratings_from_logits(sorted_logits_df, label_weights)
    
    # Assemble
    final_df = pd.concat([balanced_df, sorted_logits_df, ratings_df], axis=1)

    if use_cache:
            # Ensure directory exists
            os.makedirs(cache_path.parent, exist_ok=True)
            print(f"💾 Saving result to {cache_path}...")
            final_df.to_pickle(cache_path, protocol=5)

    if return_dirty_df: 
        return final_df, dirty_df

    return final_df



    




