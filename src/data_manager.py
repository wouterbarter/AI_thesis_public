# In src/data_manager.py

import os
import pandas as pd
from pathlib import Path
from typing import Optional


import torch
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Optional, List, Dict

from src.prompt_manager import PromptManager, PromptTemplate

from src.results import ResultsContainer


class DataManager:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.master_df: pd.DataFrame = pd.DataFrame()
        self._tokenizer_cache = {}

    def _get_tokenizer(self, model_name: str):
        if model_name not in self._tokenizer_cache:
            self._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                model_name)
        return self._tokenizer_cache[model_name]

    def _tokenize_ids(self, tokenizer, ids):
        """Helper to tokenize a list or nested list of token IDs."""
        if ids is None:
            return None

        # Convert tensor to list if needed
        ids_list = ids.tolist() if hasattr(ids, 'tolist') else ids

        # Handle nested lists (like top_k_indices with shape [Seq, K])
        if ids_list and isinstance(ids_list[0], list):
            return [tokenizer.convert_ids_to_tokens(step) for step in ids_list]
        else:
            return tokenizer.convert_ids_to_tokens(ids_list)

    def _process_and_cache_folder(self, folder: Path, tokenize: bool = True, force_recompile: bool = False) -> pd.DataFrame:
        """
        Compiles shards into a single DataFrame, tokenizes it, and caches it locally in the folder.
        Automatically recompiles if new shards are detected.
        """
        cache_path = folder / "compiled_folder.pkl"
        shard_files = list(folder.glob("*_part_*.pt"))

        if not shard_files:
            return pd.DataFrame()  # Empty folder

        # --- 🕒 STALENESS CHECK ---
        is_cache_stale = False
        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            # Find the modification time of the newest shard
            newest_shard_mtime = max(f.stat().st_mtime for f in shard_files)

            if newest_shard_mtime > cache_mtime:
                is_cache_stale = True

        # 1. Check for existing, UP-TO-DATE folder cache
        if cache_path.exists() and not force_recompile and not is_cache_stale:
            print(f"⚡ Fast-loading compiled folder: {folder.name}")
            try:
                return pd.read_pickle(cache_path)
            except Exception as e:
                print(
                    f"⚠️ Corrupted cache in {folder.name}, rebuilding... ({e})")

        # 2. Trigger Recompile
        if is_cache_stale:
            print(
                f"🔄 New data detected in {folder.name}. Recompiling cache...")
        else:
            print(f"📦 Compiling and tokenizing shards for: {folder.name}")

        # 2. Load Shards
        try:
            res = ResultsContainer.load_from_shards(folder)
            if len(res.metadata) == 0:
                return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ Error loading shards in {folder.name}: {e}")
            return pd.DataFrame()

        # 3. Build the DataFrame for THIS folder
        df = pd.DataFrame(res.metadata)

        # Safety check: if somehow empty after metadata load
        if df.empty:
            return df

        # Add tensor data as lists
        df['sequences'] = [t.tolist() for t in res.data.get('sequences', [])]

        for field in ['top_k_logits', 'constrained_logits']:
            if field in res.data and res.data[field]:
                df[field] = [
                    t.tolist() if t is not None else None for t in res.data[field]]

        if 'formatted_prompts' in res.data:
            df['formatted_prompts'] = res.data['formatted_prompts']

        # 4. Tokenization (Slow step, but now we cache it!)
        if tokenize:
            # Assume one model per folder, which matches your naming convention
            model_name = df['model_name'].iloc[0]
            tokenizer = self._get_tokenizer(model_name)

            if 'top_k_indices' in res.data and res.data['top_k_indices']:
                df['top_k_tokens'] = [
                    self._tokenize_ids(tokenizer, ids) for ids in res.data['top_k_indices']
                ]

            if 'constrained_token_ids' in df.columns:
                df['constrained_tokens'] = [
                    self._tokenize_ids(tokenizer, ids) for ids in df['constrained_token_ids']
                ]

        # 5. Save the compiled and tokenized DataFrame to the folder
        df.to_pickle(cache_path, protocol=5)
        return df

    def _validate_schema_alignment(self, folder_dfs: dict, log_dir: Path) -> None:
        """
        Validates that all loaded folder DataFrames have the exact same columns.
        Writes a diagnostic JSON log and raises an error if mismatches are found.
        """
        import json

        # Group folders by their exact column signatures using frozenset
        schema_groups = {}
        for folder_name, df in folder_dfs.items():
            signature = frozenset(df.columns)
            if signature not in schema_groups:
                schema_groups[signature] = []
            schema_groups[signature].append(folder_name)

        # If everyone agrees on the schema, exit cleanly
        if len(schema_groups) <= 1:
            return

        # --- Mismatch Handler ---
        log_path = log_dir / "schema_mismatch_log.json"

        log_data = {
            "error": "Schema mismatch detected across compiled folders.",
            "total_folders_checked": len(folder_dfs),
            "distinct_schemas_found": len(schema_groups),
            "schemas": []
        }

        all_possible_columns = set().union(*schema_groups.keys())

        for signature, folders in schema_groups.items():
            missing_cols = all_possible_columns - signature
            log_data["schemas"].append({
                "column_count": len(signature),
                "columns": list(signature),
                "missing_compared_to_global": list(missing_cols),
                "affected_folders_count": len(folders),
                "affected_folders": folders
            })

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)

        print(
            f"\n🛑 CRITICAL: {len(schema_groups)} different schema variations detected!")
        print(f"A detailed mismatch report has been saved to: {log_path}")

        raise ValueError(
            "Cannot safely concatenate DataFrames with different columns. "
            f"Please review {log_path.name} to identify the outliers and delete their 'compiled_folder.pkl' files."
        )

    def load_all(self, allowed_prompt_ids: Optional[list] = None, allowed_models: Optional[list] = None, tokenize: bool = True):
        '''
        Loads all experiments, utilizing folder-level caching.
        '''
        experiment_folders = [
            x for x in self.results_dir.iterdir() if x.is_dir()]

        if allowed_prompt_ids:
            filtered_folders = [
                f for f in experiment_folders
                if any(f"_prompt-{pid}" in f.name for pid in allowed_prompt_ids)
            ]
            print(
                f"Filtered to {len(filtered_folders)} folders based on allowed_prompt_ids.")
            experiment_folders = filtered_folders

        if allowed_models:
            filtered_folders = [
                f for f in experiment_folders
                if any(f"_model-{model}" in f.name for model in allowed_models)
            ]
            print(
                f"Filtered to {len(filtered_folders)} folders based on allowed_models.")
            experiment_folders = filtered_folders

        folder_dfs = {}

        for folder in experiment_folders:
            folder_df = self._process_and_cache_folder(
                folder, tokenize=tokenize)
            if not folder_df.empty:
                folder_dfs[folder.name] = folder_df

        if not folder_dfs:
            print("⚠️ No valid data loaded from any experiment folders.")
            self.master_df = pd.DataFrame(columns=['model_name'])
            return

        # 1. Validate the contract
        self._validate_schema_alignment(folder_dfs, self.results_dir)

        # 2. Commit the data
        self.master_df = pd.concat(
            list(folder_dfs.values()), ignore_index=True)
        self.models = set(self.master_df['model_name'].dropna().unique())

    def create_analysis_dataframe(self) -> pd.DataFrame:
        """
        Simply returns the master_df, as all tensor conversion and tokenization 
        is now handled and cached at the folder level.
        """
        return self.master_df.copy()


# class DataManager_old_V2:
#     def __init__(self, results_dir: Path):
#         self.results_dir = results_dir
#         self.master_df: pd.DataFrame = pd.DataFrame()
#         self.models = set()
#         self._tokenizer_cache = {}

#     def _get_tokenizer(self, model_name: str):
#         if model_name not in self._tokenizer_cache:
#             self._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
#                 model_name)
#         return self._tokenizer_cache[model_name]

#     def load_all(self, allowed_prompt_ids: Optional[list] = None):
#         '''
#         Loads all experiments in a folder from subfolders containing shards.
#         Optionally filters by specific prompt_ids.
#         Returns: pd.DataFrame(metadata), dict(logits_tensor, sequences_tensor)
#         '''
#         experiment_folders = [
#             x for x in self.results_dir.iterdir() if x.is_dir()]

#         if allowed_prompt_ids:
#             filtered_folders = []
#             for folder in experiment_folders:
#                 # Check if any of the allowed prompt IDs exist in the folder name
#                 # We format it exactly as it appears in generate_output_filename_stem to prevent partial matches
#                 if any(f"_prompt-{pid}" in folder.name for pid in allowed_prompt_ids):
#                     filtered_folders.append(folder)

#             print(
#                 f"Filtered to {len(filtered_folders)} folders (out of {len(experiment_folders)}) based on allowed_prompt_ids.")
#             experiment_folders = filtered_folders

#         metadata_list = []
#         tensor_dict = {'sequences': [], 'top_k_logits': [],
#                        'top_k_indices': [], 'constrained_logits': [], 'formatted_prompts': []}

#         for folder in experiment_folders:
#             try:
#                 res = ResultsContainer.load_from_shards(folder)
#                 n_rows = len(res.metadata)
#                 if n_rows == 0:
#                     continue

#                 # 1. ATOMICITY: Validate contract BEFORE modifying state
#                 missing_keys = [
#                     key for key in tensor_dict.keys() if key not in res.data]
#                 if missing_keys:
#                     print(
#                         f"⚠️ Contract Violation in {folder.name}: Missing keys {missing_keys}")
#                     continue  # Skip this folder entirely so metadata and tensors stay aligned

#                 # 2. COMMIT: Now that we know it's valid, commit the data
#                 metadata_list.append(res.metadata)

#                 # data
#                 for key in tensor_dict:
#                     tensor_dict[key].extend(res.data[key])

#             except KeyError as e:
#                 print(
#                     f"⚠️ Contract Violation in {folder.name}: Missing key {e}")
#             except Exception as e:
#                 print(f"⚠️ Skipping {folder.name}: {e}")

#         # 3. GUARD CLAUSE: Handle empty state (Fixes the pd.concat ValueError)
#         if not metadata_list:
#             print("⚠️ No valid data loaded from any experiment folders.")
#             # Initialize empty schema to prevent downstream KeyErrors
#             self.metadata_df = pd.DataFrame(columns=['model_name'])
#             self.models = set()
#             self.tensor_dict = tensor_dict
#             return

#         self.metadata_df = pd.concat(metadata_list, ignore_index=True)
#         self.models = set(self.metadata_df['model_name'].dropna().unique())
#         self.tensor_dict = tensor_dict

#     def create_analysis_dataframe(self, tokenize=True) -> pd.DataFrame:
#         """Create analysis dataframe with optional tokenization."""
#         if tokenize and not self.models:
#             print("Tokenize is true but no models have been added. Quitting.")
#             return pd.DataFrame()

#         df = self.metadata_df.copy()

#         # Add tensor data as lists
#         df['sequences'] = [t.tolist() for t in self.tensor_dict['sequences']]

#         # Add optional fields
#         for field in ['top_k_logits', 'constrained_logits']:
#             if self.tensor_dict[field]:
#                 df[field] = [t.tolist() if t is not None else None
#                              for t in self.tensor_dict[field]]

#         df['formatted_prompts'] = self.tensor_dict['formatted_prompts']

#         # Tokenization using fast iterator (avoid iterrows)
#         if tokenize:

#             # Tokenize Top-K
#             df['top_k_tokens'] = [
#                 self._tokenize_ids(self._get_tokenizer(m), ids)
#                 for m, ids in zip(df['model_name'], self.tensor_dict['top_k_indices'])
#             ]

#             # Tokenize Constraints (Optional: check column existence first)
#             df['constrained_tokens'] = [
#                 self._tokenize_ids(self._get_tokenizer(m), ids)
#                 for m, ids in zip(df['model_name'], df['constrained_token_ids'])
#             ]

#         return df

#     def _tokenize_ids(self, tokenizer, ids):
#         """Helper to tokenize a list or nested list of token IDs."""
#         if ids is None:
#             return None

#         # Convert tensor to list if needed
#         ids_list = ids.tolist() if hasattr(ids, 'tolist') else ids

#         # Handle nested lists (like top_k_indices with shape [Seq, K])
#         if ids_list and isinstance(ids_list[0], list):
#             return [tokenizer.convert_ids_to_tokens(step) for step in ids_list]
#         else:
#             return tokenizer.convert_ids_to_tokens(ids_list)


# class DataManager_old_v1:
#     """
#     Manages loading and accessing results from MULTIPLE models.
#     """

#     def __init__(self,
#                  default_results_dir: Optional[str | Path] = None,
#                  prompts_dict: Optional[Dict[str, PromptTemplate]] = None):

#         self.default_results_dir = Path(
#             default_results_dir) if default_results_dir else None

#         self.master_df: pd.DataFrame = pd.DataFrame()
#         self.tensors_by_model = {}  # This will be the nested dictionary
#         self._tokenizer_cache = {}  # Cache tokenizers to avoid re-downloading
#         self.models = set()
#         self.prompts = prompts_dict if prompts_dict else {}

#     def _get_tokenizer(self, model_name: str):
#         """Helper to load and cache tokenizers."""
#         if model_name not in self._tokenizer_cache:
#             self._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
#                 model_name)
#         return self._tokenizer_cache[model_name]

#     def load_all(
#         self,
#         results_dir: Optional[str | Path] = None,
#         prompts_to_analyze: Optional[List[str]] = None
#     ):
#         """
#         Loads all results, creating one master metadata DF and a dictionary
#         of consolidated tensors, one entry per model.
#         """

#         dir_to_load = results_dir if results_dir else self.default_results_dir
#         if dir_to_load is None:
#             raise ValueError("No results_dir provided. "
#                              "Pass it to load_all() or set default_results_dir in __init__.")

#         dir_to_load = Path(dir_to_load)
#         result_files = list(dir_to_load.glob("*.pt"))

#         if not result_files:
#             print(
#                 f"No .pt files found in {dir_to_load}. Initializing empty DataManager state.")

#             # Initialize your class properties to a valid, empty state
#             # This is the crucial part.
#             self.master_df = pd.DataFrame()
#             self.consolidated_tensors = {}

#             return

#         if prompts_to_analyze is None:
#             # Infer the list of IDs from the prompts dictionary
#             prompts_to_analyze = list(self.prompts.keys())
#         print(
#             f"Loading files. Filtering for {len(prompts_to_analyze)} prompt IDs.")

#         # This will group raw tensor lists by model, e.g., {'model-A': {'sequences': [t1, t2]}}
#         raw_tensors_by_model = defaultdict(lambda: defaultdict(list))
#         metadata_by_model = defaultdict(list)

#         # TODO make interface with ResultsManager.load_all()
#         # Will load files from single experiment (now shards instead of single file)

#         # 1. Load and group all data by model name
#         for file_path in result_files:  # TODO this will be file_stem_*
#             data = torch.load(file_path, weights_only=False,
#                               map_location='cpu')
#             metadata = data['metadata']
#             # Apply prompt filter. Prompts can be specified in advance, if they are, only the ones specified will be analyzed
#             if prompts_to_analyze and metadata['prompt_id'][0] not in prompts_to_analyze:
#                 continue

#             model_name = metadata['model_name'].iloc[0]

#             self.models.add(model_name)
#             metadata_by_model[model_name].append(metadata)

#             for key, tensor in data['tensors'].items():
#                 raw_tensors_by_model[model_name][key].append(tensor)

#         all_metadata_dfs = []
#         # 2. Create the single master DataFrame
#         for model_name, metadata_list in metadata_by_model.items():
#             model_df = pd.concat(metadata_list, ignore_index=True)
#             model_df['model_tensor_index'] = range(len(model_df))
#             all_metadata_dfs.append(model_df)
#             self.tensors_by_model[model_name] = self._consolidate_tensors_for_model(
#                 model_name, raw_tensors_by_model[model_name])
#             # TODO: Fix for when constrained_tokens are of different length
#         self.master_df = pd.concat(all_metadata_dfs, ignore_index=True)

#     def _consolidate_tensors_for_model(self, model_name, tensor_groups):
#         print(f"Consolidating tensors for model: {model_name}")
#         tokenizer = self._get_tokenizer(model_name)

#         # return tensor_groups
#         sequences_list = tensor_groups.get('sequences', [])
#         if sequences_list:
#             sequences_list = [
#                 tensor for sublist in sequences_list for tensor in sublist]
#             max_len = max(t.shape[1] for t in sequences_list)
#             padded_sequences = [
#                 F.pad(t, (0, max_len - t.shape[1]),
#                       'constant', tokenizer.pad_token_id)
#                 for t in sequences_list
#             ]
#             tensor_groups['sequences'] = padded_sequences

#         # Concatenate all tensor lists for this model
#         consolidated_tensors = {
#             key: torch.cat(tensor_list, dim=0)
#             for key, tensor_list in tensor_groups.items() if tensor_list
#         }

#         return consolidated_tensors

#     def get_top_k_tokens(self, model_name: str):
#         """
#         Converts the top-k token IDs for a given model into their corresponding token strings.
#         """
#         # Use the helper method to fetch the tokenizer for better encapsulation
#         tokenizer = self._get_tokenizer(model_name)

#         # Get the relevant tensor
#         indices_tensor = self.tensors_by_model[model_name]['top_1000_indices']

#         # Use a list comprehension with .tolist() for safety and efficiency
#         return [tokenizer.convert_ids_to_tokens(row.tolist()) for row in indices_tensor]

#     def create_analysis_dataframe(self):
#         """
#         Processes the loaded data into a single, flat DataFrame ready for analysis.
#         """
#         if self.master_df is None:
#             # Raise an exception with a clear, helpful message.
#             raise ValueError(
#                 "Error: Data has not been loaded. "
#                 "Please call the .load_all() method before calling this function."
#             )
#         all_rows = []

#         # Iterate through the master metadata DataFrame

#         for _, row in self.master_df.iterrows():
#             model_name = row['model_name']
#             tensor_index = row['model_tensor_index']

#             # Get the corresponding tensors for this row
#             top_indices = self.tensors_by_model[model_name]['top_1000_indices'][tensor_index]
#             top_logits = self.tensors_by_model[model_name]['top_1000_logits'][tensor_index]
#             constrained_logits = self.tensors_by_model[model_name]['constrained_logits'][tensor_index]

#             # Get the tokenizer and convert indices to tokens
#             tokenizer = self._get_tokenizer(model_name)
#             top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
#             # Create the mapping between the constrained logits and the corresponding tokens
#             constrained_indices = row['constrained_indices']
#             constrained_tokens = tokenizer.convert_ids_to_tokens(
#                 constrained_indices)

#             # Create a dictionary for this row
#             analysis_row = {
#                 'input_id': row['input_id'],
#                 'model_name': model_name,
#                 'prompt_id': row['prompt_id'],
#                 'assistant_prefix': row['assistant_prefix'],
#                 'top_1000_tokens': top_tokens,
#                 'top_1000_logits': top_logits.tolist(),
#                 'constrained_logits': constrained_logits.tolist(),
#                 'constrained_tokens': constrained_tokens
#             }
#             all_rows.append(analysis_row)

#         return pd.DataFrame(all_rows)

# --- How to use it ---
# data_manager = DataManager(...)
# data_manager.load_all()
# analysis_df = data_manager.create_analysis_dataframe()
# analysis_df.to_parquet("final_analysis_results.parquet")
