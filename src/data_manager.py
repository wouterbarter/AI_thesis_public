# In src/data_manager.py
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
        self.models = set()
        self._tokenizer_cache = {}


    def _get_tokenizer(self, model_name: str):
        if model_name not in self._tokenizer_cache:
            self._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self._tokenizer_cache[model_name]

    def load_all(self):
        '''
        Loads all experiments in a folder from subfolders containing shards
        Returns: pd.DataFrame(metadata), dict(logits_tensor, sequences_tensor)
        '''
        experiment_folders=[x for x in self.results_dir.iterdir() if x.is_dir()] 

        metadata_list = []
        tensor_dict = {'sequences': [], 'top_k_logits': [], 'top_k_indices': [], 'constrained_logits': []}
        
        for folder in experiment_folders:
            try:
                res = ResultsContainer.load_from_shards(folder)
                n_rows = len(res.metadata)
                if n_rows == 0: continue
                #metadata
                metadata_list.append(res.metadata)

                #data
                for key in tensor_dict:
                    # incoming_data = res.data.get(key, [None]*n_rows)
                    tensor_dict[key].extend(res.data[key])
            except KeyError as e:
                print(f"⚠️ Contract Violation in {folder.name}: Missing key {e}")
            except Exception as e:
                print(f"⚠️ Skipping {folder.name}: {e}")
    

        self.metadata_df = pd.concat(metadata_list)
        self.models = set(self.metadata_df['model_name'].unique())
        self.tensor_dict = tensor_dict

        # return metadata_df, tensor_dict
    


    def create_analysis_dataframe(self, tokenize=True) -> pd.DataFrame:
        """Create analysis dataframe with optional tokenization."""
        if tokenize and not self.models:
            print("Tokenize is true but no models have been added. Quitting.")
            return pd.DataFrame()
        
        df = self.metadata_df.copy()
        
        # Add tensor data as lists
        df['sequences'] = [t.tolist() for t in self.tensor_dict['sequences']]
        
        # Add optional fields
        for field in ['top_k_logits', 'constrained_logits']:
            if self.tensor_dict[field]:
                df[field] = [t.tolist() if t is not None else None 
                            for t in self.tensor_dict[field]]
        
        # Tokenization using fast iterator (avoid iterrows)
        if tokenize:

            # Tokenize Top-K
            df['top_k_tokens'] = [
                self._tokenize_ids(self._get_tokenizer(m), ids)
                for m, ids in zip(df['model_name'], self.tensor_dict['top_k_indices'])
            ]

            # Tokenize Constraints (Optional: check column existence first)
            df['constrained_tokens'] = [
                self._tokenize_ids(self._get_tokenizer(m), ids)
                for m, ids in zip(df['model_name'], df['constrained_token_ids'])
            ]





            # top_k_tokens = []
            # constrained_tokens = []
            
            # for model_name, top_k_ids, constrained_ids in zip(
            #     df['model_name'],
            #     self.tensor_dict['top_k_indices'],
            #     df['constrained_token_ids']
            # ):
            #     tokenizer = self._get_tokenizer(model_name)
            #     top_k_tokens.append(self._tokenize_ids(tokenizer, top_k_ids))
            #     constrained_tokens.append(self._tokenize_ids(tokenizer, constrained_ids))
            
            # df['top_k_tokens'] = top_k_tokens
            # df['constrained_tokens'] = constrained_tokens
        
        return df
    


        
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






    


        



class DataManager_old:
    """
    Manages loading and accessing results from MULTIPLE models.
    """

    def __init__(self,
                 default_results_dir: Optional[str | Path] = None,
                 prompts_dict: Optional[Dict[str, PromptTemplate]] = None):

        self.default_results_dir = Path(
            default_results_dir) if default_results_dir else None

        self.master_df: pd.DataFrame = pd.DataFrame()
        self.tensors_by_model = {}  # This will be the nested dictionary
        self._tokenizer_cache = {}  # Cache tokenizers to avoid re-downloading
        self.models = set()
        self.prompts = prompts_dict if prompts_dict else {}

    def _get_tokenizer(self, model_name: str):
        """Helper to load and cache tokenizers."""
        if model_name not in self._tokenizer_cache:
            self._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                model_name)
        return self._tokenizer_cache[model_name]


    def load_all(
        self,
        results_dir: Optional[str | Path] = None,
        prompts_to_analyze: Optional[List[str]] = None
    ):
        """
        Loads all results, creating one master metadata DF and a dictionary
        of consolidated tensors, one entry per model.
        """

        dir_to_load = results_dir if results_dir else self.default_results_dir
        if dir_to_load is None:
            raise ValueError("No results_dir provided. "
                             "Pass it to load_all() or set default_results_dir in __init__.")

        dir_to_load = Path(dir_to_load)
        result_files = list(dir_to_load.glob("*.pt"))

        if not result_files:
            print(f"No .pt files found in {dir_to_load}. Initializing empty DataManager state.")
            
            # Initialize your class properties to a valid, empty state
            # This is the crucial part.
            self.master_df = pd.DataFrame()
            self.consolidated_tensors = {}
            
            return


        if prompts_to_analyze is None:
            # Infer the list of IDs from the prompts dictionary
            prompts_to_analyze = list(self.prompts.keys())
        print(
            f"Loading files. Filtering for {len(prompts_to_analyze)} prompt IDs.")

        # This will group raw tensor lists by model, e.g., {'model-A': {'sequences': [t1, t2]}}
        raw_tensors_by_model = defaultdict(lambda: defaultdict(list))
        metadata_by_model = defaultdict(list)

        #TODO make interface with ResultsManager.load_all()
        ## Will load files from single experiment (now shards instead of single file)


        # 1. Load and group all data by model name
        for file_path in result_files: #TODO this will be file_stem_*
            data = torch.load(file_path, weights_only=False,
                              map_location='cpu')
            metadata = data['metadata']
            # Apply prompt filter. Prompts can be specified in advance, if they are, only the ones specified will be analyzed
            if prompts_to_analyze and metadata['prompt_id'][0] not in prompts_to_analyze:
                continue

            model_name = metadata['model_name'].iloc[0]

            self.models.add(model_name)
            metadata_by_model[model_name].append(metadata)

            for key, tensor in data['tensors'].items():
                raw_tensors_by_model[model_name][key].append(tensor)

        all_metadata_dfs = []
        # 2. Create the single master DataFrame
        for model_name, metadata_list in metadata_by_model.items():
            model_df = pd.concat(metadata_list, ignore_index=True)
            model_df['model_tensor_index'] = range(len(model_df))
            all_metadata_dfs.append(model_df)
            self.tensors_by_model[model_name] = self._consolidate_tensors_for_model(
                model_name, raw_tensors_by_model[model_name])
            # TODO: Fix for when constrained_tokens are of different length
        self.master_df = pd.concat(all_metadata_dfs, ignore_index=True)



    def _consolidate_tensors_for_model(self, model_name, tensor_groups):
        print(f"Consolidating tensors for model: {model_name}")
        tokenizer = self._get_tokenizer(model_name)

        # return tensor_groups
        sequences_list = tensor_groups.get('sequences', [])
        if sequences_list:
            sequences_list = [
                tensor for sublist in sequences_list for tensor in sublist]
            max_len = max(t.shape[1] for t in sequences_list)
            padded_sequences = [
                F.pad(t, (0, max_len - t.shape[1]),
                      'constant', tokenizer.pad_token_id)
                for t in sequences_list
            ]
            tensor_groups['sequences'] = padded_sequences

        # Concatenate all tensor lists for this model
        consolidated_tensors = {
            key: torch.cat(tensor_list, dim=0)
            for key, tensor_list in tensor_groups.items() if tensor_list
        }

        return consolidated_tensors

    def get_top_k_tokens(self, model_name: str):
        """
        Converts the top-k token IDs for a given model into their corresponding token strings.
        """
        # Use the helper method to fetch the tokenizer for better encapsulation
        tokenizer = self._get_tokenizer(model_name)

        # Get the relevant tensor
        indices_tensor = self.tensors_by_model[model_name]['top_1000_indices']

        # Use a list comprehension with .tolist() for safety and efficiency
        return [tokenizer.convert_ids_to_tokens(row.tolist()) for row in indices_tensor]

    def create_analysis_dataframe(self):
        """
        Processes the loaded data into a single, flat DataFrame ready for analysis.
        """
        if self.master_df is None:
            # Raise an exception with a clear, helpful message.
            raise ValueError(
                "Error: Data has not been loaded. "
                "Please call the .load_all() method before calling this function."
            )
        all_rows = []

        # Iterate through the master metadata DataFrame

        for _, row in self.master_df.iterrows():
            model_name = row['model_name']
            tensor_index = row['model_tensor_index']

            # Get the corresponding tensors for this row
            top_indices = self.tensors_by_model[model_name]['top_1000_indices'][tensor_index]
            top_logits = self.tensors_by_model[model_name]['top_1000_logits'][tensor_index]
            constrained_logits = self.tensors_by_model[model_name]['constrained_logits'][tensor_index]

            # Get the tokenizer and convert indices to tokens
            tokenizer = self._get_tokenizer(model_name)
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
            # Create the mapping between the constrained logits and the corresponding tokens
            constrained_indices = row['constrained_indices']
            constrained_tokens = tokenizer.convert_ids_to_tokens(
                constrained_indices)

            # Create a dictionary for this row
            analysis_row = {
                'input_id': row['input_id'],
                'model_name': model_name,
                'prompt_id': row['prompt_id'],
                'assistant_prefix': row['assistant_prefix'],
                'top_1000_tokens': top_tokens,
                'top_1000_logits': top_logits.tolist(),
                'constrained_logits': constrained_logits.tolist(),
                'constrained_tokens': constrained_tokens
            }
            all_rows.append(analysis_row)

        return pd.DataFrame(all_rows)

# --- How to use it ---
# data_manager = DataManager(...)
# data_manager.load_all()
# analysis_df = data_manager.create_analysis_dataframe()
# analysis_df.to_parquet("final_analysis_results.parquet")
