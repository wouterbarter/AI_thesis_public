# In a file like 'src/results.py'
import torch
import pandas as pd
from pathlib import Path
from src.modeler import ModelOutput
from dataclasses import dataclass



@dataclass
class ResultsContainer:
    data: dict     # 'sequences', 'logits_values', 'logits_indices'
    metadata: pd.DataFrame
    config: dict

    @classmethod
    def from_model_outputs(cls, 
                           model_name: str,
                           top_k: int,
                           outputs: list[ModelOutput], 
                           ) -> 'ResultsContainer':
        """
        Factory method: Converts a batch of ModelOutput objects into a Container.
        """
        
        # 1. Initialize Buffers
        sequences = []
        logits_values = []
        logits_indices = []
        constrained_values = []
        
        meta_records = []

        # 2. Loop and Unpack
        for output in outputs:
            # --- Data ---
            # Important: Move to CPU immediately to clear GPU RAM
            sequences.append(output.sequence.cpu())
            
            # Unwrap the LogitsContainer
            # This ensures saved files are Class-Agnostic
            logits_values.append(output.logits.values.cpu())
            
            if output.logits.indices is not None:
                logits_indices.append(output.logits.indices.cpu())
                
            if output.logits.constrained_values is not None:
                constrained_values.append(output.logits.constrained_values.cpu())

            # --- Metadata ---
            # Append the dictionary to the list
            record = output.prompt.to_analysis_record()
            record['input_length'] = output.input_length
            meta_records.append(record)

        # 3. Structure Data
        data = {
            'sequences': sequences,
            'logits_values': logits_values
        }
        
        # Only add optional fields if they exist (Sparse/Constrained support)
        if logits_indices: data['logits_indices'] = logits_indices
        if constrained_values: data['constrained_values'] = constrained_values

        # 4. Structure Config
        # Extract constraints from the first item if present
        config = {
            'model_name': model_name,
            'top_k': top_k,
        }
        
        # If constraints exist, grab the IDs from the first container for the config
        first_container = outputs[0].logits
        if first_container.constrained_indices is not None:
            # Assuming these are static, we save them once in config
            if isinstance(first_container.constrained_indices, torch.Tensor):
                config['constrained_token_ids'] = first_container.constrained_indices.tolist()
            else:
                config['constrained_token_ids'] = first_container.constrained_indices

        # 5. Create DataFrame
        metadata_df = pd.DataFrame(meta_records)

        return cls(data=data, config=config, metadata=metadata_df)


    @classmethod
    def get_experiment_state(cls, experiment_output_dir: Path):
        '''
        Fetches processed IDs and nr. of shards
        output_dir: Folder containing experiment output
        # Constructed from experimental variables in config.yaml
        '''
        shard_files = list(experiment_output_dir.glob("*.pt"))

        processed_ids = set()
        max_index = -1

        if not shard_files:
            return 0, processed_ids

        print(f"Scanning {len(shard_files)} existing shards for state recovery...")

        processed_ids = set()
        for f in shard_files:
            # 1. Determine Index
            try:
                # "name_part_005.pt" -> "005" -> 5
                idx = int(f.stem.split('_part_')[-1])
                if idx > max_index:
                    max_index = idx
            except ValueError:
                continue

            # 2. Extract IDs (Lightweight Load)
            try:
                # Load on CPU to avoid OOM
                data = torch.load(f, map_location='cpu', weights_only=False)
                if 'metadata' in data and not data['metadata'].empty:
                    input_ids = data['metadata']['input_id']
                    processed_ids.update(input_ids.values)

            except Exception as e:
                print(f"Warning: Corrupt shard {f.name} - {e}")

        return max_index + 1, processed_ids

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "data": self.data,
            "config": self.config
        }
        torch.save(payload, filepath)

    @classmethod
    def load_from_shards(cls, output_dir: Path) -> 'ResultsContainer':
        shard_files = sorted(list(output_dir.glob("*pt")))

        if not shard_files:
            raise FileNotFoundError(f"No .pt files found in {output_dir}")

        # Initialize containers
        metadata_list = []
        # Pre-define keys so we can iterate cleanly
        target_keys = ['sequences', 'top_k_logits', 'top_k_indices', 'constrained_logits']
        data_dict = {k: [] for k in target_keys}

        for shard_path in shard_files:
            shard = torch.load(shard_path, weights_only=False, map_location='cpu')
            # Metadata
            meta = shard['metadata']
            config = shard['config']
            data = shard['data']

            num_rows = len(meta)
            final_config = config


            meta['model_name'] = config.get('model_name', 'unknown')
            meta['top_k'] = config.get('top_k', 0)
            c_ids = shard['config'].get('constrained_token_ids')
            meta['constrained_token_ids'] = [c_ids]*num_rows

            metadata_list.append(meta)

            # Data
            data_dict['sequences'].extend(data['sequences'])

            # Logits Values
            vals = data.get('logits_values', [None] * num_rows) #TODO possibly rename?
            data_dict['top_k_logits'].extend(vals)
            
            # Logits Indices
            idxs = data.get('logits_indices', [None] * num_rows)
            data_dict['top_k_indices'].extend(idxs)
            
            # Constrained Values
            c_vals = data.get('constrained_values', [None] * num_rows)
            data_dict['constrained_logits'].extend(c_vals)

        metadata_df = pd.concat(metadata_list, ignore_index = True)

        assert len(metadata_df) == len(data_dict['top_k_logits']), "Alignment Error: Data/Meta length mismatch!"

        # return {'data': data_dict, 'metadata': metadata_df}
        return cls(
                    metadata=metadata_df, 
                    data=data_dict, 
                    config=final_config
                )






class ResultsContainer_old: 
    """
    A simple container to hold and manage the final results of an experiment.
    It links a metadata DataFrame with a dictionary of tensors.
    """
    # def __init__(self, metadata_df: pd.DataFrame, tensors: dict[str, torch.Tensor]):
    #     # Validate that the DataFrame and tensors are aligned
    #     if len(metadata_df) != list(tensors.values())[0].shape[0]:
    #         raise ValueError("DataFrame length and tensor dimension 0 must match.")
        
    #     self.metadata = metadata_df
    #     self.tensors = tensors

    def __init__(self, metadata_df=None, tensors=None):
            self.metadata = metadata_df if metadata_df is not None else pd.DataFrame()
            self.tensors = tensors if tensors is not None else {}


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """Allows for easy access like results[i]"""
        item_metadata = self.metadata.iloc[index].to_dict()
        item_tensors = {key: tensor[index] for key, tensor in self.tensors.items()}
        return {'metadata': item_metadata, 'tensors': item_tensors}

    def save(self, filepath: Path):
        """Saves the entire object to a single file."""
        data_to_save = {
            'metadata': self.metadata,
            'tensors': self.tensors
        }
        torch.save(data_to_save, filepath)
        print(f"Results saved to {filepath}")


    def append(self, new_results: 'ResultsContainer'):
        """Appends data from another ResultsContainer, handling empty cases."""
        # If the current container is empty, just become the new one.
        if self.metadata.empty:
            self.metadata = new_results.metadata
            self.tensors = new_results.tensors
            return

        # Otherwise, concatenate the data.
        self.metadata = pd.concat([self.metadata, new_results.metadata], ignore_index=True)

        for key, new_tensor in new_results.tensors.items():
            if type(new_tensor) == list:
                self.tensors[key].extend(new_tensor)
            elif type(new_tensor) == torch.Tensor:
                self.tensors[key] = torch.cat([self.tensors[key], new_tensor], dim=0)
            else:
                raise ValueError("The new_results object can only contain types List and torch.Tensor")

        # for key, new_tensor in new_results.tensors.items():
        #     self.tensors[key] = torch.cat([self.tensors[key], new_tensor], dim=0)


        # for key, new_tensor in new_results.tensors.items():
        #     if key in self.tensors:
        #         self.tensors[key] = torch.cat([self.tensors[key], new_tensor], dim=0)
        #     else: # This case is unlikely but makes the method more robust
        #         self.tensors[key] = new_tensor

    @classmethod
    def load(cls, filepath: Path):
        """Loads results from a file."""
        data = torch.load(filepath, weights_only=False, map_location='cpu')
        print(f"Loaded {len(data['metadata'])} results from {filepath}")
        return cls(data['metadata'], data['tensors'])
    
