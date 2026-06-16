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
        formatted_prompts = []
        logits_values = []
        logits_indices = []
        constrained_values = []

        meta_records = []

        # 2. Loop and Unpack
        for output in outputs:
            # --- Data ---
            # Important: Move to CPU immediately to clear GPU RAM
            sequences.append(output.sequence.cpu())
            formatted_prompts.append(output.formatted_prompt)

            # Unwrap the LogitsContainer
            # This ensures saved files are Class-Agnostic
            logits_values.append(output.logits.values.cpu())

            if output.logits.indices is not None:
                logits_indices.append(output.logits.indices.cpu())

            if output.logits.constrained_values is not None:
                constrained_values.append(
                    output.logits.constrained_values.cpu())

            # --- Metadata ---
            # Append the dictionary to the list
            record = output.prompt.to_analysis_record()
            record['input_length'] = output.input_length
            meta_records.append(record)

        # 3. Structure Data
        data = {
            'sequences': sequences,
            'formatted_prompts': formatted_prompts,
            'logits_values': logits_values
        }

        # Only add optional fields if they exist (Sparse/Constrained support)
        if logits_indices:
            data['logits_indices'] = logits_indices
        if constrained_values:
            data['constrained_values'] = constrained_values

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
    def get_experiment_state(cls, experiment_output_dir: Path) -> tuple[int,  set]:
        shard_files = list(experiment_output_dir.glob("*.pt"))

        # Now stores tuples: (input_id, dimension_name)
        processed_keys = set()
        max_index = -1

        if not shard_files:
            return 0, processed_keys

        print(
            f"Scanning {len(shard_files)} existing shards for state recovery...")

        for f in shard_files:
            try:
                idx = int(f.stem.split('_part_')[-1])
                max_index = max(max_index, idx)
            except ValueError:
                continue

            try:
                data = torch.load(f, map_location='cpu', weights_only=False)
                if 'metadata' in data and not data['metadata'].empty:
                    meta_df = data['metadata']
                    # Zip input_id and dimension_name to create the unique composite key
                    if 'input_id' in meta_df.columns and 'dimension_name' in meta_df.columns:
                        keys = set(
                            zip(meta_df['input_id'], meta_df['dimension_name']))
                        processed_keys.update(keys)
            except Exception as e:
                print(f"Warning: Corrupt shard {f.name} - {e}")

        return max_index + 1, processed_keys

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
        # # Pre-define keys so we can iterate cleanly
        # target_keys = ['sequences', 'formatted_prompts', 'top_k_logits', 'top_k_indices', 'constrained_logits']
        # data_dict = {k: [] for k in target_keys}

        # Map your container keys to the actual keys used in the shard 'data' dictionary
        data_key_mapping = {
            'sequences': 'sequences',
            'formatted_prompts': 'formatted_prompts',
            'top_k_logits': 'logits_values',
            'top_k_indices': 'logits_indices',
            'constrained_logits': 'constrained_values'
        }
        data_dict = {k: [] for k in data_key_mapping.keys()}

        for shard_path in shard_files:
            shard = torch.load(
                shard_path, weights_only=False, map_location='cpu')
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
            # --- Data Loading with Graceful Fallbacks ---
            for target_key, shard_key in data_key_mapping.items():
                try:
                    # Attempt to get the data
                    values = data[shard_key]
                except KeyError:
                    # "Throw" our warning and create dummy data
                    print(
                        f"⚠️ Warning: '{shard_key}' missing in {shard_path.name}. Filling with NAs (legacy compatibility).")
                    values = [None] * num_rows

                data_dict[target_key].extend(values)
            # data_dict['sequences'].extend(data['sequences'])
            # data_dict['formatted_prompts'].extend(data['formatted_prompts'])

            # # Logits Values
            # vals = data.get('logits_values', [None] * num_rows) #TODO possibly rename?
            # data_dict['top_k_logits'].extend(vals)

            # # Logits Indices
            # idxs = data.get('logits_indices', [None] * num_rows)
            # data_dict['top_k_indices'].extend(idxs)

            # # Constrained Values
            # c_vals = data.get('constrained_values', [None] * num_rows)
            # data_dict['constrained_logits'].extend(c_vals)

        metadata_df = pd.concat(metadata_list, ignore_index=True)

        assert len(metadata_df) == len(
            data_dict['top_k_logits']), "Alignment Error: Data/Meta length mismatch!"

        # return {'data': data_dict, 'metadata': metadata_df}
        return cls(
            metadata=metadata_df,
            data=data_dict,
            config=final_config
        )
