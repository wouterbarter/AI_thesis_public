import yaml
import pandas as pd
from pathlib import Path
from src import data_processing, paths # Adjust imports based on your structure
from src.prompt_manager import PromptManager

class AnalysisEnvironment:
    def __init__(self, active_analysis: str, config_path: str = '../src/configs/config.yaml'):
        self.active_analysis = active_analysis
        
        # 1. Load Config
        with open(config_path, 'r') as f:
            self.full_config = yaml.safe_load(f) 
        
        self.model_vars = self.full_config['analyses'][active_analysis]['model_vars']
        self.experimental_groups = self.model_vars['experimental_groups']

    def load_data(self, raw_filename: str, use_cache: bool = False):
        # 2. Get Processed DF
        self.final_df = data_processing.get_analysis_ready_df(
            full_config=self.full_config, 
            active_analysis=self.active_analysis,
            use_cache=use_cache
        )

        # 3. Get Original/Clean DF
        self.og_df = pd.read_parquet(paths.RAW_DATA_DIR / raw_filename)
        return self.final_df, self.og_df

    def load_prompts(self, prompt_subfolder: str):
        # 4. Prompt Management
        
        prompt_path = Path(f"../prompts/PromptSuites/{prompt_subfolder}")
        self.pm = PromptManager(folder=prompt_path)
        self.pm.load_all()
        
        self.prompt_hash_map = {
            key: item.metadata.get('description', 'No description') 
            for key, item in self.pm.suites.items()
        }
        return self.pm, self.prompt_hash_map