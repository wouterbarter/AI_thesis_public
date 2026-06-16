import yaml
import pandas as pd
from pathlib import Path
# Adjust imports based on your structure
from src import data_processing, paths
from src.prompt_manager import PromptManager
from src import paths

from typing import Optional
# from src.data_processing import load_input_data


class AnalysisEnvironment:
    def __init__(self, active_analysis: Optional[str] = None, sandbox_mode: Optional[bool] = None, config_path: str = '../src/configs/config.yaml'):
        self.active_analysis = active_analysis

        # 1. Load main Config
        with open(config_path, 'r') as f:
            self.full_config = yaml.safe_load(f)

        # load experiment config
        if not active_analysis:
            self.active_analysis = self.full_config['active_analysis']
        if not sandbox_mode:
            self.sandbox_mode = self.full_config.get('sandbox_mode')

        self.analysis_config = self.full_config['analyses'][active_analysis]
        self.analysis_config.setdefault('metadata', {})

        # analysis_name = self.full_config['active_analysis'].upper()
        # analysis_name = self.analysis_config['active_analysis_name'].upper()
        self.analysis_name = active_analysis.upper()

        if not self.sandbox_mode:
            results_dir = paths.RESULTS_DIR / self.analysis_name
        else:
            results_dir = paths.RESULTS_DIR / 'sandbox' / self.analysis_name

        exp_config_path = results_dir / 'experiment_config.yaml'
        if exp_config_path.exists():
            with open(exp_config_path) as f:
                exp_config = yaml.safe_load(f)
            input_filename = exp_config['analysis_config']['input_filenames']['processed_data_filename']
        else:
            print(
                "Experiment config not found in results folder. Using fallback to main config.")
            input_filename = self.analysis_config['input_filenames']['processed_data_filename']

        self.analysis_config['metadata']['input_data_path'] = paths.PROCESSED_DATA_DIR / \
            f"{input_filename}.parquet"

        self.analysis_config['metadata']['results_dir'] = results_dir

        self.model_vars = self.full_config['analyses'][self.active_analysis]['model_vars']
        self.experimental_groups = self.model_vars['experimental_groups']

    def load_data(self, use_cache: bool = False, force_refresh: bool = False):

        # 2. Get Processed DF
        self.final_df, self.og_df = data_processing.get_analysis_ready_df(
            # full_config=self.full_config,
            analysis_config=self.analysis_config,
            active_analysis=self.active_analysis,
            use_cache=use_cache,
            force_refresh=force_refresh
        )

        # # 3. Get Original/Clean DF
        # use_processed_input_data = self.full_config['generation_globals']['use_processed_data']
        # print(f"USE PROCESSED INPUT DATA = {use_processed_input_data}")
        # input_data_dir = paths.PROCESSED_DATA_DIR if use_processed_input_data else paths.RAW_DATA_DIR
        # input_filename = self.analysis_config['input_filenames']['processed_data_filename'] if use_processed_input_data else self.analysis_config['input_filenames']['raw_data_filename']

        # input_data_path = input_data_dir / (input_filename + ".parquet")

        # self.og_df = pd.read_parquet(input_data_path)

        return self.final_df, self.og_df

    def load_prompts(self):

        # 4. Prompt Management
        if self.sandbox_mode:
            prompt_path = Path(
                f"../prompts/PromptSuites/sandbox/{self.analysis_name}")
        else:
            prompt_path = Path(f"../prompts/PromptSuites/{self.analysis_name}")

        self.pm = PromptManager(folder=prompt_path)
        self.pm.load_all()

        self.prompt_hash_map = {
            key: item.metadata.get('description', 'No description')
            for key, item in self.pm.suites.items()
        }
        return self.pm, self.prompt_hash_map
