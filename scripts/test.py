import torch
import pandas as pd
from pathlib import Path

from src.prompt_manager import PromptManager
from src.data_manager import DataManager

from src import data_processing
import yaml
from src import paths
import hashlib

from src.analysis.reliability import ReliabilityAnalyzer


with open('src/configs/config.yaml', 'r') as f:
    full_config = yaml.safe_load(f)

# active_analysis = 'MCGILL_QA_FEEDBACK'
analysis_config = full_config['analyses']['barter_deals']

# active_analysis = full_config['active_analysis']
raw_data_filename = analysis_config['input_filenames']['raw_data_filename']
model_vars = analysis_config['model_vars']
experimental_groups = model_vars['experimental_groups']


raw_df = pd.read_parquet(paths.RAW_DATA_DIR / f'{raw_data_filename}.parquet')


final_df, dirty_df = data_processing.get_analysis_ready_df(full_config=full_config,
                                                           active_analysis='barter_deals',
                                                           use_cache=False,
                                                           force_refresh=False,
                                                           return_dirty_df=True,
                                                           balance_experimental_trials=False)
