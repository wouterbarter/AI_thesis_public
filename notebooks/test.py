from src.analysis.regression_models import create_regression_runner
from src.analysis.engine import AnalysisEnvironment
from src.utils import interactive_regression_results_selector  # Setup


env = AnalysisEnvironment(active_analysis='barter_deals',
                          config_path='src/configs/config.yaml')

# Load everything
final_df, og_df = env.load_data(raw_filename='BARTER_DEALS_CLEAN.parquet')
pm, prompt_map = env.load_prompts(prompt_subfolder='sandbox/BARTER_DEALS')
