import yaml
import argparse
import logging
from pathlib import Path

from src import paths
from src.barter.exposure import process_exposure_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Execute the Exposure Calculation pipeline.")
    parser.add_argument('--config', type=str,
                        default='src/configs/config.yaml')
    args = parser.parse_args()

    # 1. Define paths
    # INPUTS: Output from your barter_applications_dataset_creator notebook
    deals_input_path = paths.RAW_DATA_DIR / 'BARTER_DEALS.parquet'
    logs_input_path = paths.RAW_DATA_DIR / 'DEAL_ACTIVITY_LOGS.parquet'

    # OUTPUT: This becomes the direct input for run_barter_cleaning.py
    enriched_output_path = paths.RAW_DATA_DIR / 'BARTER_DEALS_WITH_EXPOSURE.parquet'

    # 2. Execute
    logger.info("Initiating exposure processing module...")
    df_enriched = process_exposure_data(
        deals_path=deals_input_path,
        logs_path=logs_input_path,
        output_path=enriched_output_path
    )
    logger.info("✅ Exposure calculation pipeline executed successfully.")


if __name__ == "__main__":
    main()
