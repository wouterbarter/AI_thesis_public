# RUN run_barter_cleaning.py AND run_barter_features.py FIRST!

import yaml
import argparse
import logging
from pathlib import Path

from src import paths
from src.barter.pipeline_model_prep import prepare_model_matrix

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Safely loads the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, 'r') as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Execute the final Model Preparation pipeline.")
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/config.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    # 2. Load the configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # config = load_config(args.config)
    prep_config = config['analyses']['barter_deals'].get(
        'model_prep_thresholds', {})

    print("prep config: {prep_config}")
    input_path = paths.PROCESSED_DATA_DIR / 'BARTER_DEALS_FEATURES.parquet'
    output_path = paths.PROCESSED_DATA_DIR / 'BARTER_DEALS_MODEL_READY.parquet'

    logger.info("Initiating model preparation module...")

    print(f"Prep config:\n{prep_config}")

    df_model_ready = prepare_model_matrix(
        features_path=input_path,
        output_path=output_path,
        config=prep_config
    )
    logger.info("✅ Model Prep pipeline executed successfully.")


if __name__ == "__main__":
    main()
