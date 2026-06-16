import yaml
import argparse
import logging
from pathlib import Path

# RUN run_barter_cleaning.py FIRST!

# Import your custom paths and the new feature engineering orchestrator
from src import paths
from src.barter.pipeline_features import generate_all_features

# Set up basic logging for the runner
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
    # 1. Parse command-line arguments (Updated default path)
    parser = argparse.ArgumentParser(
        description="Execute the Barter Deals Feature Engineering pipeline.")
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

    # 3. Extract the nested feature parameters safely
    try:
        # Navigate the nested dictionary: analyses -> barter_deals -> feature_engineering
        feature_config = config['analyses']['barter_deals'].get(
            'feature_engineering', {})
    except KeyError as e:
        logger.error(
            f"Missing expected config hierarchy for Barter Deals: {e}")
        return

    # 4. Define paths
    clean_deals_path = paths.INTERIM_DATA_DIR / 'BARTER_DEALS_CLEAN.parquet'
    raw_apps_path = paths.RAW_DATA_DIR / 'BARTER_DEAL_APPLICATIONS.parquet'
    processed_output_path = paths.PROCESSED_DATA_DIR / 'BARTER_DEALS_FEATURES.parquet'

    # 5. Execute
    logger.info("Initiating generate_all_features module...")
    df_features = generate_all_features(
        deals_path=clean_deals_path,
        apps_path=raw_apps_path,
        output_path=processed_output_path,
        config=feature_config
    )
    logger.info("✅ Feature Engineering pipeline executed successfully.")


if __name__ == "__main__":
    main()
