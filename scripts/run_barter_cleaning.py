import yaml
import argparse
import logging
from pathlib import Path

# Import your custom paths and the new cleaning module
from src import paths
from src.barter.cleaning import clean_barter_data

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
        description="Execute the Barter Deals cleaning pipeline.")
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

    print(f"Config: {config}")

    # 3. Extract the nested thresholds safely
    try:
        # Navigate the nested dictionary: analyses -> barter_deals -> general_cleaning_thresholds
        cleaning_config = config['analyses']['barter_deals'].get(
            'general_cleaning_thresholds', {})
    except KeyError as e:
        logger.error(
            f"Missing expected config hierarchy for Barter Deals: {e}")
        return

    print(f"Prep config:\n{cleaning_config}")

    # 4. Define paths
    raw_data_path = paths.RAW_DATA_DIR / 'BARTER_DEALS_WITH_EXPOSURE.parquet'
    interim_output_path = paths.INTERIM_DATA_DIR / 'BARTER_DEALS_CLEAN.parquet'

    # 5. Execute
    logger.info("Initiating clean_barter_data module...")
    df_clean = clean_barter_data(
        raw_path=raw_data_path,
        output_path=interim_output_path,
        config=cleaning_config
    )
    logger.info("✅ Cleaning pipeline executed successfully.")


if __name__ == "__main__":
    main()
