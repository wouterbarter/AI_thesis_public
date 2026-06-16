import argparse
import logging
from pathlib import Path

# Import your custom paths and the new extraction module
from src import paths
from src.data.barter.extract_deal_logs import extract_deal_activity_logs

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Deal Activity Logs from PostgreSQL.")
    parser.add_argument(
        '--env',
        type=str,
        default='private_data/.env',
        help='Path to the .env file containing database credentials.'
    )
    args = parser.parse_args()

    # 1. Define paths
    # Resolve the env path relative to the current working directory
    env_file_path = Path(args.env).resolve()

    # OUTPUT: The exact file needed by run_barter_exposure.py
    logs_output_path = paths.RAW_DATA_DIR / 'DEAL_ACTIVITY_LOGS.parquet'

    # 2. Execute
    logger.info("Initiating log extraction module...")

    try:
        df_logs = extract_deal_activity_logs(
            env_path=env_file_path,
            output_path=logs_output_path
        )
        logger.info("✅ Log extraction pipeline executed successfully.")
    except Exception as e:
        logger.error("❌ Log extraction pipeline failed.")
        raise e


if __name__ == "__main__":
    main()
