import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
import re
import html

# Set up basic logging (Cookiecutter standard)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns to match the standard pipeline schema."""
    df = df.copy()
    df = df.rename(columns={'title': 'deal_title', 'description': 'deal_text'})
    logger.info("Standardized schema (renamed title and description).")
    return df


def process_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Removes timezones and calculates deal duration metrics."""
    df = df.copy()
    time_cols = ['created_at', 'deleted_at', 'updated_at']

    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)

    df['diff_created_deleted'] = df['deleted_at'] - df['created_at']
    # df['diff_updated_deleted'] = df['updated_at'] - df['created_at']

    logger.info("Processed timestamps and calculated durations.")
    return df


def restrict_analysis_period(df: pd.DataFrame, start_month: str = "2025-04") -> pd.DataFrame:
    """
    Restricts the raw export to the intended analysis period.
    With start_month='2025-04', this keeps Deals after April 2025,
    i.e. month > '2025-04'.
    """
    df = df.copy()
    initial_len = len(df)

    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    df = df[df["month"] > start_month].copy()

    logger.info(
        f"Dropped {initial_len - len(df)} rows: Outside analysis period (month <= {start_month})."
    )
    return df


def apply_quality_filters(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies exclusion criteria based on thresholds defined in config.yaml.
    Logs the number of rows dropped at each step.
    """
    df = df.copy()
    initial_len = len(df)

    # 1. Filter: Minimum Online Duration
    # min_days = config.get('min_online_days', 7)
    # df = df[~(df['diff_created_deleted'] < pd.Timedelta(days=min_days))]
    # logger.info(
    #     f"Dropped {initial_len - len(df)} rows: Online < {min_days} days.")
    # len_after_duration = len(df)

    min_hours = config.get('min_online_hours', 160)
    df = df[df['cumulative_exposure_hours'] >= min_hours]
    logger.info(
        f"Dropped {initial_len - len(df)} rows: Online < {min_hours} hours.")
    len_after_duration = len(df)

    # 2. Filter: Spam / Short Text
    df['text_word_count'] = df['deal_text'].str.split().str.len().fillna(0)
    df['title_word_count'] = df['deal_title'].str.split().str.len().fillna(0)
    df['requirements_word_count'] = df['creators_requirement'].str.split(
    ).str.len().fillna(0)  # TODO Add to mask_short_text?

    mask_short_text = (
        (df['title_word_count'] <= config.get('max_spam_title_words', 2)) &
        (df['text_word_count'] <= config.get('max_spam_text_words', 5)) &
        # TODO check if I need to lower
        (df['requirements_word_count'] <= config.get('max_spam_req_words', 5)) &
        (df['applicants_applications_count'] < config.get('max_spam_apps', 10))
    )
    df = df.loc[~mask_short_text]
    logger.info(
        f"Dropped {len_after_duration - len(df)} rows: Short text/Spam heuristics.")
    len_after_spam = len(df)

    # 3. Filter: Test/Duplicate Deals
    mask_duplicate = (
        df['deal_title'].str.contains('duplicate', case=False, na=False) &
        (df['applicants_applications_count'] <
         config.get('max_duplicate_apps', 5))
    )
    df = df[~mask_duplicate]
    logger.info(
        f"Dropped {len_after_spam - len(df)} rows: Flagged as 'duplicate'.")
    len_after_dup = len(df)

    # 4. Filter: Never went live
    df = df[~df['go_live_at'].isna()]
    logger.info(
        f"Dropped {len_after_dup - len(df)} rows: Never went live (go_live_at is NaN).")

    logger.info(
        f"Total Quality Filtering complete. Final row count: {len(df)} (Retained {len(df)/initial_len:.1%})")
    return df


def clean_html_text(text):
    if pd.isna(text):
        return ""

    # Cast to string just in case
    text = str(text)

    # 1. Unescape ALL HTML entities (&nbsp;, &amp;, etc.)
    text = html.unescape(text)

    # 2. Intelligently map structural HTML tags to newlines
    # This catches <br>, <p>, </div>, <li>, etc., and turns them into \n
    text = re.sub(
        r'<(br\s*/?|/?p|/?div|/?li|/?ul|/?ol|/?h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)

    # 3. Replace any remaining non-structural HTML tags (like <b>, <span>, <a>) with a space
    text = re.sub(r'<[^>]+>', ' ', text)

    # 4. Clean up horizontal whitespace ONLY (spaces, tabs) without killing newlines
    # [^\S\n] means "match any whitespace character EXCEPT a newline"
    text = re.sub(r'[^\S\n]+', ' ', text)

    # 5. Compress excessive newlines (e.g., turn 3+ blank lines into just 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def clean_barter_data(raw_path: Path, output_path: Path, config: Dict[str, Any]) -> pd.DataFrame:
    """Master orchestrator for the initial cleaning phase."""
    logger.info(f"Starting cleaning pipeline for: {raw_path.name}")

    df = pd.read_parquet(raw_path)

    df = standardize_schema(df)
    # Clean html texts (some dirty texts contain html tags, remove them)
    for col in ['deal_title', 'deal_text', 'creators_requirement']:
        df[col] = df[col].apply(clean_html_text)

    df = process_timestamps(df)

    df = restrict_analysis_period(
        df,
        start_month=config.get("START_DATE", "2025-04")
    )
    df = apply_quality_filters(df, config)

    # Save the output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Clean data saved to: {output_path}")

    return df
