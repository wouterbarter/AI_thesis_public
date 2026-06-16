import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def extract_status(payload: Any) -> str | None:
    """Safely extracts the 'status' field from the nested deal payload."""
    if pd.isna(payload) or not payload:
        return None

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
    else:
        data = payload

    if isinstance(data, dict):
        return data.get('deal', {}).get('status')
    return None


def get_status_transitions(df_logs: pd.DataFrame) -> pd.DataFrame:
    """Parses logs to detect exactly when a deal's status changed."""
    df = df_logs.copy()

    df['old_status'] = df['before'].apply(extract_status)
    df['new_status'] = df['after'].apply(extract_status)

    # Keep only rows where the status actually changed
    transitions = df[
        (df['old_status'] != df['new_status']) &
        (df['old_status'].notna() | df['new_status'].notna())
    ].copy()

    # Map entity_id to deal_id and rename the log creation time to transition_time
    transitions['deal_id'] = transitions['entity_id'].astype(str)
    result = transitions[['deal_id', 'created_at', 'old_status', 'new_status']]

    result = result.sort_values(
        by=['deal_id', 'created_at']).reset_index(drop=True)
    result = result.rename(columns={'created_at': 'transition_time'})

    return result


def compute_cumulative_exposure(df_deals: pd.DataFrame, status_history: pd.DataFrame, n_days: int = 7) -> pd.DataFrame:
    """
    Calculates the exact cumulative hours a deal spent in 'live' status 
    within the strict n_days wall-clock window after it first went live.
    """
    # 1. Get first_live_at for each deal
    first_live = status_history[status_history['new_status']
                                == 'live'].drop_duplicates('deal_id', keep='first')
    first_live = first_live[['deal_id', 'transition_time']].rename(
        columns={'transition_time': 'first_live_at'})

    exposure_df = df_deals[['deal_id', 'created_at']].copy()
    exposure_df['deal_id'] = exposure_df['deal_id'].astype(str)
    exposure_df = exposure_df.merge(first_live, on='deal_id', how='left')

    # Fallback to created_at if no live transition exists, ensure UTC
    exposure_df['first_live_at'] = pd.to_datetime(
        exposure_df['first_live_at'].fillna(exposure_df['created_at']), utc=True)
    exposure_df['cutoff_time'] = exposure_df['first_live_at'] + \
        pd.Timedelta(days=n_days)

    # 2. Prepare the transition logs for interval math
    logs = status_history.copy()
    logs['transition_time'] = pd.to_datetime(logs['transition_time'], utc=True)
    logs = logs.sort_values(['deal_id', 'transition_time'])
    logs['next_transition_time'] = logs.groupby(
        'deal_id')['transition_time'].shift(-1)

    # Merge the cutoff boundaries into the logs
    logs = logs.merge(
        exposure_df[['deal_id', 'first_live_at', 'cutoff_time']], on='deal_id', how='inner')

    # 3. Filter to intervals that overlap with the 7-day window
    logs = logs[logs['transition_time'] <= logs['cutoff_time']]

    logs['end_time'] = logs['next_transition_time'].fillna(logs['cutoff_time'])
    logs['end_time'] = logs[['end_time', 'cutoff_time']].min(axis=1)

    # 4. Sum durations where status was 'live'
    live_intervals = logs[logs['new_status'] == 'live'].copy()
    live_intervals['live_duration_hours'] = (
        live_intervals['end_time'] - live_intervals['transition_time']).dt.total_seconds() / 3600

    cumulative_live = live_intervals.groupby(
        'deal_id')['live_duration_hours'].sum().reset_index()
    cumulative_live = cumulative_live.rename(
        columns={'live_duration_hours': 'cumulative_exposure_hours'})

    # Merge back to main dataframe
    exposure_df = exposure_df.merge(cumulative_live, on='deal_id', how='left')
    exposure_df['cumulative_exposure_hours'] = exposure_df['cumulative_exposure_hours'].fillna(
        0.0)

    # Add the econometric offset
    exposure_df['actual_exposure_hours'] = exposure_df['cumulative_exposure_hours'].clip(
        lower=1.0)
    exposure_df['log_exposure'] = np.log(exposure_df['actual_exposure_hours'])

    return exposure_df[['deal_id', 'first_live_at', 'cumulative_exposure_hours', 'actual_exposure_hours', 'log_exposure']]


def process_exposure_data(deals_path: Path, logs_path: Path, output_path: Path) -> pd.DataFrame:
    """Master orchestrator for the exposure calculation phase."""
    logger.info("Loading raw deals and activity logs...")
    df_deals = pd.read_parquet(deals_path)
    df_logs = pd.read_parquet(logs_path)

    logger.info("Parsing JSON payloads for status transitions...")
    status_history = get_status_transitions(df_logs)

    logger.info("Computing 7-day cumulative wall-clock exposure...")
    exposure_metrics = compute_cumulative_exposure(
        df_deals, status_history, n_days=7)

    # Merge the calculated metrics back into the main deals dataframe
    df_deals['deal_id'] = df_deals['deal_id'].astype(str)
    df_merged = df_deals.merge(exposure_metrics, on='deal_id', how='left')

    # Save the output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(output_path)
    logger.info(f"Exposure-enriched data saved to: {output_path}")

    return df_merged
