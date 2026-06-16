import pandas as pd
import ast
import logging
from pathlib import Path
from typing import Dict, Any

from src.barter.features_target import calculate_apps_target
from src.barter.features_temporal import build_online_liquidity, build_online_liquidity_with_followers, build_global_liquidity
from src.barter.features_spatial import build_physical_liquidity

logger = logging.getLogger(__name__)


def safe_literal_eval(val):
    """Safely parses string representations of lists."""
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except:
        return []


def prepare_base_data(df_deals: pd.DataFrame, df_apps: pd.DataFrame, config: Dict[str, Any]):
    """Standardizes dates, mappings, and lists before engineering begins."""
    # Dates
    min_date = config.get('min_date', '2023-01-01')
    df_apps = df_apps[df_apps['application_created_at'] > min_date].copy()
    df_deals = df_deals[df_deals['created_at'] > min_date].copy()

    # Countries mapping
    country_map = config.get('country_to_iso3', {})
    df_apps['country_code_creators'] = df_apps['country_creators'].map(
        country_map)
    df_deals['accepted_countries'] = df_deals['accepted_countries'].apply(
        safe_literal_eval)
    df_apps['accepted_countries_deals'] = df_apps['accepted_countries_deals'].apply(
        safe_literal_eval)

    # Attach company_location_id back to deals
    loc_mapping = df_apps[['deal_id', 'company_location_id']
                          ].dropna().drop_duplicates(subset=['deal_id'])
    df_deals = pd.merge(df_deals, loc_mapping, on='deal_id', how='left')

    # Nr of followers (for eligibility filter) #TODO move this to df_apps generation, for now this is easiest way without re-accessing database
    df_apps['socials_creators'] = df_apps['socials_creators'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    # df_apps['max_followers_creators'] = df_apps['socials_creators'].apply(
    #     get_max_followers)

    return df_deals, df_apps


def generate_all_features(deals_path: Path, apps_path: Path, output_path: Path, config: Dict[str, Any]):
    """Master Pipeline for Feature Generation."""
    logger.info("Loading cleaned deals and raw applications...")
    df_deals = pd.read_parquet(deals_path)
    df_apps = pd.read_parquet(apps_path)

    df_deals, df_apps = prepare_base_data(df_deals, df_apps, config)

    logger.info("1/4: Calculating target variable (Apps)...")
    df_model = calculate_apps_target(
        df_deals, df_apps, n_days=config.get('target_days', 7))

    logger.info("2/4: Calculating Online Liquidity...")
    # df_model = build_online_liquidity(df_model, df_apps)

    df_model = build_online_liquidity_with_followers(df_model,
                                                     df_apps,
                                                     lookback_windows=config.get('lookback_windows', [7, 30]))

    logger.info("3/4: Calculating Physical Liquidity (All Radii)...")
    # Grab the list of radii from config, or default to our three standard ones
    radii_list = config.get('spatial_radii_km', [20.0, 50.0])

    df_model = build_physical_liquidity(
        df_model, df_apps,
        radii_km=radii_list,
        lookback_windows=config.get('lookback_windows', [7, 30])
    )

    logger.info("4/4: Calculating Global Platform Liquidity...")
    df_model = build_global_liquidity(df_model, df_apps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_parquet(output_path)
    logger.info(f"✅ Feature Engineering complete. Saved to: {output_path}")

    return df_model
