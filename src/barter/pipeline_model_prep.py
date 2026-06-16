import pandas as pd
import numpy as np
import logging
from functools import wraps
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from pathlib import Path
from typing import Dict, Any
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# DECORATORS & LOGGING
# ---------------------------------------------------------


def log_step(func):
    """Decorator to track row loss during filtering steps."""
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        start_len = len(df)
        result_df = func(df, *args, **kwargs)
        end_len = len(result_df)
        diff = start_len - end_len
        pct_lost = (diff / start_len * 100) if start_len > 0 else 0

        logger.info(
            f"[{func.__name__:<25}] Start: {start_len:<6} | End: {end_len:<6} | Dropped: {diff:<5} ({pct_lost:.1f}%)")
        return result_df
    return wrapper

# ---------------------------------------------------------
# FEATURE ENGINEERING & CLUSTERING (Run before filtering)
# ---------------------------------------------------------


def create_market_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Tier-based market scope proxy."""
    df = df.copy()

    def _get_tier(row):
        # Convert list/array to set safely
        countries = row.get('accepted_countries', [])
        if not isinstance(countries, (list, np.ndarray)):
            countries = []
        c_set = set(countries)

        if "NLD" not in c_set:
            if c_set == {"BEL"}:
                return "Tier3_BEL_Only"
            return "Tier4_Germany"
        else:
            if c_set == {"NLD"}:
                return "Tier2_NLD_only"
            return "Tier1_NLD_international"

    df['market_scope'] = df.apply(_get_tier, axis=1)
    return df

# def compute_competitor_density_old(input_df):
#     df = input_df.copy()
#     # Count deals per category per day
#     df['date'] = df['created_at'].dt.date
#     daily_counts = df.groupby(['date', 'consolidated_categories']).size(
#     ).reset_index(name='category_daily_count')

#     # Merge back (subtract 1 so the deal doesn't count itself)
#     df = df.merge(daily_counts, on=['date', 'consolidated_categories'])
#     df['competitor_density'] = df['category_daily_count'] - 1

#     return df


def compute_competitor_density(input_df):
    df = input_df.copy()

    # Extract date if it doesn't already exist
    if 'date' not in df.columns:
        df['date'] = df['created_at'].dt.date

    # Use transform to broadcast the group size back to the original rows, minus 1
    df['competitor_density'] = df.groupby(['date', 'macro_category'])[
        'date'].transform('size') - 1

    return df


def generate_time_FE_columns(input_df):
    df = input_df.copy()
    df['year_month'] = df['created_at'].dt.strftime('%Y-%m')
    df['year_week'] = df['created_at'].dt.strftime('%G-W%V')

    return df


def create_social_media_platform_strategy(input_df):

    df = input_df.copy()

    print("--- 🚀 OPTIMIZED GREPPING (O(N) on Base Deals) ---")

    # 1. Perform string concatenation strictly on the unique base dataframe
    combined_text = (
        df['deal_title'].fillna('') + ' ' +
        df['deal_text'].fillna('') + ' ' +
        df['creators_requirement'].fillna('')
    ).str.lower()

    # 2. Extract platforms using regex
    platforms = {
        'instagram': r'\b(?:instagram|ig|insta)\b',
        'tiktok': r'\btiktok\b',
        'youtube': r'\b(?:youtube|yt)\b',
        'facebook': r'\b(?:facebook|fb)\b'
    }

    # We can store these temporarily directly in df
    for plat, regex_pattern in platforms.items():
        df[f'req_{plat}'] = combined_text.str.contains(
            regex_pattern, regex=True).astype(int)

    # 3. Calculate platform count per deal
    df['platform_count'] = df[[
        f'req_{p}' for p in platforms.keys()]].sum(axis=1)
    print("--- 🚀 REGROUPING PLATFORMS FOR STATISTICAL STABILITY ---")

    # We update the conditions to fold YouTube, Facebook, and Omnichannel into one bucket
    conditions = [
        (df['req_instagram'] == 1) & (
            df['platform_count'] == 1),
        (df['req_tiktok'] == 1) & (
            df['platform_count'] == 1),
        (df['req_instagram'] == 1) & (
            df['req_tiktok'] == 1) & (df['platform_count'] == 2),
        # Keep Unspecified separate, as it means "blank"
        (df['platform_count'] == 0)
    ]

    choices = [
        'Instagram_Only',
        'TikTok_Only',
        'IG_and_TikTok',
        'Unspecified'
    ]

    # Anything that doesn't fit the simple IG/TikTok molds (e.g., YouTube, 3+ platforms, FB)
    # gets grouped into a catch-all high-complexity bucket.
    df['platform_strategy'] = np.select(
        conditions,
        choices,
        default='High_Friction_Deliverable'
    )

    print("\nUpdated Distribution in Base Data:")
    print(df['platform_strategy'].value_counts())

    return df

    # # 5. Broadcast (Merge) back to the long analysis dataframe
    # print("\n--- 🔗 MERGING BACK TO FINAL DATAFRAME ---")

    # # We only bring over the column we need to avoid bloating final_df_balanced
    # merge_cols = ['deal_id', 'platform_strategy']

    # final_df_balanced = final_df_balanced.merge(
    #     df[merge_cols],
    #     on='deal_id',
    #     how='left'
    # )

    # # Optional cleanup: drop the redundant 'deal_id' column if it was copied over during the merge
    # # if 'deal_id' in final_df_balanced.columns and 'deal_id' != 'input_id':
    # #     final_df_balanced = final_df_balanced.drop(columns=['deal_id'])

    # print("Merge complete! 'platform_strategy' is now ready for regression.")


def compute_input_word_count(input_df):
    df = input_df.copy()

    df['raw_input_word_count'] = df['title_word_count'] + \
        df['text_word_count'] + df['requirements_word_count']

    return df


def compute_generic_features(input_df):
    df = input_df.copy()

    # Deals that have status live are known to be successful- that's why they have not been deleted. Therefore, we need to control
    df['is_live'] = df['status'] == 'live'

    # Dummy-encode deals that require a large nr. of followers
    df['strict_follower_requirement'] = df.min_social_media_followers > 4000

    return df


def apply_icf_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Thesis-Grade ICF Weighted Jaccard Clustering to content types."""
    df = df.copy()
    logger.info("Starting ICF Clustering for content categories...")

    # 1. Safely extract sets
    df['content_types_set'] = df['content_types'].apply(
        lambda x: tuple(sorted(set(y['name'] for y in x))) if isinstance(
            x, (list, np.ndarray)) else tuple()
    )

    # 2. Calculate true ICF
    all_tags = [tag for combo in df['content_types_set'] for tag in combo]
    if not all_tags:
        df['consolidated_categories'] = 'Unknown'
        return df

    tag_counts = pd.Series(all_tags).value_counts()
    icf_lookup = (1 / tag_counts).to_dict()

    # 3. Get unique combinations
    content_counts = df['content_types_set'].value_counts(
        dropna=False).reset_index()
    content_counts.columns = ['combination', 'frequency']
    unique_combos = content_counts['combination'].tolist()
    n_combos = len(unique_combos)

    # 4. Weighted Jaccard Matrix
    def calculate_weighted_jaccard_dist(tuple1, tuple2):
        set1, set2 = set(tuple1), set(tuple2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 1.0
        intersect_w = sum(icf_lookup.get(t, 0) for t in intersection)
        union_w = sum(icf_lookup.get(t, 0) for t in union)
        return 1.0 - (intersect_w / union_w if union_w > 0 else 0)

    dist_matrix = np.zeros((n_combos, n_combos))
    for i in range(n_combos):
        for j in range(i + 1, n_combos):
            dist = calculate_weighted_jaccard_dist(
                unique_combos[i], unique_combos[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # 5. Global Clustering
    n_clusters = min(80, max(1, n_combos // 2))
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average')
    cluster_labels = model.fit_predict(dist_matrix)

    # 6. Semantic Naming
    cluster_names = {}
    for target_label in range(n_clusters):
        indices = np.where(cluster_labels == target_label)[0]
        tag_tally = Counter()
        for idx in indices:
            combo = unique_combos[idx]
            freq = content_counts.iloc[idx]['frequency']
            for tag in combo:
                tag_tally[tag] += freq

        if tag_tally:
            top_tags = [tag for tag, count in tag_tally.most_common(2)]
            # Clean string joining so statsmodels doesn't break
            cluster_names[target_label] = "_".join(
                sorted(top_tags)).replace(" ", "")
        else:
            cluster_names[target_label] = 'Unknown'

    final_mapping = {unique_combos[i]: cluster_names[label]
                     for i, label in enumerate(cluster_labels)}
    df['consolidated_categories'] = df['content_types_set'].apply(
        lambda x: final_mapping.get(x, "Other"))

    logger.info(
        f"ICF Clustering complete. Reduced {n_combos} combos to {df['consolidated_categories'].nunique()} categories.")
    return df


def assign_macro_category(row):
    tag_set = row['content_types_set']
    deal_type = str(row.get('deal_type', 'unknown')).lower()

    # Combine title and description to search for keywords (adjust column names as needed)
    text_content = str(row.get('deal_text', '')) + " " + \
        str(row.get('description', ''))

    # Handle floats, NaNs, or empty sets
    if not isinstance(tag_set, (set, list, tuple)) or pd.isna(tag_set):
        return 'Uncategorized'

    cleaned_tags = {str(t) for t in tag_set if str(
        t).lower() not in ['ugc', 'nan', 'none']}
    if not cleaned_tags:
        return 'Uncategorized'

    # --- 1. The Validated Family & Parenting Category ---
    mom_keywords = [
        r'\bbaby\b', r'\bchildren\b', r'\bkids\b', r'\bkind\b', r'\bkinderen\b',
        r'\bparents\b', r'\bvader\b', r'\bmoeder\b', r'\bfather\b', r'\bmother\b',
        r'\bouders\b', r'\btoddler\b', r'\bzwanger\b', r'\bpregnant\b'
    ]

    mom_regex = re.compile('|'.join(mom_keywords), re.IGNORECASE)

    if 'Mom' in cleaned_tags:
        if cleaned_tags == {'Mom'}:
            return 'Family & Parenting'
        # Check if the text actually contains parenting keywords
        elif mom_regex.search(text_content):
            return 'Family & Parenting'
        else:
            # If it's a fake "Mom" deal, remove the tag so it falls into its true category below
            cleaned_tags.remove('Mom')

    # If a deal didn't have the 'Mom' tag but heavily features the keywords,

    if not cleaned_tags:
        return 'Other'

    if 'Animals' in cleaned_tags:
        return 'Animals'

    # --- 3. Split Food & Beverage Logic ---
    has_food_drink = any(tag in cleaned_tags for tag in [
                         'Food', 'Drinks', 'Vegan'])
    has_cooking = 'Cooking' in cleaned_tags

    if has_cooking or (has_food_drink and deal_type == 'online'):
        return 'Home Cooking & CPG'
    elif has_food_drink and deal_type == 'physical':
        return 'Dining & Hospitality'
    elif has_food_drink:
        return 'Food & Beverage (Unknown Type)'

    # --- 4. Beauty ---
    if 'Beauty' in cleaned_tags:
        return 'Beauty'

    # --- 5. Fashion ---
    if 'Fashion' in cleaned_tags:
        return 'Fashion'

    # --- 6. Split Entertainment & Experiences Logic ---
    ent_tags = ['Entertainment', 'Experiences', 'Nightlife',
                'Movie', 'Music', 'Travel', 'Activities', 'Photography']
    has_ent = any(tag in cleaned_tags for tag in ent_tags)

    if has_ent and deal_type == 'physical':
        return 'In-Person Experiences'
    elif has_ent and deal_type == 'online':
        return 'Digital Entertainment'
    elif has_ent:
        return 'Entertainment & Experiences (Unknown Type)'

    # --- 7. Lifestyle & Home (The Catch-all) ---
    if any(tag in cleaned_tags for tag in ['Lifestyle', 'Luxury']):
        return 'Lifestyle & Home'

        # --- 2. Tech & Male-Skewed Niche ---
    if any(tag in cleaned_tags for tag in ['Gaming', 'Tech', 'Cars', 'Finance', 'Sport']):
        return 'Tech & Niche'

    return 'Other'
# ---------------------------------------------------------
# ATOMIC CLEANING STEPS
# ---------------------------------------------------------


@log_step
def filter_na_company_locations(df):
    mask = (df['deal_type'] == 'physical') & df['company_location_id'].isna()
    return df[~mask].copy()


@log_step
def start_date(df, start_date):
    return df[df['month'] > start_date].copy()


@log_step
def filter_incomplete_weeks(df):
    cutoff_date = df['created_at'].max() - pd.Timedelta(days=14)
    return df[df['created_at'] < cutoff_date].copy()


@log_step
def filter_low_n_categories(df, min_observations=30):
    counts = df['macro_category'].value_counts()
    invalid_categories = counts[counts < min_observations].index
    df.loc[df['macro_category'].isin(
        invalid_categories), 'macro_category'] = 'Other'
    return df


@log_step
def filter_dead_high_volume_deals(df, floor=30):
    condition = (df['applicants_applications_count'] >
                 floor) & (df['apps_after_7_days'] == 0)
    return df[~condition].copy()


@log_step
def filter_archived(df):
    return df[df['status'] != 'archived'].copy()


@log_step
def filter_legacy_partners(df, partner_ids_to_remove):
    if not partner_ids_to_remove:
        return df
    return df[~df['legacy_partner_id'].isin(partner_ids_to_remove)].copy()


@log_step
def filter_partners(df, partner_ids_to_remove):
    if not partner_ids_to_remove:
        return df
    return df[~df['partner_id'].isin(partner_ids_to_remove)].copy()


@log_step
def min_apps_after_seven_days(df, min_apps_after_7_days):
    if min_apps_after_7_days == -1:
        return df
    return df[df['apps_after_7_days'] > min_apps_after_7_days].copy()


@log_step
def min_apps_total(df, min_apps):
    return df[df['applicants_applications_count'] >= min_apps].copy()


@log_step
def filter_invalid_deals(df):
    valid_deals = df['go_live_at'].notna() & df['images'].notna()
    return df[valid_deals].copy()


@log_step
def max_follower_requirement(df, max_follower_requirement):
    return df[df['min_social_media_followers'] < max_follower_requirement].copy()


@log_step
def log_transform_cols(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
        else:
            logger.warning(f"Column '{col}' not found for log transform.")
    return df

# ---------------------------------------------------------
# MASTER ORCHESTRATOR
# ---------------------------------------------------------


def prepare_model_matrix(features_path: Path, output_path: Path, config: Dict[str, Any]):
    """Loads feature dataframe, applies clustering, filtering, and final transformations."""
    logger.info("Loading Feature Baseline...")
    df = pd.read_parquet(features_path)

    # 1. Base Temporals
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['month'] = df['created_at'].dt.to_period('M').astype(str)
    # Time FE cols
    df = generate_time_FE_columns(df)

    # 2. Pre-filter Engineering
    df = create_market_scope(df)

    df['content_types_set'] = df['content_types'].apply(
        lambda x: tuple(sorted(set(y['name'] for y in x))) if isinstance(
            x, (list, np.ndarray)) else tuple()
    )
    df['macro_category'] = df.apply(
        assign_macro_category, axis=1)

    # df = apply_icf_clustering(df)
    df = compute_competitor_density(df)
    df = compute_generic_features(df)
    df = create_social_media_platform_strategy(df)
    df = compute_input_word_count(df)

    # 3. Filtering Pipeline
    logger.info("--- Starting Model Prep Pipeline ---")
    clean_df = (
        df
        .pipe(filter_na_company_locations)
        .pipe(start_date, start_date=config.get('START_DATE', '2025-04'))
        .pipe(filter_low_n_categories, min_observations=config.get('MIN_CATEGORY_OBSERVATIONS', 30))
        .pipe(filter_incomplete_weeks)
        .pipe(filter_dead_high_volume_deals, floor=config.get('diff_total_apps_7_days_floor', 10))
        .pipe(filter_archived)
        .pipe(max_follower_requirement, max_follower_requirement=config.get('max_follower_requirement', 25000))
        .pipe(min_apps_after_seven_days, min_apps_after_7_days=config.get('min_apps_after_7_days', -1))
        .pipe(min_apps_total, min_apps=config.get('MIN_APPS_TOTAL', 0))
        .pipe(filter_invalid_deals)
        .pipe(filter_legacy_partners, partner_ids_to_remove=config.get('LEGACY_PARTNER_IDS_TO_REMOVE', []))
        .pipe(filter_partners, partner_ids_to_remove=config.get('PARTNER_IDS_TO_REMOVE', []))
        .pipe(log_transform_cols, cols=config.get('LOG_TRANSFORM_COLS', []))
    )
    logger.info("--- Model Prep Pipeline Complete ---")

    # 4. Final Type Casts
    clean_df['social_requirement_type_id'] = clean_df['social_requirement_type_id'].fillna(
        -1).astype(int).astype(str)

    # Save the output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_path)
    logger.info(f"Model-ready dataframe saved to: {output_path}")

    return clean_df
