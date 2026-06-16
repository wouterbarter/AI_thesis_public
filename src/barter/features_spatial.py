import pandas as pd
from src.common.math_utils import calculate_haversine_vectorized

# def build_physical_liquidity(df_deals: pd.DataFrame, df_apps: pd.DataFrame, radius_km: float = 50.0, lookback_windows: list = [7, 30]) -> pd.DataFrame:
#     """Creates a spatial grid and calculates eligible active users within a physical radius."""
#     # 1. Filter for Physical App logs
#     df_locs = df_apps[~df_apps['company_location_id'].isna() & (df_apps['deal_type_deals'] == 'physical')].copy()

#     # 2. Extract unique locations and creators
#     df_unique_locations = df_locs[['company_location_id', 'latitude_company_locations', 'longitude_company_locations']].drop_duplicates()
#     df_unique_creators = df_locs[['creator_id', 'latitude_creators', 'longitude_creators', 'country_code_creators', 'max_followers_creators']].drop_duplicates()
#     # Only drop if we don't know WHERE they are. Missing followers should not exclude them from the grid yet.
#     df_unique_creators = df_unique_creators.dropna(subset=['latitude_creators', 'longitude_creators'])


#     # 3. Cross-Join and Vectorized Haversine
#     df_spatial_grid = df_unique_locations.merge(df_unique_creators, how='cross')
#     df_spatial_grid['distance_km'] = calculate_haversine_vectorized(
#         df_spatial_grid['latitude_company_locations'].values,
#         df_spatial_grid['longitude_company_locations'].values,
#         df_spatial_grid['latitude_creators'].values,
#         df_spatial_grid['longitude_creators'].values
#     )

#     # 4. Aggressive Radius Pruning
#     df_spatial_grid = df_spatial_grid[df_spatial_grid['distance_km'] <= radius_km]

#     # 5. Map back to Deals and check Geographic Eligibility
#     df_deals_spatial = df_deals[['deal_id', 'company_location_id', 'created_at', 'accepted_countries', 'min_social_media_followers']].merge(
#         df_spatial_grid, on='company_location_id', how='inner'
#     )

#     # Handle deals with no requirements and creators with missing data
#     min_reqs = df_deals_spatial['min_social_media_followers'].fillna(0)
#     creator_foll = df_deals_spatial['max_followers_creators'].fillna(0)

#     # Inclusive filtering
#     df_deals_spatial = df_deals_spatial[creator_foll >= min_reqs]


#     df_deals_spatial['is_eligible'] = [
#         code in accepted_list if isinstance(accepted_list, list) else True
#         for code, accepted_list in zip(df_deals_spatial['country_code_creators'], df_deals_spatial['accepted_countries'])
#     ]

#     df_deals_spatial = df_deals_spatial[df_deals_spatial['is_eligible']]

#     # 6. Apply Temporal Activity Filter
#     df_activity = df_apps[['creator_id', 'application_created_at']].drop_duplicates()
#     df_liquidity = df_deals_spatial.merge(df_activity, on='creator_id', how='inner')
#     df_liquidity['days_diff'] = (df_liquidity['created_at'] - df_liquidity['application_created_at']).dt.days

#     # 7. Calculate for multiple windows
#     df_model = df_deals.copy()
#     for daysdiff in lookback_windows:
#         mask = (df_liquidity['days_diff'] >= 0) & (df_liquidity['days_diff'] <= daysdiff)
#         active_counts = df_liquidity[mask].groupby('deal_id')['creator_id'].nunique().reset_index(
#             name=f'physical_active_creators_{daysdiff}d_{int(radius_km)}km'
#         )
#         df_model = df_model.merge(active_counts, on='deal_id', how='left')
#         df_model[f'physical_active_creators_{daysdiff}d_{int(radius_km)}km'] = \
#             df_model[f'physical_active_creators_{daysdiff}d_{int(radius_km)}km'].fillna(0).astype(int)

#     return df_model

import pandas as pd
from src.common.math_utils import calculate_haversine_vectorized


def build_physical_liquidity(
    df_deals: pd.DataFrame,
    df_apps: pd.DataFrame,
    radii_km: list = [20.0, 50.0],
    lookback_windows: list = [7, 30]
) -> pd.DataFrame:
    """Calculates physical liquidity for multiple radii and windows in a single pass."""

    # 1. Filter for Physical App logs
    df_locs = df_apps[df_apps['company_location_id'].notna() & (
        df_apps['deal_type_deals'] == 'physical')].copy()

    # 2. Extract unique locations and creators
    df_unique_locations = df_locs[[
        'company_location_id', 'latitude_company_locations', 'longitude_company_locations']].drop_duplicates()

    df_unique_creators = df_locs[['creator_id', 'latitude_creators', 'longitude_creators',
                                  'country_code_creators', 'max_followers_creators']].drop_duplicates()
    df_unique_creators = df_unique_creators.dropna(
        subset=['latitude_creators', 'longitude_creators'])

    # 3. SINGLE Cross-Join and Vectorized Haversine
    df_spatial_grid = df_unique_locations.merge(
        df_unique_creators, how='cross')
    df_spatial_grid['distance_km'] = calculate_haversine_vectorized(
        df_spatial_grid['latitude_company_locations'].values,
        df_spatial_grid['longitude_company_locations'].values,
        df_spatial_grid['latitude_creators'].values,
        df_spatial_grid['longitude_creators'].values
    )

    # 4. Prune to the MAXIMUM radius to save memory
    max_radius = max(radii_km)
    df_spatial_grid = df_spatial_grid[df_spatial_grid['distance_km'] <= max_radius]

    # 5. Map back to Deals and check Eligibility (Using our fast Explode method)
    df_deals_subset = df_deals[['deal_id', 'company_location_id',
                                'created_at', 'accepted_countries', 'min_social_media_followers']]
    df_deals_exploded = df_deals_subset.explode('accepted_countries')

    df_deals_spatial = df_deals_exploded.merge(
        df_spatial_grid, on='company_location_id', how='inner'
    )

    # Apply Follower Filter
    min_reqs = df_deals_spatial['min_social_media_followers'].fillna(0)
    creator_foll = df_deals_spatial['max_followers_creators'].fillna(0)
    df_deals_spatial = df_deals_spatial[creator_foll >= min_reqs]

    # Apply Country Filter
    no_restrictions = df_deals_spatial['accepted_countries'].isna()
    country_match = df_deals_spatial['accepted_countries'] == df_deals_spatial['country_code_creators']
    df_deals_spatial = df_deals_spatial[no_restrictions | country_match]

    # Drop duplicates generated by the explode
    df_deals_spatial = df_deals_spatial.drop_duplicates(
        subset=['deal_id', 'creator_id'])

    # 6. Apply Temporal Activity Base Filter
    # df_activity = df_apps[['creator_id', 'application_created_at']].drop_duplicates()
    # df_liquidity = df_deals_spatial.merge(df_activity, on='creator_id', how='inner')
    # df_liquidity['days_diff'] = (df_liquidity['created_at'] - df_liquidity['application_created_at']).dt.days

    # 7. Nested Loop: Calculate all combinations of Time and Distance
    # df_model = df_deals.copy()

    # for daysdiff in lookback_windows:
    #     # Filter strictly for the time window
    #     time_mask = (df_liquidity['days_diff'] >= 0) & (df_liquidity['days_diff'] <= daysdiff)
    #     df_time_filtered = df_liquidity[time_mask]

    #     for radius in radii_km:
    #         # Further filter for the specific radius
    #         radius_mask = df_time_filtered['distance_km'] <= radius

    #         # Group and count unique creators
    #         col_name = f'physical_active_creators_{daysdiff}d_{int(radius)}km'
    #         active_counts = df_time_filtered[radius_mask].groupby('deal_id')['creator_id'].nunique().reset_index(name=col_name)

    #         # Merge back to the master deal list
    #         df_model = df_model.merge(active_counts, on='deal_id', how='left')
    #         df_model[col_name] = df_model[col_name].fillna(0).astype(int)

    # ---------------------------------------------------------
    # 6. Apply Temporal Activity Base Filter
    # ---------------------------------------------------------
    df_activity = df_apps[['creator_id',
                           'application_created_at']].drop_duplicates()
    df_liquidity = df_deals_spatial.merge(
        df_activity, on='creator_id', how='inner')
    df_liquidity['days_diff'] = (
        df_liquidity['created_at'] - df_liquidity['application_created_at']).dt.days

    # ---------------------------------------------------------
    # 7. Create Mutually Exclusive Bins (Model-Ready for Regression)
    # ---------------------------------------------------------
    # Ensure our edges start at 0
    # e.g., [0.0, 10.0, 20.0, 50.0]
    radii_edges = [0.0] + sorted(radii_km)
    # e.g., [-1, 7, 30] (-1 so we include day 0)
    time_edges = [-1] + sorted(lookback_windows)

    # Create bin labels
    r_labels = [
        f"{int(radii_edges[i])}_{int(radii_edges[i+1])}km" for i in range(len(radii_edges)-1)]
    t_labels = [
        f"{max(0, time_edges[i])}_{time_edges[i+1]}d" for i in range(len(time_edges)-1)]

    # pd.cut automatically assigns every row to exactly ONE exclusive bin
    df_liquidity['radius_bin'] = pd.cut(
        df_liquidity['distance_km'], bins=radii_edges, labels=r_labels)
    df_liquidity['time_bin'] = pd.cut(
        df_liquidity['days_diff'], bins=time_edges, labels=t_labels)

    # Drop anything that fell outside our maximum tracking windows (e.g., >50km or >30 days)
    df_liquidity = df_liquidity.dropna(subset=['radius_bin', 'time_bin'])

    # Group by the exact combination of Deal, Time Bin, and Radius Bin
    agg_df = df_liquidity.groupby(
        ['deal_id', 'time_bin', 'radius_bin'], observed=True
    )['creator_id'].nunique().reset_index()

    # Create the final, model-ready column names (e.g., 'phys_active_7_30d_10_20km')
    agg_df['feature_name'] = 'phys_active_' + \
        agg_df['time_bin'].astype(str) + '_' + agg_df['radius_bin'].astype(str)

    # Pivot the data so each bin becomes its own column
    pivot_df = agg_df.pivot(index='deal_id', columns='feature_name',
                            values='creator_id').fillna(0).astype(int)

    # Merge back to the main deals dataframe
    df_model = df_deals.merge(pivot_df, on='deal_id', how='left')

    # Fill remaining NaNs to 0 for deals that had absolutely zero creators in the grid
    feature_cols = pivot_df.columns
    df_model[feature_cols] = df_model[feature_cols].fillna(0).astype(int)

    return df_model
