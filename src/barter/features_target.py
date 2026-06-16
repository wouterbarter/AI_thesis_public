# import pandas as pd


# def calculate_apps_target(df_deals: pd.DataFrame, df_apps: pd.DataFrame, n_days: int = 7, count_start_col='created_at') -> pd.DataFrame:
#     """
#     Calculates the number of valid applications within 'n_days' of a deal going live.
#     Accounts for data quirks where deleted_at might imply an invalid application.

#     count_start_col used to be live_since, but this incorrectly led to 0 apps after 7 days for many deals. Created_at is not perfect. To solve this, the deal history should be incorporated, but this is not done for brevity.
#     """
#     app_counts = df_apps.groupby('deal_id').size(
#     ).reset_index(name='actual_app_count')

#     merged = df_apps[['deal_id', 'application_created_at', 'deleted_at']].merge(
#         df_deals[['deal_id', count_start_col,
#                   'applicants_applications_count']],
#         on='deal_id',
#         how='inner'
#     )

#     merged = merged.merge(app_counts, on='deal_id', how='left')

#     # Vectorized time difference
#     merged['days_since_live'] = (
#         merged['application_created_at'] - merged[count_start_col]).dt.days

#     # Time window and specific data-quirk mask
#     time_mask = (merged['days_since_live'] >= 0) & (
#         merged['days_since_live'] <= n_days)
#     quirk_mask = (merged['actual_app_count'] == merged['applicants_applications_count']) | \
#                  (merged['deleted_at'].isna())

#     valid_apps = merged[time_mask & quirk_mask]

#     # Aggregate back to deals
#     target_col = f'apps_after_{n_days}_days'
#     target_counts = valid_apps.groupby(
#         'deal_id').size().reset_index(name=target_col)

#     # Merge to original and fill NaNs
#     df_deals_updated = df_deals.merge(target_counts, on='deal_id', how='left')
#     df_deals_updated[target_col] = df_deals_updated[target_col].fillna(
#         0).astype(int)

#     return df_deals_updated


import pandas as pd


def calculate_apps_target(
    df_deals: pd.DataFrame,
    df_apps: pd.DataFrame,
    n_days: int = 7,
    count_start_col: str = 'first_live_at'
) -> pd.DataFrame:
    """
    Calculates the number of valid applications within a strict 'n_days' wall-clock 
    window from when the deal first went live. Accounts for deletion data quirks.
    """
    # 1. Count actual total apps per deal
    app_counts = df_apps.groupby('deal_id').size(
    ).reset_index(name='actual_app_count')

    # 2. Merge deals and apps
    merged = df_apps[['deal_id', 'application_created_at', 'deleted_at']].merge(
        df_deals[['deal_id', count_start_col, 'applicants_applications_count']],
        on='deal_id',
        how='inner'
    )
    merged = merged.merge(app_counts, on='deal_id', how='left')

    # 3. Force consistent timezones
    merged['application_created_at'] = pd.to_datetime(
        merged['application_created_at'], utc=True)
    merged[count_start_col] = pd.to_datetime(merged[count_start_col], utc=True)

    # 4. Calculate exact elapsed time
    merged['time_since_live'] = merged['application_created_at'] - \
        merged[count_start_col]

    # 5. Build strict masks using exact Timedelta (avoids .dt.days rounding/flooring issues)
    time_mask = (
        (merged['time_since_live'] >= pd.Timedelta(seconds=0)) &
        (merged['time_since_live'] <= pd.Timedelta(days=n_days))
    )

    quirk_mask = (
        (merged['actual_app_count'] == merged['applicants_applications_count']) |
        (merged['deleted_at'].isna())
    )

    valid_apps = merged[time_mask & quirk_mask]

    # 6. Aggregate back to the deal level
    target_col = f'apps_after_{n_days}_days'
    target_counts = valid_apps.groupby(
        'deal_id').size().reset_index(name=target_col)

    # 7. Merge back into the original dataframe (dropping the column first if it already exists)
    df_deals_updated = df_deals.drop(columns=[target_col], errors='ignore').merge(
        target_counts, on='deal_id', how='left'
    )

    # Fill deals that got 0 valid apps
    df_deals_updated[target_col] = df_deals_updated[target_col].fillna(
        0).astype(int)

    return df_deals_updated
