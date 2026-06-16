import pandas as pd
from collections import defaultdict
import numpy as np


def get_daily_uniques(target_dates_ns, event_dates, creators, window_days):
    """Core sliding window engine for calculating unique creators over time."""
    window_ns = pd.Timedelta(days=window_days).value
    n_events = len(event_dates)
    results = []
    counts = defaultdict(int)
    unique_count = 0
    left_ptr, right_ptr = 0, 0

    for current_day_ns in target_dates_ns:
        min_time_ns = current_day_ns - window_ns

        while right_ptr < n_events and event_dates[right_ptr] < current_day_ns:
            inf = creators[right_ptr]
            if counts[inf] == 0:
                unique_count += 1
            counts[inf] += 1
            right_ptr += 1

        while left_ptr < right_ptr and event_dates[left_ptr] <= min_time_ns:
            inf = creators[left_ptr]
            counts[inf] -= 1
            if counts[inf] == 0:
                unique_count -= 1
            left_ptr += 1

        results.append(unique_count)
    return results


def get_independent_actives(df: pd.DataFrame, columns_list: list) -> pd.DataFrame:
    """Wrapper to calculate rolling actives split by category (like country)."""
    df = df.sort_values('application_created_at').reset_index(drop=True)
    dateranges = pd.date_range(
        start=df['application_created_at'].min().normalize(),
        end=df['application_created_at'].max().normalize(), freq='D'
    )
    target_dates_ns = dateranges.astype('int64').values
    all_segments = []

    for col in columns_list:
        for val in df[col].dropna().unique():
            df_subset = df[df[col] == val]
            event_dates = df_subset['application_created_at'].astype(
                'int64').values
            influencers = df_subset['creator_id'].values

            df_temp = pd.DataFrame({
                'date': dateranges,
                'category': col,
                'value': val,
                'active_last_month': get_daily_uniques(target_dates_ns, event_dates, influencers, 30),
                'active_last_week': get_daily_uniques(target_dates_ns, event_dates, influencers, 7)
            })
            all_segments.append(df_temp)

    return pd.concat(all_segments, ignore_index=True)


def build_online_liquidity(df_deals: pd.DataFrame, df_apps: pd.DataFrame) -> pd.DataFrame:
    """Calculates eligible online creators by summing matching country pools."""
    active_by_country = get_independent_actives(
        df_apps, ['country_code_creators']).sort_values('date')
    df_active_wide = active_by_country.pivot(index='date', columns='value', values=[
                                             'active_last_month', 'active_last_week'])
    df_active_wide.columns = [
        f"{metric}_{country}" for metric, country in df_active_wide.columns]
    df_active_wide = df_active_wide.reset_index()

    df_deals_wactives = pd.merge_asof(
        df_deals.sort_values('created_at'),
        df_active_wide.sort_values('date'),
        left_on='created_at',
        right_on='date'
    )

    valid_countries = df_apps['country_code_creators'].unique()

    def return_actives(row):
        accepted_countries = [
            x for x in row['accepted_countries'] if x in valid_countries]

        week_cols = [f"active_last_week_{x}" for x in accepted_countries]
        month_cols = [f"active_last_month_{x}" for x in accepted_countries]

        return pd.Series(
            {"eligible_last_week": row[week_cols].sum(),
             "eligible_last_month": row[month_cols].sum()}
        )

    df_deals_wactives[['online_eligible_last_week', 'online_eligible_last_month']
                      ] = df_deals_wactives.apply(return_actives, axis=1)
    df_deals_wactives.loc[df_deals_wactives['deal_type'] == 'physical', [
        'online_eligible_last_week', 'online_eligible_last_month']] = 0

    return df_deals_wactives.drop(columns=df_active_wide.columns)



# def build_online_liquidity_with_followers(df_deals: pd.DataFrame, df_apps: pd.DataFrame, lookback_windows=[7, 30]) -> pd.DataFrame:
#     """Calculates eligible online creators by dynamically filtering on-demand."""
    
#     df_result = df_deals.copy()
    
#     # 1. Isolate relevant App data and drop missing dates
#     df_a = df_apps[['creator_id', 'country_code_creators', 'max_followers_creators', 'application_created_at', 'deal_type_deals']].copy()
#     df_a = df_a.dropna(subset=['application_created_at'])
    
#     # 2. Convert Data to Pure NumPy Arrays for insane speed
#     # (Pandas loops are slow; NumPy bitwise operations are written in C)
#     app_creators = df_a['creator_id'].values
#     app_countries = df_a['country_code_creators'].astype(str).values
#     app_followers = df_a['max_followers_creators'].fillna(0).values
    
#     # Convert dates to nanoseconds (integers) for lightning-fast math
#     app_dates_ns = df_a['application_created_at'].astype('int64').values

#     # Initialize results columns with 0
#     for days in lookback_windows:
#         df_result[f'online_eligible_last_{days}d'] = 0

#     # 3. Iterate deals and evaluate on-demand
#     for idx, deal in df_result.iterrows():
        
#         # Skip physical deals to save compute
#         if deal.get('deal_type') == 'physical':
#             continue

#         # Extract Deal criteria safely
#         deal_date_ns = pd.Timestamp(deal['created_at']).value
#         min_foll = deal.get('min_social_media_followers', 0)
#         min_foll = min_foll if pd.notna(min_foll) else 0
        
#         acc_countries = deal.get('accepted_countries', [])
#         acc_countries = acc_countries if isinstance(acc_countries, list) else []

#         # -- THE MAGIC: Vectorized Bitwise Masks --
#         # 1. Which apps match the country list?
#         country_mask = np.isin(app_countries, acc_countries)
        
#         # 2. Which apps meet the follower minimum?
#         foll_mask = app_followers >= min_foll
        
#         # 3. Which apps occurred ON or BEFORE the deal creation?
#         time_mask_max = app_dates_ns <= deal_date_ns

#         # Combine base criteria
#         base_mask = country_mask & foll_mask & time_mask_max

#         # Now apply the sliding windows
#         for days in lookback_windows:
#             min_date_ns = deal_date_ns - pd.Timedelta(days=days).value
            
#             # Apps strictly within the lookback window
#             time_mask_min = app_dates_ns >= min_date_ns
            
#             final_mask = base_mask & time_mask_min
            
#             # Extract valid creator IDs and count uniques
#             valid_creators = app_creators[final_mask]
#             unique_count = len(np.unique(valid_creators))
            
#             # Assign using .at for speed
#             df_result.at[idx, f'online_eligible_last_{days}d'] = unique_count

#     return df_result


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def build_online_liquidity_with_followers(
    df_deals: pd.DataFrame, 
    df_apps: pd.DataFrame, 
    lookback_windows: list = [7, 30]
) -> pd.DataFrame:
    """Calculates non-overlapping (mutually exclusive) online creators using dynamic filters."""
    
    df_result = df_deals.copy()
    
    # 1. Isolate relevant App data
    df_a = df_apps[['creator_id', 'country_code_creators', 'max_followers_creators', 'application_created_at']].copy()
    df_a = df_a.dropna(subset=['application_created_at'])
    
    # 2. Convert Data to Pure NumPy Arrays
    app_creators = df_a['creator_id'].values
    app_countries = df_a['country_code_creators'].astype(str).values
    app_followers = df_a['max_followers_creators'].fillna(0).values
    app_dates_ns = df_a['application_created_at'].astype('int64').values

    # 3. Create Mutually Exclusive Time Bins
    time_edges = [0] + sorted(lookback_windows)
    time_bins = [(time_edges[i], time_edges[i+1]) for i in range(len(time_edges)-1)]
    col_names = [f"online_active_{start}_{end}d" for start, end in time_bins]
    
    for col in col_names:
        df_result[col] = 0

    # 4. Iterate deals and evaluate on-demand
    for idx, deal in df_result.iterrows():
        
        # Check both possible column names for deal type
        d_type = deal.get('deal_type_deals', deal.get('deal_type'))
        if d_type == 'physical':
            continue

        # Extract Deal criteria safely
        deal_date_ns = pd.Timestamp(deal['created_at']).value
        min_foll = deal.get('min_social_media_followers', 0)
        min_foll = min_foll if pd.notna(min_foll) else 0
        
        acc_countries = deal.get('accepted_countries', [])
        acc_countries = acc_countries if isinstance(acc_countries, list) else []

        # --- THE FIX: Country Mask Logic ---
        if not acc_countries:
            # If the list is empty, there are no restrictions. EVERYONE passes.
            country_mask = np.ones(len(app_countries), dtype=bool)
        else:
            # Otherwise, check against the specific list
            country_mask = np.isin(app_countries, acc_countries)
        # -----------------------------------

        foll_mask = app_followers >= min_foll
        
        # Filter down to just the valid creators for this deal before doing time math
        base_mask = country_mask & foll_mask & (app_dates_ns <= deal_date_ns)
        
        valid_creators = app_creators[base_mask]
        valid_dates = app_dates_ns[base_mask]

        # -- MUTUALLY EXCLUSIVE TIME MASKS --
        for (start_days, end_days), col_name in zip(time_bins, col_names):
            max_ns = deal_date_ns - pd.Timedelta(days=start_days).value
            min_ns = deal_date_ns - pd.Timedelta(days=end_days).value
            
            if start_days == 0:
                bin_mask = (valid_dates >= min_ns) & (valid_dates <= max_ns)
            else:
                bin_mask = (valid_dates >= min_ns) & (valid_dates < max_ns)
                
            bin_creators = valid_creators[bin_mask]
            
            df_result.at[idx, col_name] = len(np.unique(bin_creators))

    return df_result

# def build_online_liquidity_with_followers(
#     df_deals: pd.DataFrame, 
#     df_apps: pd.DataFrame, 
#     lookback_windows: list = [7, 30]
# ) -> pd.DataFrame:
#     """Calculates non-overlapping (mutually exclusive) online creators using dynamic filters."""
    
#     df_result = df_deals.copy()
    
#     # 1. Isolate relevant App data
#     df_a = df_apps[['creator_id', 'country_code_creators', 'max_followers_creators', 'application_created_at', 'deal_type_deals']].copy()
#     df_a = df_a.dropna(subset=['application_created_at'])
    
#     # 2. Convert Data to Pure NumPy Arrays
#     app_creators = df_a['creator_id'].values
#     app_countries = df_a['country_code_creators'].astype(str).values
#     app_followers = df_a['max_followers_creators'].fillna(0).values
#     app_dates_ns = df_a['application_created_at'].astype('int64').values

#     # 3. Create Mutually Exclusive Time Bins
#     # E.g., if lookback_windows is [7, 30], edges become [0, 7, 30]
#     time_edges = [0] + sorted(lookback_windows)
#     time_bins = [(time_edges[i], time_edges[i+1]) for i in range(len(time_edges)-1)]
    
#     # Pre-generate column names (e.g., 'online_active_0_7d', 'online_active_7_30d')
#     col_names = [f"online_active_{start}_{end}d" for start, end in time_bins]
#     for col in col_names:
#         df_result[col] = 0

#     # 4. Iterate deals and evaluate on-demand
#     for idx, deal in df_result.iterrows():
        
#         # Skip physical deals to save compute
#         if deal.get('deal_type') == 'physical':
#             continue

#         # Extract Deal criteria safely
#         deal_date_ns = pd.Timestamp(deal['created_at']).value
#         min_foll = deal.get('min_social_media_followers', 0)
#         min_foll = min_foll if pd.notna(min_foll) else 0
        
#         acc_countries = deal.get('accepted_countries', [])
#         acc_countries = acc_countries if isinstance(acc_countries, list) else []

#         # -- BASE MASKS (Space & Influence) --
#         country_mask = np.isin(app_countries, acc_countries)
#         foll_mask = app_followers >= min_foll
        
#         # Filter down to just the valid creators for this deal before doing time math
#         base_mask = country_mask & foll_mask & (app_dates_ns <= deal_date_ns)
        
#         valid_creators = app_creators[base_mask]
#         valid_dates = app_dates_ns[base_mask]

#         # -- MUTUALLY EXCLUSIVE TIME MASKS --
#         for (start_days, end_days), col_name in zip(time_bins, col_names):
            
#             # Calculate the nanosecond boundaries for this specific bin
#             max_ns = deal_date_ns - pd.Timedelta(days=start_days).value
#             min_ns = deal_date_ns - pd.Timedelta(days=end_days).value
            
#             # CRITICAL: Prevent Double Counting on the boundary!
#             # If start_days is 0, we include the deal_date exactly (<=)
#             # For older bins (e.g., 7_30), we strictly exclude the 7th-day exact mark (<)
#             if start_days == 0:
#                 bin_mask = (valid_dates >= min_ns) & (valid_dates <= max_ns)
#             else:
#                 bin_mask = (valid_dates >= min_ns) & (valid_dates < max_ns)
                
#             # Extract valid creator IDs in this exact time ring and count uniques
#             bin_creators = valid_creators[bin_mask]
#             unique_count = len(np.unique(bin_creators))
            
#             # Assign directly to the dataframe
#             df_result.at[idx, col_name] = unique_count

#     return df_result




# def build_global_liquidity(df_deals: pd.DataFrame, df_apps: pd.DataFrame) -> pd.DataFrame:
#     """Calculates platform-wide global active users."""
#     df_apps_sorted = df_apps.sort_values(
#         'application_created_at').reset_index(drop=True)
#     dateranges = pd.date_range(
#         start=df_apps_sorted['application_created_at'].min().normalize(),
#         end=df_apps_sorted['application_created_at'].max().normalize(), freq='D'
#     )

#     event_dates = df_apps_sorted['application_created_at'].astype(
#         'int64').values
#     creators = df_apps_sorted['creator_id'].values
#     target_dates_ns = dateranges.astype('int64').values

#     df_global = pd.DataFrame({'date': dateranges})
#     df_global['global_active_last_month'] = get_daily_uniques(
#         target_dates_ns, event_dates, creators, 30)
#     df_global['global_active_last_week'] = get_daily_uniques(
#         target_dates_ns, event_dates, creators, 7)

#     df_model = pd.merge_asof(
#         df_deals.sort_values('created_at'),
#         df_global.sort_values('date'),
#         left_on='created_at',
#         right_on='date',
#         direction='backward'
#     )
#     return df_model.drop(columns=['date'])



def build_global_liquidity(
    df_deals: pd.DataFrame, 
    df_apps: pd.DataFrame, 
    lookback_windows: list = [7, 30]
) -> pd.DataFrame:
    """Calculates mutually exclusive global active users over time."""
    
    # 1. Prepare Dates
    df_apps_sorted = df_apps.dropna(subset=['application_created_at']).sort_values('application_created_at')
    dateranges = pd.date_range(
        start=df_apps_sorted['application_created_at'].min().normalize(),
        end=df_apps_sorted['application_created_at'].max().normalize(), 
        freq='D'
    )

    # 2. Convert Data to NumPy Arrays for Speed
    event_dates_ns = df_apps_sorted['application_created_at'].astype('int64').values
    creators = df_apps_sorted['creator_id'].values
    target_dates_ns = dateranges.astype('int64').values

    # 3. Create Mutually Exclusive Time Bins
    time_edges = [0] + sorted(lookback_windows)
    time_bins = [(time_edges[i], time_edges[i+1]) for i in range(len(time_edges)-1)]
    col_names = [f"global_active_{start}_{end}d" for start, end in time_bins]

    # Initialize results dictionary
    results = {col: [] for col in col_names}
    results['date'] = dateranges

    # 4. Iterate through each target date to calculate historical unique bins
    for target_ns in target_dates_ns:
        
        # Mask to only look at events ON or BEFORE this target date
        past_events_mask = event_dates_ns <= target_ns
        valid_creators = creators[past_events_mask]
        valid_dates = event_dates_ns[past_events_mask]

        # Apply our mutually exclusive time boundaries
        for start_days, end_days in time_bins:
            max_ns = target_ns - pd.Timedelta(days=start_days).value
            min_ns = target_ns - pd.Timedelta(days=end_days).value

            # Boundary control to prevent overlap
            if start_days == 0:
                bin_mask = (valid_dates >= min_ns) & (valid_dates <= max_ns)
            else:
                bin_mask = (valid_dates >= min_ns) & (valid_dates < max_ns)

            # Extract valid creator IDs in this ring and count uniques
            bin_creators = valid_creators[bin_mask]
            results[f"global_active_{start_days}_{end_days}d"].append(len(np.unique(bin_creators)))

    df_global = pd.DataFrame(results)

    # 5. Snap the daily global metrics back to the exact deal timestamps
    df_model = pd.merge_asof(
        df_deals.sort_values('created_at'),
        df_global.sort_values('date'),
        left_on='created_at',
        right_on='date',
        direction='backward'
    )
    
    return df_model.drop(columns=['date'])