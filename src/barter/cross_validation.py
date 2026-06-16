# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import warnings
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
# from src.barter.regression_models import create_regression_runner

# # Make sure to import your factory function
# # from regression_models import create_regression_runner


# def cross_validate_barter_models(
#     df: pd.DataFrame,
#     target_col: str,
#     experimental_groups: list,
#     cluster_col: str,
#     cat_vars: list,
#     bin_vars: list,
#     num_vars: list,
#     model_configs: dict,
#     n_splits: int = 5
# ) -> pd.DataFrame:
#     """
#     Cross-validates multiple model configurations using GroupKFold to prevent
#     data leakage across clusters (e.g., partner_id).

#     model_configs should be a dict like:
#     {
#         'Baseline': None,
#         'Mean': ['z_mean_rating'],
#         'Mean + Entropy': ['z_mean_rating', 'normalized_entropy']
#     }
#     """
#     # Create a dummy runner just to access your robust data prep/pivoting logic
#     dummy_runner = create_regression_runner(
#         target_col=target_col,
#         experimental_groups=experimental_groups,
#         cluster_col=cluster_col,
#         model_type='negbin'
#     )

#     results = []
#     grouped = df.groupby(experimental_groups)

#     for group_keys, group_df in grouped:
#         if not isinstance(group_keys, tuple):
#             group_keys = (group_keys,)

#         print(f"\n--- Cross-validating group: {group_keys} ---")

#         for model_name, predictors in model_configs.items():
#             # 1. Use your existing logic to build the design matrix (X), target (y), and clusters
#             data_dict = dummy_runner.prepare_group_data(
#                 group_df=group_df,
#                 cat_predictors=cat_vars,
#                 bin_predictors=bin_vars,
#                 num_predictors=num_vars,
#                 new_predictors=predictors
#             )

#             if data_dict is None or data_dict.get("X") is None:
#                 print(
#                     f"Skipping {model_name} for {group_keys}: Data prep failed.")
#                 continue

#             X = data_dict["X"]
#             y = data_dict["y"]
#             clusters = data_dict["clusters"]

#             # 2. Setup GroupKFold
#             gkf = GroupKFold(n_splits=n_splits)

#             fold_rmse, fold_mae, fold_poisson = [], [], []

#             for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=clusters)):
#                 X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#                 X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

#                 try:
#                     # 3. Fit the model on the train fold
#                     # Suppress statsmodels warnings for cleaner console output during CV
#                     with warnings.catch_warnings():
#                         warnings.simplefilter("ignore")
#                         model = sm.NegativeBinomial(
#                             y_train, X_train, loglike_method="nb2")
#                         # We don't need clustered SEs here because we only care about point predictions for CV
#                         res = model.fit(maxiter=2000, disp=0)

#                     # 4. Predict on the test fold
#                     y_pred = res.predict(X_test)

#                     # 5. Compute out-of-sample metrics
#                     fold_rmse.append(
#                         np.sqrt(mean_squared_error(y_test, y_pred)))
#                     fold_mae.append(mean_absolute_error(y_test, y_pred))

#                     # Poisson deviance is an excellent metric for count data predictions
#                     if (y_pred > 0).all() and (y_test >= 0).all():
#                         fold_poisson.append(
#                             mean_poisson_deviance(y_test, y_pred))

#                 except Exception as e:
#                     print(f"  Fold {fold} failed for {model_name}: {e}")
#                     continue

#             # 6. Aggregate results
#             if fold_rmse:
#                 results.append({
#                     'Experiment Group': " | ".join(map(str, group_keys)),
#                     'Model': model_name,
#                     'RMSE (Mean)': np.nanmean(fold_rmse),
#                     'MAE (Mean)': np.nanmean(fold_mae),
#                     'Poisson Dev (Mean)': np.nanmean(fold_poisson) if fold_poisson else np.nan,
#                 })
#                 print(f"✓ {model_name} CV complete.")

#     return pd.DataFrame(results)


import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from scipy.stats import pearsonr, spearmanr, kendalltau

# Make sure to import your factory function
from src.barter.regression_models import create_regression_runner


def cross_validate_barter_models(
    df: pd.DataFrame,
    target_col: str,
    experimental_groups: list,
    cluster_col: str,
    cat_vars: list,
    bin_vars: list,
    num_vars: list,
    model_configs: dict,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Cross-validates multiple model configurations using GroupKFold to prevent
    data leakage across clusters (e.g., partner_id).
    """
    # Create a dummy runner just to access your robust data prep/pivoting logic
    dummy_runner = create_regression_runner(
        target_col=target_col,
        experimental_groups=experimental_groups,
        cluster_col=cluster_col,
        model_type='negbin'
    )

    results = []
    grouped = df.groupby(experimental_groups)

    for group_keys, group_df in grouped:
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        print(f"\n--- Cross-validating group: {group_keys} ---")

        for model_name, predictors in model_configs.items():
            data_dict = dummy_runner.prepare_group_data(
                group_df=group_df,
                cat_predictors=cat_vars,
                bin_predictors=bin_vars,
                num_predictors=num_vars,
                new_predictors=predictors
            )

            if data_dict is None or data_dict.get("X") is None:
                print(
                    f"Skipping {model_name} for {group_keys}: Data prep failed.")
                continue

            X = data_dict["X"]
            y = data_dict["y"]
            clusters = data_dict["clusters"]

            gkf = GroupKFold(n_splits=n_splits)

            fold_rmse, fold_mae, fold_poisson = [], [], []
            fold_pearson, fold_spearman, fold_kendall = [], [], []

            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=clusters)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = sm.NegativeBinomial(
                            y_train, X_train, loglike_method="nb2")
                        res = model.fit(maxiter=2000, disp=0)

                    y_pred = res.predict(X_test)

                    # Compute standard error metrics
                    fold_rmse.append(
                        np.sqrt(mean_squared_error(y_test, y_pred)))
                    fold_mae.append(mean_absolute_error(y_test, y_pred))

                    if (y_pred > 0).all() and (y_test >= 0).all():
                        fold_poisson.append(
                            mean_poisson_deviance(y_test, y_pred))

                    # Compute correlation metrics safely (requires variance in both arrays)
                    if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
                        # scipy.stats returns (statistic, pvalue); we extract the statistic [0]
                        fold_pearson.append(pearsonr(y_test, y_pred)[0])
                        fold_spearman.append(spearmanr(y_test, y_pred)[0])
                        fold_kendall.append(kendalltau(y_test, y_pred)[0])
                    else:
                        fold_pearson.append(np.nan)
                        fold_spearman.append(np.nan)
                        fold_kendall.append(np.nan)

                except Exception as e:
                    print(f"  Fold {fold} failed for {model_name}: {e}")
                    continue

            # Aggregate results using np.nanmean to ignore any NaN folds safely
            if fold_rmse:
                results.append({
                    'Experiment Group': " | ".join(map(str, group_keys)),
                    'Model': model_name,
                    'RMSE (Mean)': np.nanmean(fold_rmse),
                    'MAE (Mean)': np.nanmean(fold_mae),
                    'Poisson Dev (Mean)': np.nanmean(fold_poisson) if fold_poisson else np.nan,
                    'Pearson r (Mean)': np.nanmean(fold_pearson) if fold_pearson else np.nan,
                    'Spearman Rho (Mean)': np.nanmean(fold_spearman) if fold_spearman else np.nan,
                    'Kendall Tau (Mean)': np.nanmean(fold_kendall) if fold_kendall else np.nan,
                })
                print(f"✓ {model_name} CV complete.")

    return pd.DataFrame(results)
