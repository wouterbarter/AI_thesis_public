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


def cross_validate_barter_models_with_oof(
    df: pd.DataFrame,
    target_col: str,
    experimental_groups: list,
    cluster_col: str,
    cat_vars: list,
    bin_vars: list,
    num_vars: list,
    model_configs: dict,
    n_splits: int = 5,
    id_col: str = "deal_id",
    model_type: str = "negbin",
    offset_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-validates Barter models and returns:
    1. full-sample OOF metrics per experimental group and model configuration;
    2. deal-level out-of-fold predictions.

    The function uses the regression runner's prepare_group_data() method, so the
    CV design matrix matches the in-sample model construction.
    """

    dummy_runner = create_regression_runner(
        target_col=target_col,
        experimental_groups=experimental_groups,
        cluster_col=cluster_col,
        offset_col=offset_col,
        model_type=model_type,
    )

    metric_rows = []
    oof_rows = []

    grouped = df.groupby(experimental_groups, dropna=False)

    for group_keys, group_df in grouped:
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        group_dict = dict(zip(experimental_groups, group_keys))
        experiment_group = " | ".join(map(str, group_keys))

        print(f"\n--- Cross-validating group: {group_keys} ---")

        for model_name, predictors in model_configs.items():
            data_dict = dummy_runner.prepare_group_data(
                group_df=group_df,
                cat_predictors=cat_vars,
                bin_predictors=bin_vars,
                num_predictors=num_vars,
                new_predictors=predictors,
            )

            if data_dict is None or data_dict.get("X") is None:
                print(
                    f"Skipping {model_name} for {group_keys}: data prep failed.")
                continue

            X = data_dict["X"]
            y = data_dict["y"]
            clusters = data_dict.get("clusters")
            offset = data_dict.get("offset")
            wide_df = data_dict.get("wide_df")

            if wide_df is None:
                raise KeyError(
                    "prepare_group_data() did not return 'wide_df'. "
                    "Cannot align OOF predictions back to deal_id."
                )

            if id_col not in wide_df.columns:
                raise KeyError(
                    f"id_col='{id_col}' not found in wide_df. "
                    f"Available columns include: {list(wide_df.columns[:20])}"
                )

            # Critical alignment step:
            # X.index is inherited from the prepared wide_df rows after dropna().
            obs_ids = wide_df.loc[X.index, id_col]

            if obs_ids.duplicated().any():
                duplicated = obs_ids[obs_ids.duplicated()].unique()[:10]
                raise ValueError(
                    f"Duplicate observation IDs after model preparation: {duplicated}. "
                    "Expected one row per deal in the prepared model matrix."
                )

            if clusters is not None:
                n_clusters = pd.Series(clusters).nunique()
                if n_clusters < n_splits:
                    print(
                        f"Skipping {model_name} for {group_keys}: "
                        f"only {n_clusters} clusters for {n_splits} folds."
                    )
                    continue

                splitter = GroupKFold(n_splits=n_splits)
                split_iterator = splitter.split(X, y, groups=clusters)
            else:
                splitter = KFold(n_splits=n_splits,
                                 shuffle=True, random_state=42)
                split_iterator = splitter.split(X, y)

            y_pred_oof = pd.Series(np.nan, index=X.index,
                                   name="OOF_prediction")

            for fold, (train_idx, test_idx) in enumerate(split_iterator):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]

                offset_train = offset.iloc[train_idx] if offset is not None else None
                offset_test = offset.iloc[test_idx] if offset is not None else None
                clusters_train = clusters.iloc[train_idx] if clusters is not None else None

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        # Use the runner's own model dispatch so this mirrors
                        # your in-sample architecture.
                        res = dummy_runner._fit_model(
                            y=y_train,
                            X=X_train,
                            offset=offset_train,
                            clusters=clusters_train,
                        )

                    if offset_test is not None:
                        fold_pred = res.predict(X_test, offset=offset_test)
                    else:
                        fold_pred = res.predict(X_test)

                    y_pred_oof.iloc[test_idx] = np.asarray(fold_pred)

                except Exception as e:
                    print(
                        f"  Fold {fold} failed for {model_name}, {group_keys}: {e}")
                    continue

            valid = y_pred_oof.notna() & y.notna()

            if valid.sum() == 0:
                print(
                    f"Skipping {model_name} for {group_keys}: no valid OOF predictions.")
                continue

            y_true_valid = y.loc[valid].to_numpy()
            y_pred_valid = y_pred_oof.loc[valid].to_numpy()

            rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
            mae = mean_absolute_error(y_true_valid, y_pred_valid)

            if (y_pred_valid > 0).all() and (y_true_valid >= 0).all():
                poisson_dev = mean_poisson_deviance(y_true_valid, y_pred_valid)
            else:
                poisson_dev = np.nan

            if len(np.unique(y_true_valid)) > 1 and len(np.unique(y_pred_valid)) > 1:
                pearson = pearsonr(y_true_valid, y_pred_valid)[0]
                spearman = spearmanr(y_true_valid, y_pred_valid)[0]
                kendall = kendalltau(y_true_valid, y_pred_valid)[0]
            else:
                pearson = np.nan
                spearman = np.nan
                kendall = np.nan

            metric_rows.append({
                **group_dict,
                "Experiment Group": experiment_group,
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "Poisson Dev": poisson_dev,
                "Pearson r": pearson,
                "Spearman Rho": spearman,
                "Kendall Tau": kendall,
                "nobs": int(valid.sum()),
            })

            oof_part = pd.DataFrame({
                **group_dict,
                "Experiment Group": experiment_group,
                "Model": model_name,
                id_col: obs_ids.loc[valid].to_numpy(),
                "OOF_prediction": y_pred_oof.loc[valid].to_numpy(),
                "OOF_target": y.loc[valid].to_numpy(),
            })

            oof_part["absolute_error"] = (
                oof_part["OOF_prediction"] - oof_part["OOF_target"]
            ).abs()

            oof_rows.append(oof_part)

            print(f"✓ {model_name} CV complete.")

    metrics_df = pd.DataFrame(metric_rows)
    oof_df = pd.concat(
        oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()

    return metrics_df, oof_df


def cross_validate_barter_baseline_with_oof(
    df: pd.DataFrame,
    target_col: str,
    cluster_col: str,
    cat_vars: list,
    bin_vars: list,
    num_vars: list,
    n_splits: int = 5,
    id_col: str = "deal_id",
    model_type: str = "negbin",
    offset_col: str | None = None,
):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import warnings

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_poisson_deviance,
    )
    from scipy.stats import pearsonr, spearmanr, kendalltau

    from src.barter.regression_models import create_regression_runner

    # One baseline row per deal. Use the same structural variables only.
    keep_cols = (
        [id_col, target_col, cluster_col]
        + cat_vars
        + bin_vars
        + num_vars
        + ([offset_col] if offset_col is not None else [])
    )
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in df.columns]

    deal_df = (
        df[keep_cols]
        .drop_duplicates(subset=id_col)
        .copy()
    )

    # Use a runner with no experimental grouping: one global baseline.
    runner = create_regression_runner(
        target_col=target_col,
        experimental_groups=[],
        cluster_col=cluster_col,
        offset_col=offset_col,
        model_type=model_type,
    )

    data_dict = runner.prepare_group_data(
        group_df=deal_df,
        cat_predictors=cat_vars,
        bin_predictors=bin_vars,
        num_predictors=num_vars,
        new_predictors=None,
    )

    if data_dict is None or data_dict.get("X") is None:
        raise RuntimeError("Baseline data preparation failed.")

    X = data_dict["X"]
    y = data_dict["y"]
    clusters = data_dict.get("clusters")
    offset = data_dict.get("offset")
    wide_df = data_dict.get("wide_df")

    obs_ids = wide_df.loc[X.index, id_col]

    splitter = GroupKFold(n_splits=n_splits)
    split_iterator = splitter.split(X, y, groups=clusters)

    y_pred_oof = pd.Series(np.nan, index=X.index, name="OOF_prediction")

    for fold, (train_idx, test_idx) in enumerate(split_iterator):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        offset_train = offset.iloc[train_idx] if offset is not None else None
        offset_test = offset.iloc[test_idx] if offset is not None else None
        clusters_train = clusters.iloc[train_idx] if clusters is not None else None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = runner._fit_model(
                    y=y_train,
                    X=X_train,
                    offset=offset_train,
                    clusters=clusters_train,
                )

            if offset_test is not None:
                pred = res.predict(X_test, offset=offset_test)
            else:
                pred = res.predict(X_test)

            y_pred_oof.iloc[test_idx] = np.asarray(pred)

        except Exception as e:
            print(f"Baseline fold {fold} failed: {e}")

    valid = y_pred_oof.notna() & y.notna()

    y_true = y.loc[valid].to_numpy()
    y_pred = y_pred_oof.loc[valid].to_numpy()

    metrics = {
        "Experiment Group": "Baseline",
        "Model": "Baseline",
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Poisson Dev": (
            mean_poisson_deviance(y_true, y_pred)
            if (y_pred > 0).all() and (y_true >= 0).all()
            else np.nan
        ),
        "Pearson r": pearsonr(y_true, y_pred)[0] if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else np.nan,
        "Spearman Rho": spearmanr(y_true, y_pred)[0] if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else np.nan,
        "Kendall Tau": kendalltau(y_true, y_pred)[0] if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else np.nan,
        "nobs": int(valid.sum()),
    }

    oof = pd.DataFrame({
        id_col: obs_ids.loc[valid].to_numpy(),
        "Experiment Group": "Baseline",
        "Model": "Baseline",
        "OOF_prediction": y_pred_oof.loc[valid].to_numpy(),
        "OOF_target": y.loc[valid].to_numpy(),
    })

    oof["absolute_error"] = (oof["OOF_prediction"] - oof["OOF_target"]).abs()

    return pd.DataFrame([metrics]), oof
