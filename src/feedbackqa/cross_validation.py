import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata
from typing import Dict, Optional


class FeedbackQACrossValidator:
    """
    Executes grouped cross-validation for models fitted via FeedbackQARegressionRunner.
    Enforces strict observation pairing via Document-Rater composite identifiers.
    """

    def __init__(self, runner) -> None:
        if not runner.run_data:
            raise ValueError(
                "Runner contains no fitted data. Execute run_regression() first.")
        self.runner = runner
        # Cache for paired out-of-fold predictions mapped by deterministic identifiers
        self.oof_predictions: Dict[str, pd.DataFrame] = {}

    def evaluate_models(self, n_splits: int = 5) -> pd.DataFrame:
        """
        Performs out-of-fold prediction, maps to composite index, and calculates metrics.
        """
        results = []

        for label, data in self.runner.run_data.items():
            X = data["X"]
            y = data["y"]
            long_df = data["long_df"]
            clusters = data.get("clusters")

            # 1. Construct deterministic observation identifiers for strict pairing
            input_col = self.runner.input_col
            if "rater_id" in long_df.columns:
                obs_idx = long_df[input_col].astype(
                    str) + "_" + long_df["rater_id"].astype(str)
            else:
                obs_idx = long_df[input_col].astype(str)

            y_true_series = pd.Series(y.values, index=obs_idx, name='y_true')
            y_pred_series = pd.Series(np.nan, index=obs_idx, name='y_pred')

            # 2. Setup cross-validation
            if clusters is not None:
                splitter = GroupKFold(n_splits=n_splits)
                split_gen = splitter.split(X, y, groups=clusters)
            else:
                splitter = KFold(n_splits=n_splits,
                                 shuffle=True, random_state=42)
                split_gen = splitter.split(X, y)

            for train_idx, test_idx in split_gen:
                X_train, X_test = X.iloc[train_idx].copy(
                ), X.iloc[test_idx].copy()
                y_train = y.iloc[train_idx]

                X_train_const = sm.add_constant(X_train, has_constant="add")
                X_test_const = sm.add_constant(X_test, has_constant="add")

                # Align columns
                missing_cols = set(X_train_const.columns) - \
                    set(X_test_const.columns)
                for c in missing_cols:
                    X_test_const[c] = 0
                X_test_const = X_test_const[X_train_const.columns]

                # Fit and predict
                model = sm.OLS(y_train, X_train_const).fit()
                y_pred_series.iloc[test_idx] = model.predict(
                    X_test_const).values

            # Cache the index-aligned predictions
            self.oof_predictions[label] = pd.concat(
                [y_true_series, y_pred_series], axis=1)

            # Calculate metrics for the model on its available data
            metrics = self._calculate_metrics(
                y_true_series.to_numpy(), y_pred_series.to_numpy())
            metrics["label"] = label
            metrics["nobs"] = len(y_true_series)
            results.append(metrics)

        return pd.DataFrame(results).set_index("label")[
            ['RMSE', 'Pearson_r', 'Spearman_rho', 'Kendall_tau', 'nobs']
        ]

    def test_structural_delta(
        self,
        base_label: str,
        test_label: str,
        metric: str = 'spearman',
        n_permutations: int = 10000,
        seed: int = 42
    ) -> float:
        """
        Executes an exact paired permutation test. Uses an inner join to mathematically
        guarantee that asymmetric missingness does not misalign the arrays.
        """
        if base_label not in self.oof_predictions or test_label not in self.oof_predictions:
            raise KeyError(
                "Specified labels not found in cache. Execute evaluate_models() first.")

        df_base = self.oof_predictions[base_label]
        df_test = self.oof_predictions[test_label]

        # Inner join strictly enforces observation pairing and drops asymmetric NAs
        merged = df_base.join(df_test, lsuffix='_base',
                              rsuffix='_test', how='inner')

        y_true = merged['y_true_base'].to_numpy()
        y_pred_base = merged['y_pred_base'].to_numpy()
        y_pred_test = merged['y_pred_test'].to_numpy()

        np.random.seed(seed)
        n_obs = len(y_true)

        # 1. Establish the observed test statistic
        if metric == 'pearson':
            obs_base, _ = pearsonr(y_true, y_pred_base)
            obs_test, _ = pearsonr(y_true, y_pred_test)
        elif metric == 'spearman':
            obs_base, _ = spearmanr(y_true, y_pred_base)
            obs_test, _ = spearmanr(y_true, y_pred_test)
        else:
            raise ValueError("Strictly supports 'pearson' or 'spearman'.")

        obs_diff = obs_test - obs_base

        if obs_diff <= 0:
            return 1.0

        # 2. Pre-compute constants for fast dot-product correlations
        y_true_calc = rankdata(y_true) if metric == 'spearman' else y_true
        y_true_centered = y_true_calc - np.mean(y_true_calc)
        y_true_norm = np.linalg.norm(y_true_centered)

        stacked_preds = np.vstack([y_pred_base, y_pred_test])
        count_extreme = 0

        # 3. Execute fast permutations
        for _ in range(n_permutations):
            swap_idx = np.random.randint(0, 2, size=n_obs)

            perm_base = stacked_preds[swap_idx, np.arange(n_obs)]
            perm_test = stacked_preds[1 - swap_idx, np.arange(n_obs)]

            perm_base_calc = rankdata(
                perm_base) if metric == 'spearman' else perm_base
            perm_test_calc = rankdata(
                perm_test) if metric == 'spearman' else perm_test

            base_c = perm_base_calc - np.mean(perm_base_calc)
            test_c = perm_test_calc - np.mean(perm_test_calc)

            r_base = np.dot(y_true_centered, base_c) / \
                (y_true_norm * np.linalg.norm(base_c))
            r_test = np.dot(y_true_centered, test_c) / \
                (y_true_norm * np.linalg.norm(test_c))

            if (r_test - r_base) >= obs_diff:
                count_extreme += 1

        return (count_extreme + 1) / (n_permutations + 1)

    def compute_human_baseline(self, df: pd.DataFrame, score_col_1: str = 'score_1', score_col_2: str = 'score_2') -> pd.Series:
        """
        Computes the human-human alignment baseline from wide-format annotations.
        """
        valid_pairs = df[[score_col_1, score_col_2]].dropna()
        y_1 = valid_pairs[score_col_1].to_numpy()
        y_2 = valid_pairs[score_col_2].to_numpy()

        if len(y_1) == 0:
            return pd.Series({
                "RMSE": np.nan,
                "Pearson_r": np.nan,
                "Spearman_rho": np.nan,
                "Kendall_tau": np.nan,
                "nobs": 0
            })

        metrics = self._calculate_metrics(y_1, y_2)
        metrics["nobs"] = len(y_1)
        return pd.Series(metrics)

    # @staticmethod
    # def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    #     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    #     if np.std(y_pred) == 0 or np.std(y_true) == 0:
    #         r, rho = np.nan, np.nan
    #     else:
    #         r, _ = pearsonr(y_true, y_pred)
    #         rho, _ = spearmanr(y_true, y_pred)
    #     return {"RMSE": rmse, "Pearson_r": r, "Spearman_rho": rho}

    # def batch_test_structural_delta(
    #     self,
    #     base_suffix: str,
    #     test_suffix: str,
    #     metric: str = 'spearman',
    #     n_permutations: int = 10000,
    #     seed: int = 42
    # ) -> pd.DataFrame:
    #     """
    #     Auto-discovers model families and executes paired permutation tests across the grid.

    #     Parameters:
    #     -----------
    #     base_suffix : str
    #         The prompt suffix of the baseline (e.g., 'Holistic Informed').
    #     test_suffix : str
    #         The prompt suffix of the test condition (e.g., 'Formative').
    #     metric : str
    #         The correlation metric to test ('spearman' or 'pearson').

    #     Returns:
    #     --------
    #     pd.DataFrame
    #         A matrix of p-values and boolean significance flags for easy table integration.
    #     """
    #     results = []

    #     # Auto-discover unique model families (assuming format "ModelName_Condition")
    #     all_labels = list(self.oof_predictions.keys())
    #     model_families = sorted(
    #         list(set([label.split("_")[0] for label in all_labels])))

    #     for model in model_families:
    #         base_label = f"{model}_{base_suffix}"
    #         test_label = f"{model}_{test_suffix}"

    #         if base_label in self.oof_predictions and test_label in self.oof_predictions:
    #             p_val = self.test_structural_delta(
    #                 base_label=base_label,
    #                 test_label=test_label,
    #                 metric=metric,
    #                 n_permutations=n_permutations,
    #                 seed=seed
    #             )

    #             results.append({
    #                 "Model": model,
    #                 "Base": base_suffix,
    #                 "Test": test_suffix,
    #                 f"p_value_{metric}": p_val,
    #                 "Significant (p<0.05)": "*" if p_val < 0.05 else ""
    #             })
    #         else:
    #             print(
    #                 f"Warning: Missing keys for {model}. Check suffixes '{base_suffix}' and '{test_suffix}'.")

    #     return pd.DataFrame(results).set_index("Model")

    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            return {
                "RMSE": np.nan,
                "Pearson_r": np.nan,
                "Spearman_rho": np.nan,
                "Kendall_tau": np.nan
            }

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            r, rho, tau = np.nan, np.nan, np.nan
        else:
            r, _ = pearsonr(y_true, y_pred)
            rho, _ = spearmanr(y_true, y_pred)
            tau, _ = kendalltau(y_true, y_pred)

        return {
            "RMSE": rmse,
            "Pearson_r": r,
            "Spearman_rho": rho,
            "Kendall_tau": tau
        }
