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
            obs_idx = self._make_observation_ids(long_df)
            # input_col = self.runner.input_col
            # if "rater_id" in long_df.columns:
            #     obs_idx = long_df[input_col].astype(
            #         str) + "_" + long_df["rater_id"].astype(str)
            # else:
            #     obs_idx = long_df[input_col].astype(str)

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

    def _make_observation_ids(self, long_df: pd.DataFrame) -> pd.Index:
        """
        Constructs deterministic, unique observation identifiers for OOF pairing.
        The base identifier preserves the document-rater structure, while the row
        suffix prevents accidental collisions when duplicated artefacts occur.
        """
        input_col = self.runner.input_col

        if input_col not in long_df.columns:
            raise KeyError(f"Input column '{input_col}' not found in long_df.")

        base = long_df[input_col].astype(str)

        if "rater_id" in long_df.columns:
            base = base + "__rater=" + long_df["rater_id"].astype(str)

        row_component = pd.Series(
            long_df.index.to_numpy(),
            index=long_df.index
        ).astype(str)

        obs_ids = base + "__row=" + row_component
        return pd.Index(obs_ids.to_numpy(), name="_obs_id")

    def get_oof_predictions_long(
        self,
        label: str,
        pred_col: str = "OOF_prediction"
    ) -> pd.DataFrame:
        """
        Returns the annotation-/observation-level dataset with out-of-fold
        predictions attached.

        This is the exact level at which cross-validated metrics are computed.
        For FeedbackQA, this means one row per document-rater observation.
        """
        if label not in self.oof_predictions:
            raise KeyError(
                f"Label '{label}' not found in OOF cache. "
                "Execute evaluate_models() first."
            )

        if label not in self.runner.run_data:
            raise KeyError(f"Label '{label}' not found in runner.run_data.")

        long_df = self.runner.run_data[label]["long_df"].copy()
        obs_ids = self._make_observation_ids(long_df)

        out = long_df.copy()
        out["_obs_id"] = obs_ids.to_numpy()

        oof = (
            self.oof_predictions[label]
            .rename(columns={"y_pred": pred_col})
            .reset_index()
            .rename(columns={"index": "_obs_id"})
        )

        out = out.merge(
            oof[["_obs_id", "y_true", pred_col]],
            on="_obs_id",
            how="left",
            validate="one_to_one"
        )

        return out

    def build_oof_prediction_dataset(
        self,
        label: str,
        original_df: Optional[pd.DataFrame] = None,
        id_col: Optional[str] = None,
        pred_col: str = "OOF_prediction",
        target_col: str = "OOF_target",
        aggregate: str = "mean",
        include_n_observations: bool = True
    ) -> pd.DataFrame:
        """
        Builds an artefact-level dataset with OOF predictions aligned to the
        original dataset.

        Parameters
        ----------
        label:
            Model/condition label stored in runner.run_data and oof_predictions.
        original_df:
            Optional original wide-format dataset. If provided, predictions are
            left-joined onto it. If omitted, a minimal artefact-level dataset is
            constructed from long_df.
        id_col:
            Artefact-level key used for alignment. If omitted, runner.input_col
            is used. Prefer a stable document/deal ID if available.
        pred_col:
            Name of the created OOF prediction column.
        target_col:
            Name of the aggregated target column.
        aggregate:
            Aggregation used when multiple validation rows correspond to the same
            artefact. For FeedbackQA, 'mean' is appropriate because each document
            has two human annotations.
        include_n_observations:
            Whether to include the number of validation rows per artefact.

        Returns
        -------
        pd.DataFrame
            Artefact-level dataframe containing the OOF prediction column.
        """
        long_oof = self.get_oof_predictions_long(label, pred_col=pred_col)

        if id_col is None:
            id_col = self.runner.input_col

        if id_col not in long_oof.columns:
            raise KeyError(f"id_col '{id_col}' not found in OOF long dataset.")

        grouped = long_oof.groupby(id_col, dropna=False)

        agg_kwargs = {
            pred_col: (pred_col, aggregate),
            target_col: ("y_true", aggregate),
        }

        if include_n_observations:
            agg_kwargs["OOF_n_observations"] = (pred_col, "size")

        artefact_oof = grouped.agg(**agg_kwargs).reset_index()

        if original_df is None:
            base = long_oof[[id_col]].drop_duplicates().copy()
        else:
            if id_col not in original_df.columns:
                raise KeyError(f"id_col '{id_col}' not found in original_df.")
            base = original_df.copy()

        out = base.merge(
            artefact_oof,
            on=id_col,
            how="left",
            validate="many_to_one"
        )

        return out
