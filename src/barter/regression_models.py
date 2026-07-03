import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.genmod.generalized_linear_model import GLMResults, GLMResultsWrapper
from statsmodels.discrete.discrete_model import NegativeBinomialResultsWrapper, CountResultsWrapper
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing import List, Dict, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# =============================================================================
# STRUCTURED RESULT CONTAINER
# =============================================================================


@dataclass
class RunRecord:
    """
    A single fitted-model record.

    Keeps the original group key tuple intact so downstream code never needs
    to parse a concatenated string to recover experimental-group membership.

    Attributes
    ----------
    group_key   : tuple of scalar values, one per experimental group dimension
    label       : human-readable string used for display / plot titles
    result      : fitted statsmodels result object
    y           : dependent-variable series used for this fit
    X           : design matrix used for this fit
    long_df     : the pre-model data frame (wide pivoted form)
    iv_cols     : list of predictor column names (excluding 'const')
    clusters    : cluster series (None when not clustering)
    y_pred      : in-sample predicted values from the fitted model
    """
    group_key: tuple
    label: str
    result: Any
    y: pd.Series
    X: pd.DataFrame
    long_df: pd.DataFrame
    iv_cols: List[str]
    clusters: Optional[pd.Series] = None
    y_pred: Optional[pd.Series] = None

    def as_run_data_dict(self) -> Dict[str, Any]:
        """Convenience accessor for backward compatibility."""
        return {
            "y": self.y,
            "y_pred": self.y_pred,
            "X": self.X,
            "long_df": self.long_df,
            "iv_cols": self.iv_cols,
            "clusters": self.clusters,
        }

# =============================================================================
# BASE CLASS
# =============================================================================


class BaseRegressionRunner(ABC):
    """Base class for regression analysis with robust error handling."""

    def __init__(
        self,
        target_col: str,
        experimental_groups: List[str],
        offset_col: Optional[str] = None,
    ):
        self.target_col = target_col
        self.experimental_groups = experimental_groups
        self.offset_col = offset_col          # pre-computed log(offset) column

        # All fitted models live here as structured RunRecord objects.
        self._records: List[RunRecord] = []

    # ------------------------------------------------------------------ #
    # Backward-compatible properties                                       #
    # ------------------------------------------------------------------ #

    @property
    def results(self) -> Dict[str, Any]:
        """Legacy dict: label -> statsmodels result object."""
        return {r.label: r.result for r in self._records}

    @property
    def run_data(self) -> Dict[str, Dict[str, Any]]:
        """Legacy dict: label -> run-data dict."""
        return {r.label: r.as_run_data_dict() for r in self._records}

    @property
    def records(self) -> List[RunRecord]:
        """Direct access to the structured record list."""
        return self._records

    # ------------------------------------------------------------------ #
    # Tidy-frame export                                                    #
    # ------------------------------------------------------------------ #

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy DataFrame with one row per fitted model, including fit metrics."""
        rows = []
        for rec in self._records:
            r = rec.result
            row: Dict[str, Any] = {}

            # group_key is () for the baseline model
            for dim, val in zip(self.experimental_groups, rec.group_key):
                row[dim] = val

            row["label"] = rec.label
            row["nobs"] = getattr(r, "nobs", np.nan)
            row["r_squared"] = getattr(r, "rsquared", np.nan)
            row["adj_r_squared"] = getattr(r, "rsquared_adj", np.nan)
            row["pseudo_r_squared"] = getattr(r, "prsquared", np.nan)
            row["aic"] = getattr(r, "aic", np.nan)
            row["bic"] = getattr(r, "bic", np.nan)
            row["llf"] = getattr(r, "llf", np.nan)
            row["condition_number"] = getattr(r, "condition_number", np.nan)

            # Predictive Validity Metrics
            y, y_pred = rec.y, rec.y_pred
            if y is not None and y_pred is not None and len(y) > 1:
                row["mse"] = np.mean((y - y_pred)**2)
                row["mae"] = np.mean(np.abs(y - y_pred))

                # Suppress constant input warnings for edge cases
                try:
                    row["pearson_r"] = stats.pearsonr(y, y_pred)[0]
                    row["spearman_rho"] = stats.spearmanr(y, y_pred)[0]
                    row['kendall_tau'] = stats.kendalltau(y, y_pred)[0]
                except Exception:
                    row["pearson_r"] = np.nan
                    row["spearman_rho"] = np.nan
                    row['kendall_tau'] = np.nan
            else:
                row["mse"] = row["mae"] = row["pearson_r"] = row["spearman_rho"] = row['kendall_tau'] = np.nan

            row["result"] = r
            row["record"] = rec
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Internal helpers: pivoting and design matrix                       #
    # ------------------------------------------------------------------ #

    def _pivot_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        score_cols: Union[str, List[str], None],
        extra_index_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Pivot LLM scores from long to wide format using deal_id as the minimal index,
        then merge all other columns back. Supports multiple score columns.
        """
        if score_cols is None:
            return group_df.copy(), []

        if isinstance(score_cols, str):
            score_cols = [score_cols]

        unique_dims = group_df["dimension_name"].unique()

        side_cols = (
            self.experimental_groups
            + cat_predictors
            + bin_predictors
            + num_predictors
            + [self.target_col, "deal_text"]
        )
        if self.offset_col:
            side_cols.append(self.offset_col)
        if extra_index_cols:
            side_cols.extend(extra_index_cols)
        side_cols = list(dict.fromkeys(
            c for c in side_cols if c in group_df.columns
        ))

        # Pivot multiple metrics over dimensions
        pivot_df = (
            group_df
            .pivot_table(
                index="deal_id",
                columns="dimension_name",
                values=score_cols,
                aggfunc="first",
            )
        )

        # Flatten MultiIndex if multiple score_cols are present: (metric, dimension) -> "dimension_metric"
        if isinstance(pivot_df.columns, pd.MultiIndex):
            new_cols = [f"{dim}_{met}" for met, dim in pivot_df.columns]
            pivot_df.columns = new_cols
        elif len(unique_dims) == 1 and unique_dims[0] == "quality":
            # If it's not a MultiIndex and there's only one dimension "quality"
            pivot_df.columns = [f"quality_{c}" for c in pivot_df.columns]

        pivot_df = pivot_df.reset_index()
        score_cols_out = [c for c in pivot_df.columns if c != "deal_id"]

        side_df = (
            group_df[["deal_id"] + side_cols]
            .drop_duplicates(subset="deal_id")
        )

        wide_df = (
            pivot_df
            .merge(side_df, on="deal_id", how="inner")
            .dropna(subset=score_cols_out)
        )
        return wide_df, score_cols_out

    def _build_predictor_matrix(
        self,
        wide_df: pd.DataFrame,
        score_cols: List[str],
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
    ) -> Optional[pd.DataFrame]:
        """Build the X matrix and check for rank deficiency."""
        import scipy.linalg

        predictor_parts = []

        numeric_cols = score_cols + bin_predictors + num_predictors
        if numeric_cols:
            X_num = wide_df[numeric_cols].copy()
            for col in bin_predictors:
                X_num[col] = X_num[col].astype(int)
            predictor_parts.append(X_num)

        if cat_predictors:
            cat_subset = wide_df[cat_predictors].copy().astype(str)
            X_cat = pd.get_dummies(cat_subset, drop_first=True, dtype=int)
            predictor_parts.append(X_cat)

        if not predictor_parts:
            return None

        X = sm.add_constant(pd.concat(predictor_parts, axis=1))

        try:
            X_clean = X.astype(float)
        except ValueError as e:
            print(f"⚠️  Warning: Could not cast matrix to float. {e}")
            return None

        try:
            if np.linalg.matrix_rank(X_clean.values) < X_clean.shape[1]:
                print(
                    "⚠️  Warning: Design matrix is rank-deficient (perfect multicollinearity)")
                return None
        except np.linalg.LinAlgError:
            try:
                _, s, _ = scipy.linalg.svd(
                    X_clean.values, full_matrices=False, lapack_driver="gesvd")
                tol = X_clean.values.max() * max(X_clean.shape) * np.finfo(float).eps
                if np.count_nonzero(s > tol) < X_clean.shape[1]:
                    print(
                        "⚠️  Warning: Design matrix is rank-deficient (perfect multicollinearity)")
                    return None
            except Exception as e:
                print(f"🚨 Both SVD solvers failed. {e}")
                return None

        return X_clean

    # ------------------------------------------------------------------ #
    # Abstract interface                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def prepare_group_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        new_predictors: Union[str, List[str], None],
        dims_to_exclude: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def _fit_model(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        offset: Optional[pd.Series] = None,
        **kwargs,
    ) -> Any:
        pass

    @property
    def dep_var(self) -> str:
        return self.target_col

    # ------------------------------------------------------------------ #
    # Internal: build a RunRecord and append it                          #
    # ------------------------------------------------------------------ #

    def _store_record(
        self,
        group_key: tuple,
        label: str,
        result: Any,
        y: pd.Series,
        X: pd.DataFrame,
        wide_df: pd.DataFrame,
        clusters: Optional[pd.Series],
    ) -> None:

        # Force extraction in the native response scale (count space)
        try:
            # result.predict() applies the inverse log-link function automatically
            y_pred = result.predict()
            if y_pred is not None:
                y_pred = pd.Series(y_pred, index=y.index)
        except Exception as e:
            print(f"Prediction extraction failed for {label}: {e}")
            y_pred = None

        rec = RunRecord(
            group_key=group_key,
            label=label,
            result=result,
            y=y,
            X=X,
            long_df=wide_df,
            iv_cols=[c for c in X.columns if c != "const"],
            clusters=clusters,
            y_pred=y_pred,
        )
        self._records.append(rec)
    # ------------------------------------------------------------------ #
    # Public fitting methods                                             #
    # ------------------------------------------------------------------ #

    def run_baseline_model(
        self,
        df: pd.DataFrame,
        cat_vars: List[str],
        bin_vars: List[str],
        num_vars: List[str],
        label: str = "baseline",
    ) -> "BaseRegressionRunner":
        """Fit a single baseline model without LLM predictors."""

        data_dict = self.prepare_group_data(
            df, cat_vars, bin_vars, num_vars,
            new_predictors=None,
            dims_to_exclude=None,
        )
        if data_dict is None:
            print("Baseline model: Data preparation failed.")
            return self

        y, X = data_dict.get("y"), data_dict.get("X")
        if y is None or X is None or len(y) < (X.shape[1] + 2):
            print("Baseline model: Missing data or insufficient observations.")
            return self

        offset = data_dict.get("offset")
        clusters = data_dict.get("clusters")
        wide_df = data_dict.get("wide_df", X.join(y))

        try:
            res = self._fit_model(y=y, X=X, offset=offset, clusters=clusters)
            if hasattr(res, "converged") and not res.converged:
                print("⚠️  Warning: Baseline model did not converge.")

            self._store_record(
                group_key=tuple("BASELINE" for _ in self.experimental_groups),
                label=label,
                result=res,
                y=y,
                X=X,
                wide_df=wide_df,
                clusters=clusters,
            )
            metric = f"AIC={res.aic:.2f}" if hasattr(
                res, "aic") else f"R²={res.rsquared:.3f}"
            print(
                f"✓ Baseline model fitted: {res.nobs} observations, {metric}")

        except Exception as e:
            print(f"Error fitting baseline model: {e}")

        return self

    def run_regression(
        self,
        df: pd.DataFrame,
        cat_vars: List[str],
        bin_vars: List[str],
        num_vars: List[str],
        predictor_names: Union[str, List[str]],
        label_map: Optional[Dict[str, str]] = None,
        dims_to_exclude: Optional[List[str]] = None,
    ) -> "BaseRegressionRunner":
        """Main execution loop for LLM-based models."""

        grouped = df.groupby(self.experimental_groups)

        for group_keys, group_df in grouped:
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_key_tuple = tuple(str(k) for k in group_keys)

            data_dict = self.prepare_group_data(
                group_df, cat_vars, bin_vars, num_vars,
                predictor_names, dims_to_exclude,
            )
            if data_dict is None:
                continue

            y, X = data_dict.get("y"), data_dict.get("X")
            if y is None or X is None or len(y) < (X.shape[1] + 2):
                print(
                    f"Skipping {group_keys}: Missing data or insufficient observations.")
                continue

            offset = data_dict.get("offset")
            clusters = data_dict.get("clusters")
            wide_df = data_dict.get("wide_df", X.join(y))

            if label_map and len(group_key_tuple) >= 2:
                label = f"{group_key_tuple[0]} {label_map.get(group_key_tuple[1], group_key_tuple[1])}"
            else:
                label = " ".join(group_key_tuple)

            try:
                res = self._fit_model(
                    y=y, X=X, offset=offset, clusters=clusters)
                if hasattr(res, "converged") and not res.converged:
                    print(
                        f"⚠️  Warning: Model for {group_keys} did not converge.")

                self._store_record(
                    group_key=group_key_tuple,
                    label=label,
                    result=res,
                    y=y,
                    X=X,
                    wide_df=wide_df,
                    clusters=clusters,
                )

            except Exception as e:
                print(f"Error fitting model for {group_keys}: {e}")

        return self

    def summarize(self) -> pd.DataFrame:
        """Return a tidy DataFrame of fit statistics for all stored results."""
        df = self.to_frame().drop(
            columns=["result", "record"], errors="ignore")
        sort_col = "adj_r_squared" if df["adj_r_squared"].notna(
        ).any() else "aic"
        ascending = sort_col == "aic"
        return df.sort_values(sort_col, ascending=ascending)

    run_negative_binomial = run_regression

    # =============================================================================
    # VARIANCE INFLATION FACTORS FUNCTION (UPDATED)
    # =============================================================================

    # =============================================================================
    # VARIANCE INFLATION FACTORS
    # =============================================================================

    def _build_vif_dataframe(self, groups: Dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame:
        """
        Internal helper to compute and format VIFs for a dictionary of design matrices.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        rows = []
        for label, X in groups.items():
            X = X.dropna()

            # Separate true predictors from the automatically generated intercept column
            predictor_cols = [c for c in X.columns if c != "const"]

            if len(predictor_cols) < 2:
                # VIF is mathematically undefined with fewer than 2 continuous predictors
                for col in predictor_cols:
                    rows.append({
                        "group": label,
                        "predictor": col,
                        "vif": float("nan"),
                        "flagged": False
                    })
                continue

            X_arr = X.values.astype(float)

            for col in predictor_cols:
                idx = X.columns.tolist().index(col)
                try:
                    vif_val = float(variance_inflation_factor(X_arr, idx))
                    is_flagged = vif_val > threshold
                    vif_rounded = round(vif_val, 3) if np.isfinite(
                        vif_val) else vif_val
                except Exception:
                    vif_rounded = float("nan")
                    is_flagged = False

                rows.append({
                    "group": label,
                    "predictor": col,
                    "vif": vif_rounded,
                    "flagged": is_flagged,
                })

        return (
            pd.DataFrame(rows, columns=[
                         "group", "predictor", "vif", "flagged"])
            .sort_values(["group", "vif"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def compute_vifs(
        self,
        df: Optional[pd.DataFrame] = None,
        cat_vars: Optional[List[str]] = None,
        bin_vars: Optional[List[str]] = None,
        num_vars: Optional[List[str]] = None,
        predictor_names: Optional[Union[str, List[str]]] = None,
        label_map: Optional[Dict[str, str]] = None,
        dims_to_exclude: Optional[List[str]] = None,
        run_labels: Optional[List[str]] = None,
        threshold: float = 5.0,
    ) -> pd.DataFrame:
        """
        Compute Variance Inflation Factors for the experimental LLM groups.
        """
        groups: Dict[str, pd.DataFrame] = {}

        if df is not None:
            if cat_vars is None or bin_vars is None or num_vars is None:
                raise ValueError(
                    "When passing df= to compute_vifs on the fly, you must also provide "
                    "cat_vars=, bin_vars=, and num_vars= configuration lists."
                )

            grouped = df.groupby(self.experimental_groups)
            for group_keys, group_df in grouped:
                if not isinstance(group_keys, tuple):
                    group_keys = (group_keys,)
                group_key_tuple = tuple(str(k) for k in group_keys)

                data = self.prepare_group_data(
                    group_df, cat_vars, bin_vars, num_vars,
                    predictor_names, dims_to_exclude
                )
                if data is not None and data.get("X") is not None:
                    if label_map and len(group_key_tuple) >= 2:
                        label = f"{group_key_tuple[0]} {label_map.get(group_key_tuple[1], group_key_tuple[1])}"
                    else:
                        label = " ".join(group_key_tuple)
                    groups[label] = data["X"]
        else:
            if not self._records:
                raise RuntimeError(
                    "No stored runs found. Call run_regression() first, "
                    "or pass df= along with variable lists to compute VIFs on the fly."
                )
            labels = run_labels if run_labels is not None else list(
                self.results.keys())
            groups = {
                r.label: r.X for r in self._records if r.label in labels and r.label != "baseline"}

        return self._build_vif_dataframe(groups, threshold)

    def compute_baseline_vifs(
        self,
        df: Optional[pd.DataFrame] = None,
        cat_vars: Optional[List[str]] = None,
        bin_vars: Optional[List[str]] = None,
        num_vars: Optional[List[str]] = None,
        threshold: float = 5.0,
        label: str = "baseline"
    ) -> pd.DataFrame:
        """
        Compute Variance Inflation Factors for the baseline model (without LLM predictors)
        over the entire dataset.
        """
        groups: Dict[str, pd.DataFrame] = {}

        if df is not None:
            if cat_vars is None or bin_vars is None or num_vars is None:
                raise ValueError(
                    "When passing df= to compute_baseline_vifs on the fly, you must also provide "
                    "cat_vars=, bin_vars=, and num_vars= configuration lists."
                )

            data = self.prepare_group_data(
                df, cat_vars, bin_vars, num_vars,
                new_predictors=None, dims_to_exclude=None
            )
            if data is not None and data.get("X") is not None:
                groups[label] = data["X"]
        else:
            baseline_record = next(
                (r for r in self._records if r.label == label), None)
            if baseline_record is None:
                raise RuntimeError(
                    f"No stored run found with label '{label}'. Call run_baseline_model() first, "
                    "or pass df= along with variable lists to compute VIFs on the fly."
                )
            groups[label] = baseline_record.X

        return self._build_vif_dataframe(groups, threshold)

    # =============================================================================
    # Cross-validation
    # =============================================================================

    def run_cross_validation(
        self,
        df: pd.DataFrame,
        cat_vars: List[str],
        bin_vars: List[str],
        num_vars: List[str],
        predictor_names: Union[str, List[str]],
        label: str,
        n_splits: int = 5,
        dims_to_exclude: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes k-fold cross-validation natively, utilizing polymorphic model dispatch
        to ensure out-of-sample metrics match the specified in-sample architecture.
        """
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
        import warnings

        grouped = df.groupby(self.experimental_groups)
        cv_results = []

        for group_keys, group_df in grouped:
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)

            data_dict = self.prepare_group_data(
                group_df, cat_vars, bin_vars, num_vars,
                predictor_names, dims_to_exclude,
            )
            if data_dict is None or data_dict.get("X") is None:
                continue

            X, y, clusters = data_dict["X"], data_dict["y"], data_dict.get(
                "clusters")
            offset = data_dict.get("offset")

            # Default to standard KFold if no clustering variable is specified
            gkf = GroupKFold(n_splits=n_splits)
            split_iterator = gkf.split(X, y, groups=clusters) if clusters is not None else \
                KFold(n_splits=n_splits, shuffle=True).split(X, y)

            fold_rmse, fold_mae, fold_poisson = [], [], []

            for train_idx, test_idx in split_iterator:
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                offset_train = offset.iloc[train_idx] if offset is not None else None
                offset_test = offset.iloc[test_idx] if offset is not None else None
                clusters_train = clusters.iloc[train_idx] if clusters is not None else None

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Polymorphic dispatch guarantees we fit the exact model specified by the factory
                        res = self._fit_model(
                            y=y_train, X=X_train, offset=offset_train, clusters=clusters_train)

                    y_pred = res.predict(X_test, offset=offset_test)

                    fold_rmse.append(
                        np.sqrt(mean_squared_error(y_test, y_pred)))
                    fold_mae.append(mean_absolute_error(y_test, y_pred))

                    if (y_pred > 0).all() and (y_test >= 0).all():
                        fold_poisson.append(
                            mean_poisson_deviance(y_test, y_pred))

                except Exception:
                    continue

            if fold_rmse:
                cv_results.append({
                    "group": group_keys,
                    "label": label,
                    "rmse_cv": np.nanmean(fold_rmse),
                    "mae_cv": np.nanmean(fold_mae),
                    "poisson_dev_cv": np.nanmean(fold_poisson) if fold_poisson else np.nan,
                })

        return cv_results


# =============================================================================
# NEGATIVE BINOMIAL IMPLEMENTATIONS (COUNT MODELS)
# =============================================================================

class StandardErrorRegression(BaseRegressionRunner):
    """Standard Negative Binomial Regression."""

    def prepare_group_data(
        self, group_df, cat_predictors, bin_predictors, num_predictors,
        new_predictors, dims_to_exclude=None,
    ):
        wide_df, score_cols = self._pivot_data(
            group_df, cat_predictors, bin_predictors, num_predictors, new_predictors
        )

        if dims_to_exclude:
            filtered_cols = []
            for col in score_cols:
                # E.g., if col is "logic_rating" and dims_to_exclude contains "logic"
                if not any(col.startswith(f"{d}_") or col == d for d in dims_to_exclude):
                    filtered_cols.append(col)
            score_cols = filtered_cols

        X = self._build_predictor_matrix(
            wide_df, score_cols, cat_predictors, bin_predictors, num_predictors
        )
        if X is None:
            return None

        y = wide_df[self.target_col]
        parts = [y, X]
        if self.offset_col and self.offset_col in wide_df.columns:
            parts.append(wide_df[self.offset_col])

        combined = pd.concat(parts, axis=1).dropna()
        if combined.empty:
            return None

        return {
            "y": combined[self.target_col],
            "X": combined[X.columns],
            "offset": combined[self.offset_col] if self.offset_col else None,
            "clusters": None,
            "wide_df": wide_df,
        }

    def _fit_model(self, y, X, offset=None, **kwargs):
        model = sm.NegativeBinomial(y, X, loglike_method="nb2", offset=offset)
        return model.fit(maxiter=2000, method="bfgs", disp=0)


class ClusteredErrorRegression(BaseRegressionRunner):
    """Clustered Negative Binomial Regression."""

    def __init__(self, target_col, experimental_groups, cluster_col, offset_col=None):
        super().__init__(target_col, experimental_groups, offset_col)
        self.cluster_col = cluster_col

    def prepare_group_data(
        self, group_df, cat_predictors, bin_predictors, num_predictors,
        new_predictors, dims_to_exclude=None,
    ):
        wide_df, score_cols = self._pivot_data(
            group_df, cat_predictors, bin_predictors, num_predictors,
            new_predictors, extra_index_cols=[self.cluster_col],
        )

        if dims_to_exclude:
            filtered_cols = []
            for col in score_cols:
                if not any(col.startswith(f"{d}_") or col == d for d in dims_to_exclude):
                    filtered_cols.append(col)
            score_cols = filtered_cols

        X = self._build_predictor_matrix(
            wide_df, score_cols, cat_predictors, bin_predictors, num_predictors
        )
        if X is None:
            return None

        y = wide_df[self.target_col]
        clusters = wide_df[self.cluster_col]
        parts = [y, X, clusters]
        if self.offset_col and self.offset_col in wide_df.columns:
            parts.append(wide_df[self.offset_col])

        combined = pd.concat(parts, axis=1).dropna()
        if combined.empty:
            return None

        return {
            "y": combined[self.target_col],
            "X": combined[X.columns],
            "clusters": combined[self.cluster_col],
            "offset": combined[self.offset_col] if self.offset_col else None,
            "wide_df": wide_df,
        }

    def _fit_model(self, y, X, offset=None, **kwargs):
        clusters = kwargs.get("clusters")
        if clusters is None:
            raise ValueError(
                "ClusteredErrorRegression requires 'clusters' argument.")
        model = sm.NegativeBinomial(y, X, loglike_method="nb2", offset=offset)
        return model.fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
            maxiter=2000,
            disp=0,
            use_t=True,
        )


# =============================================================================
# OLS IMPLEMENTATIONS (LOG-LINEAR MODELS)
# =============================================================================

class StandardOLSRegression(StandardErrorRegression):
    """Standard OLS Regression."""

    def _fit_model(self, y, X, offset=None, **kwargs):
        return sm.OLS(y, X).fit()


class ClusteredOLSRegression(ClusteredErrorRegression):
    """OLS with clustered standard errors."""

    def _fit_model(self, y, X, offset=None, **kwargs):
        clusters = kwargs.get("clusters")
        if clusters is None:
            raise ValueError(
                "ClusteredOLSRegression requires 'clusters' argument.")
        results = sm.OLS(y, X).fit()
        return results.get_robustcov_results(cov_type="cluster", groups=clusters)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_regression_runner(
    target_col: str,
    experimental_groups: List[str],
    cluster_col: Optional[str] = None,
    offset_col: Optional[str] = None,
    model_type: str = "negbin",
) -> BaseRegressionRunner:
    """
    Factory function to create the appropriate regression runner.

    Parameters
    ----------
    target_col          : Dependent variable (count for NegBin, log-transformed for OLS)
    experimental_groups : Column names that define the groupby dimensions
    cluster_col         : Optional column for clustered SEs
    offset_col          : Optional pre-computed log(offset) column
    model_type          : 'negbin' | 'ols'
    """
    if model_type == "ols":
        if cluster_col:
            return ClusteredOLSRegression(target_col, experimental_groups, cluster_col, offset_col)
        return StandardOLSRegression(target_col, experimental_groups, offset_col)

    if cluster_col:
        return ClusteredErrorRegression(target_col, experimental_groups, cluster_col, offset_col)
    return StandardErrorRegression(target_col, experimental_groups, offset_col)
