import pandas as pd
import numpy as np
import statsmodels.api as sm
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
                  e.g. ("Gemma", "4", "Formative")
                  For the baseline model this is an empty tuple.
    label       : human-readable string used for display / plot titles
    result      : fitted statsmodels result object
    y           : dependent-variable series used for this fit
    X           : design matrix used for this fit
    long_df     : the pre-model data frame (wide pivoted form)
    iv_cols     : list of predictor column names (excluding 'const')
    clusters    : cluster series (None when not clustering)
    """
    group_key: tuple
    label: str
    result: Any
    y: pd.Series
    X: pd.DataFrame
    long_df: pd.DataFrame
    iv_cols: List[str]
    clusters: Optional[pd.Series] = None

    # ------------------------------------------------------------------ #
    # Convenience accessors that mirror the old run_data dict layout so   #
    # any code that already does runner.run_data[label] still works after  #
    # the property shim below converts records → the old dict format.      #
    # ------------------------------------------------------------------ #
    def as_run_data_dict(self) -> Dict[str, Any]:
        return {
            "y": self.y,
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

        # ── Primary store ───────────────────────────────────────────────
        # All fitted models live here as structured RunRecord objects.
        self._records: List[RunRecord] = []

    # ------------------------------------------------------------------ #
    # Backward-compatible properties                                       #
    # ------------------------------------------------------------------ #

    @property
    def results(self) -> Dict[str, Any]:
        """
        Legacy dict: label → statsmodels result object.
        Downstream code that iterates runner.results.items() keeps working.
        """
        return {r.label: r.result for r in self._records}

    @property
    def run_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Legacy dict: label → run-data dict.
        Downstream code that does runner.run_data[label] keeps working.
        """
        return {r.label: r.as_run_data_dict() for r in self._records}

    @property
    def records(self) -> List[RunRecord]:
        """Direct access to the structured record list."""
        return self._records

    # ------------------------------------------------------------------ #
    # Tidy-frame export                                                    #
    # ------------------------------------------------------------------ #

    def to_frame(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame with one row per fitted model.

        Columns
        -------
        - one column per experimental group dimension  (easy groupby / faceting)
        - 'label'    : display string
        - 'nobs'     : number of observations
        - 'r_squared', 'adj_r_squared' : OLS fit stats (NaN for NegBin)
        - 'pseudo_r_squared'           : NegBin pseudo-R² (NaN for OLS)
        - 'aic', 'bic', 'llf'
        - 'condition_number'
        - 'result'   : the raw statsmodels result object
        - 'record'   : the RunRecord itself (for ad-hoc access)
        """
        rows = []
        for rec in self._records:
            r = rec.result
            row: Dict[str, Any] = {}

            # ── Experimental-group columns ───────────────────────────
            # group_key is () for the baseline model
            for dim, val in zip(self.experimental_groups, rec.group_key):
                row[dim] = val

            row["label"] = rec.label
            row["nobs"] = getattr(r, "nobs",            np.nan)
            row["r_squared"] = getattr(r, "rsquared",        np.nan)
            row["adj_r_squared"] = getattr(r, "rsquared_adj",    np.nan)
            row["pseudo_r_squared"] = getattr(r, "prsquared",       np.nan)
            row["aic"] = getattr(r, "aic",              np.nan)
            row["bic"] = getattr(r, "bic",              np.nan)
            row["llf"] = getattr(r, "llf",              np.nan)
            row["condition_number"] = getattr(r, "condition_number", np.nan)
            row["result"] = r
            row["record"] = rec
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Internal helpers: pivoting and design matrix                        #
    # ------------------------------------------------------------------ #

    def _pivot_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        score_col: Optional[str],
        extra_index_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Pivot LLM scores from long to wide format using deal_id as the
        minimal index, then merge all other columns back.
        """
        if score_col is None:
            return group_df.copy(), []

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

        if len(unique_dims) == 1 and unique_dims[0] == "quality":
            return group_df.copy(), [score_col]

        pivot_df = (
            group_df
            .pivot_table(
                index="deal_id",
                columns="dimension_name",
                values=score_col,
                aggfunc="first",
            )
            .reset_index()
        )
        score_cols = [c for c in pivot_df.columns if c != "deal_id"]

        side_df = (
            group_df[["deal_id"] + side_cols]
            .drop_duplicates(subset="deal_id")
        )

        wide_df = (
            pivot_df
            .merge(side_df, on="deal_id", how="inner")
            .dropna(subset=score_cols)
        )
        return wide_df, score_cols

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
                    X_clean.values, full_matrices=False, lapack_driver="gesvd"
                )
                tol = X_clean.values.max() * max(X_clean.shape) * np.finfo(float).eps
                rank = np.count_nonzero(s > tol)
                if rank < X_clean.shape[1]:
                    print(
                        "⚠️  Warning: Design matrix is rank-deficient (perfect multicollinearity)")
                    return None
            except Exception as e:
                print(f"🚨 Both SVD solvers failed. {e}")
                return None

        return X_clean

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def prepare_group_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        new_predictor: Optional[str],
        dims_to_exclude: Optional[List[str]] = None,
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
    # Internal: build a RunRecord and append it                           #
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
        rec = RunRecord(
            group_key=group_key,
            label=label,
            result=result,
            y=y,
            X=X,
            long_df=wide_df,
            iv_cols=[c for c in X.columns if c != "const"],
            clusters=clusters,
        )
        self._records.append(rec)

    # ------------------------------------------------------------------ #
    # Public fitting methods                                              #
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
            new_predictor=None,
            dims_to_exclude=None,
        )
        if data_dict is None:
            print("Baseline model: Data preparation failed.")
            return self

        y, X = data_dict.get("y"), data_dict.get("X")
        if y is None or X is None:
            print("Baseline model: Missing y or X data.")
            return self
        if len(y) < (X.shape[1] + 2):
            print("Baseline model: Insufficient observations.")
            return self

        offset = data_dict.get("offset")
        clusters = data_dict.get("clusters")
        wide_df = data_dict.get("wide_df", X.join(y))

        try:
            res = self._fit_model(y=y, X=X, offset=offset, clusters=clusters)

            if hasattr(res, "converged") and not res.converged:
                print("⚠️  Warning: Baseline model did not converge.")

            self._store_record(
                group_key=tuple(
                    BASELINE_SENTINEL for _ in self.experimental_groups),
                label=label,
                result=res,
                y=y,
                X=X,
                wide_df=wide_df,
                clusters=clusters,
            )

            metric = (
                f"AIC={res.aic:.2f}" if hasattr(res, "aic")
                else f"R²={res.rsquared:.3f}"
            )
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
        predictor_name: str,
        label_map: Optional[Dict[str, str]] = None,
        dims_to_exclude: Optional[List[str]] = None,
    ) -> "BaseRegressionRunner":
        """Main execution loop for LLM-based models."""

        grouped = df.groupby(self.experimental_groups)

        for group_keys, group_df in grouped:
            # Normalise to a plain tuple of strings
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_key_tuple = tuple(str(k) for k in group_keys)

            data_dict = self.prepare_group_data(
                group_df, cat_vars, bin_vars, num_vars,
                predictor_name, dims_to_exclude,
            )
            if data_dict is None:
                continue

            y, X = data_dict.get("y"), data_dict.get("X")
            if y is None or X is None:
                print(f"Skipping {group_keys}: Missing y or X data.")
                continue
            if len(y) < (X.shape[1] + 2):
                print(f"Skipping {group_keys}: Insufficient observations.")
                continue

            offset = data_dict.get("offset")
            clusters = data_dict.get("clusters")
            wide_df = data_dict.get("wide_df", X.join(y))

            # Build human-readable label
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
        """
        Return a tidy DataFrame of fit statistics for all stored results.

        Prefer to_frame() for richer output; this method is kept for
        backward compatibility and sorts by adj R² or AIC.
        """
        df = self.to_frame().drop(
            columns=["result", "record"], errors="ignore")

        sort_col = "adj_r_squared" if df["adj_r_squared"].notna(
        ).any() else "aic"
        ascending = sort_col == "aic"
        return df.sort_values(sort_col, ascending=ascending)

    # Alias
    run_negative_binomial = run_regression


# =============================================================================
# NEGATIVE BINOMIAL IMPLEMENTATIONS (COUNT MODELS)
# =============================================================================

class StandardErrorRegression(BaseRegressionRunner):
    """Standard Negative Binomial Regression."""

    def prepare_group_data(
        self, group_df, cat_predictors, bin_predictors, num_predictors,
        new_predictor, dims_to_exclude=None,
    ):
        wide_df, score_cols = self._pivot_data(
            group_df, cat_predictors, bin_predictors, num_predictors, new_predictor
        )
        if dims_to_exclude:
            score_cols = [c for c in score_cols if c not in dims_to_exclude]

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
        new_predictor, dims_to_exclude=None,
    ):
        wide_df, score_cols = self._pivot_data(
            group_df, cat_predictors, bin_predictors, num_predictors,
            new_predictor, extra_index_cols=[self.cluster_col],
        )
        if dims_to_exclude:
            score_cols = [c for c in score_cols if c not in dims_to_exclude]

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


# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from statsmodels.genmod.generalized_linear_model import GLMResults, GLMResultsWrapper
# from statsmodels.discrete.discrete_model import NegativeBinomialResultsWrapper, CountResultsWrapper
# from statsmodels.regression.linear_model import RegressionResultsWrapper
# from typing import List, Dict, Optional, Tuple, Any, Union
# from abc import ABC, abstractmethod


# class BaseRegressionRunner(ABC):
#     """Base class for regression analysis with robust error handling and alpha estimation."""

#     def __init__(self, target_col: str, experimental_groups: List[str], offset_col: Optional[str] = None):
#         self.target_col = target_col
#         self.experimental_groups = experimental_groups
#         # Column containing pre-computed log(offset)
#         self.offset_col = offset_col
#         self.results = {}

#     # def _pivot_data(
#     #     self,
#     #     group_df: pd.DataFrame,
#     #     cat_predictors: List[str],
#     #     bin_predictors: List[str],
#     #     num_predictors: List[str],
#     #     # The LLM metric to pivot (None for baseline)
#     #     score_col: Optional[str],
#     #     extra_index_cols: Optional[List[str]] = None
#     # ) -> Tuple[pd.DataFrame, List[str]]:
#     #     """Pivot data from long to wide format while maintaining observation integrity."""

#     #     # For baseline model, skip pivoting since there's no LLM score
#     #     if score_col is None:
#     #         return group_df.copy(), []

#     #     unique_dims = group_df['dimension_name'].unique()

#     #     index_cols = (
#     #         self.experimental_groups +
#     #         cat_predictors +
#     #         bin_predictors +
#     #         num_predictors +
#     #         [self.target_col, 'deal_text', 'deal_id']
#     #     )

#     #     # Include offset column in index if it exists
#     #     if self.offset_col:
#     #         index_cols.append(self.offset_col)

#     #     if extra_index_cols:
#     #         index_cols.extend(extra_index_cols)

#     #     # Ensure we don't have duplicates before pivoting
#     #     if len(unique_dims) == 1 and unique_dims[0] == 'quality':
#     #         wide_df = group_df.copy()
#     #         score_cols = [score_col]
#     #     else:
#     #         # Use aggfunc='first' to ensure we don't accidentally mean-average descriptors
#     #         score_cols = list(unique_dims)
#     #         wide_df = group_df.pivot_table(
#     #             index=index_cols,
#     #             columns='dimension_name',
#     #             values=score_col,
#     #             aggfunc='first'
#     #         ).reset_index().dropna(subset=score_cols)

#     #     return wide_df, score_cols

#     def _pivot_data(
#         self,
#         group_df: pd.DataFrame,
#         cat_predictors: List[str],
#         bin_predictors: List[str],
#         num_predictors: List[str],
#         score_col: Optional[str],
#         extra_index_cols: Optional[List[str]] = None
#     ) -> Tuple[pd.DataFrame, List[str]]:
#         """Pivot LLM scores from long to wide format using deal_id as the minimal index,
#         then merge all other columns back. Avoids combinatorial MultiIndex explosion."""

#         if score_col is None:
#             return group_df.copy(), []

#         unique_dims = group_df['dimension_name'].unique()

#         # columns we need to carry alongside the pivot result
#         side_cols = (
#             self.experimental_groups
#             + cat_predictors
#             + bin_predictors
#             + num_predictors
#             + [self.target_col, 'deal_text']
#         )
#         if self.offset_col:
#             side_cols.append(self.offset_col)
#         if extra_index_cols:
#             side_cols.extend(extra_index_cols)
#         # deduplicate while preserving order
#         side_cols = list(dict.fromkeys(
#             c for c in side_cols if c in group_df.columns))

#         if len(unique_dims) == 1 and unique_dims[0] == 'quality':
#             wide_df = group_df.copy()
#             score_cols = [score_col]
#             return wide_df, score_cols

#         # 1. pivot only score_col over dimension_name, keyed by deal_id alone
#         pivot_df = (
#             group_df
#             .pivot_table(
#                 index='deal_id',
#                 columns='dimension_name',
#                 values=score_col,
#                 aggfunc='first',
#             )
#             .reset_index()
#         )
#         score_cols = [c for c in pivot_df.columns if c != 'deal_id']

#         # 2. pull one row per deal_id for all side columns (they are constant within a deal)
#         side_df = (
#             group_df[['deal_id'] + side_cols]
#             .drop_duplicates(subset='deal_id')
#         )

#         # 3. merge back — now wide_df has the same rows as before but no index explosion
#         wide_df = pivot_df.merge(
#             side_df, on='deal_id', how='inner').dropna(subset=score_cols)

#         return wide_df, score_cols

#     def _build_predictor_matrix(
#         self,
#         wide_df: pd.DataFrame,
#         score_cols: List[str],
#         cat_predictors: List[str],
#         bin_predictors: List[str],
#         num_predictors: List[str]
#     ) -> Optional[pd.DataFrame]:
#         """Build the X matrix and check for mathematical validity (rank)."""
#         predictor_parts = []

#         # Numeric columns: LLM scores + binary predictors + continuous numerical predictors
#         numeric_cols = score_cols + bin_predictors + num_predictors
#         if numeric_cols:
#             X_num = wide_df[numeric_cols].copy()
#             # Convert binary predictors to int
#             for col in bin_predictors:
#                 X_num[col] = X_num[col].astype(int)
#             predictor_parts.append(X_num)

#         # Categorical dummy columns
#         if cat_predictors:
#             cat_subset = wide_df[cat_predictors].copy()
#             cat_subset = cat_subset.astype(str)

#             X_cat = pd.get_dummies(cat_subset, drop_first=True, dtype=int)
#             predictor_parts.append(X_cat)

#         if not predictor_parts:
#             return None

#         X = pd.concat(predictor_parts, axis=1)
#         X = sm.add_constant(X)

#         # # Rank check: Ensure no perfect multicollinearity
#         # if np.linalg.matrix_rank(X.values) < X.shape[1]:
#         #     print(
#         #         "⚠️ Warning: Design matrix is rank-deficient (perfect multicollinearity)")
#         #     return None

#         # return X

#         try:
#             X_clean = X.astype(float)
#         except ValueError as e:
#             print(f"⚠️ Warning: Could not cast matrix to float. {e}")
#             return None

#         # --- BULLETPROOF RANK CHECK ---
#         import scipy.linalg

#         try:
#             # Attempt NumPy's fast rank check first
#             if np.linalg.matrix_rank(X_clean.values) < X_clean.shape[1]:
#                 print(
#                     "⚠️ Warning: Design matrix is rank-deficient (perfect multicollinearity)")
#                 return None

#         except np.linalg.LinAlgError:
#             # If NumPy's 'gesdd' solver crashes, catch it and use SciPy's robust 'gesvd' solver
#             # print("🔄 NumPy SVD failed to converge. Using robust SciPy solver...")
#             try:
#                 _, s, _ = scipy.linalg.svd(
#                     X_clean.values, full_matrices=False, lapack_driver='gesvd')
#                 # Calculate rank mathematically based on singular values
#                 tol = X_clean.values.max() * max(X_clean.shape) * np.finfo(float).eps
#                 rank = np.count_nonzero(s > tol)

#                 if rank < X_clean.shape[1]:
#                     print(
#                         "⚠️ Warning: Design matrix is rank-deficient (perfect multicollinearity)")
#                     return None
#             except Exception as e:
#                 print(
#                     f"🚨 Both SVD solvers failed! Matrix geometry is unresolvable. {e}")
#                 return None

#         return X_clean

#     @abstractmethod
#     def prepare_group_data(
#         self,
#         group_df: pd.DataFrame,
#         cat_predictors: List[str],
#         bin_predictors: List[str],
#         num_predictors: List[str],
#         new_predictor: Optional[str],  # None for baseline
#         dims_to_exclude: Optional[List[str]] = None
#     ) -> Optional[Dict[str, Any]]:
#         """Prepare data for regression. Implementation varies by subclass."""
#         pass

#     @abstractmethod
#     def _fit_model(
#         self,
#         y: pd.Series,
#         X: pd.DataFrame,
#         offset: Optional[pd.Series] = None,
#         **kwargs
#     ) -> Any:
#         """Fit the model (OLS or GLM). Implementation varies by subclass."""
#         pass

#     @property
#     def dep_var(self) -> str:
#         """Alias so diagnostics.py can treat both runners identically."""
#         return self.target_col

#     def run_baseline_model(
#         self,
#         df: pd.DataFrame,
#         cat_vars: List[str],
#         bin_vars: List[str],
#         num_vars: List[str],
#         label: str = "baseline"
#     ) -> Dict[str, Any]:
#         """Fit a single baseline model without LLM predictors."""

#         data_dict = self.prepare_group_data(
#             df, cat_vars, bin_vars, num_vars,
#             new_predictor=None,  # Signal this is baseline
#             dims_to_exclude=None
#         )

#         if data_dict is None:
#             print("Baseline model: Data preparation failed.")
#             return self.results

#         y = data_dict.get('y')
#         X = data_dict.get('X')

#         if y is None or X is None:
#             print("Baseline model: Missing y or X data.")
#             return self.results

#         if len(y) < (X.shape[1] + 2):
#             print("Baseline model: Insufficient observations.")
#             return self.results

#         offset = data_dict.get('offset')
#         clusters = data_dict.get('clusters')

#         try:
#             res = self._fit_model(
#                 y=y,
#                 X=X,
#                 offset=offset,
#                 clusters=clusters
#             )

#             # Helper to check convergence attribute safely (OLS always "converges" if rank is ok)
#             if hasattr(res, 'converged') and not res.converged:
#                 print(f"⚠️ Warning: Baseline model did not converge.")

#             self.results[label] = res

#             if not hasattr(self, "run_data"):
#                 self.run_data = {}
#             self.run_data[label] = {
#                 "y": y,
#                 "X": X,
#                 # wide_df is the pre-melt frame
#                 "long_df": data_dict.get("wide_df", X.join(y)),
#                 "iv_cols": [c for c in X.columns if c != "const"],
#                 "clusters": clusters,
#             }

#             # Metric extraction for print
#             metric = f"AIC={res.aic:.2f}" if hasattr(
#                 res, 'aic') else f"R2={res.rsquared:.3f}"
#             print(
#                 f"✓ Baseline model fitted: {res.nobs} observations, {metric}")

#         except Exception as e:
#             print(f"Error fitting baseline model: {e}")

#         return self.results

#     def run_regression(
#         self,
#         df: pd.DataFrame,
#         cat_vars: List[str],
#         bin_vars: List[str],
#         num_vars: List[str],
#         predictor_name: str,
#         label_map: Optional[Dict[str, str]] = None,
#         dims_to_exclude: Optional[List[str]] = None
#     ) -> Dict[str, Any]:
#         """Main execution loop for LLM-based models."""
#         grouped = df.groupby(self.experimental_groups)

#         for group_keys, group_df in grouped:
#             data_dict = self.prepare_group_data(
#                 group_df, cat_vars, bin_vars, num_vars, predictor_name, dims_to_exclude
#             )

#             if data_dict is None:
#                 continue

#             y = data_dict.get('y')
#             X = data_dict.get('X')

#             if y is None or X is None:
#                 print(f"Skipping {group_keys}: Missing y or X data.")
#                 continue

#             if X is None or len(y) < (X.shape[1] + 2):
#                 print(f"Skipping {group_keys}: Insufficient observations.")
#                 continue

#             offset = data_dict.get('offset')
#             clusters = data_dict.get('clusters')

#             try:
#                 res = self._fit_model(
#                     y=y,
#                     X=X,
#                     offset=offset,
#                     clusters=clusters
#                 )

#                 if hasattr(res, 'converged') and not res.converged:
#                     print(
#                         f"⚠️ Warning: Model for {group_keys} did not converge.")

#                 pk = list(group_keys) if isinstance(
#                     group_keys, tuple) else [group_keys]
#                 pk = [str(k) for k in pk]  # Convert all to strings

#                 if label_map and len(pk) >= 2:
#                     # Use the map for the second element (usually prompt_id)
#                     model_label = f"{pk[0]} {label_map.get(pk[1], pk[1])}"
#                 else:
#                     # Simply join with spaces: "Gemma 4 Formative"
#                     model_label = " ".join(pk)

#                 # if label_map:
#                 #     # Handle tuple keys cleanly if needed
#                 #     pk = group_keys if isinstance(
#                 #         group_keys, tuple) else (group_keys,)
#                 #     model_label = f"{pk[0]}_{label_map.get(pk[1], pk[1])}"
#                 # else:
#                 #     model_label = str(group_keys)

#                 self.results[model_label] = res

#                 if not hasattr(self, "run_data"):
#                     self.run_data = {}
#                 self.run_data[model_label] = {
#                     "y": y,
#                     "X": X,
#                     # wide_df is the pre-melt frame
#                     "long_df": data_dict.get("wide_df", X.join(y)),
#                     "iv_cols": [c for c in X.columns if c != "const"],
#                     "clusters": clusters,
#                 }

#             except Exception as e:
#                 print(f"Error fitting model for {group_keys}: {e}")

#         return self.results

#     def summarize(self) -> pd.DataFrame:
#         """Return a DataFrame of fit statistics for all stored results, sorted by adj R²."""
#         rows = []
#         for label, r in self.results.items():
#             # NegBin / GLM results use llf (log-likelihood) not rsquared
#             row = {
#                 "label": label,
#                 "nobs": getattr(r, "nobs", np.nan),
#                 # OLS only
#                 "r_squared": getattr(r, "rsquared", np.nan),
#                 # OLS only
#                 "adj_r_squared": getattr(r, "rsquared_adj", np.nan),
#                 "pseudo_r_squared": getattr(r, "prsquared", np.nan),  # NegBin
#                 "aic": getattr(r, "aic", np.nan),
#                 "bic": getattr(r, "bic", np.nan),
#                 "llf": getattr(r, "llf", np.nan),
#                 "condition_number": getattr(r, "condition_number", np.nan),
#             }
#             rows.append(row)

#         df = pd.DataFrame(rows)

#         # sort by whichever fit metric is available
#         sort_col = "adj_r_squared" if df["adj_r_squared"].notna(
#         ).any() else "aic"
#         ascending = sort_col == "aic"
#         return df.sort_values(sort_col, ascending=ascending)

#     # Alias for backward compatibility if you have other scripts calling this name
#     run_negative_binomial = run_regression


# # =============================================================================
# # NEGATIVE BINOMIAL IMPLEMENTATIONS (COUNT MODELS)
# # =============================================================================

# class StandardErrorRegression(BaseRegressionRunner):
#     """Standard Negative Binomial Regression."""

#     def prepare_group_data(
#         self, group_df, cat_predictors, bin_predictors, num_predictors, new_predictor, dims_to_exclude=None
#     ):
#         wide_df, score_cols = self._pivot_data(
#             group_df, cat_predictors, bin_predictors, num_predictors, new_predictor
#         )
#         if dims_to_exclude:
#             score_cols = [
#                 col for col in score_cols if col not in dims_to_exclude]

#         X = self._build_predictor_matrix(
#             wide_df, score_cols, cat_predictors, bin_predictors, num_predictors)
#         if X is None:
#             return None

#         y = wide_df[self.target_col]
#         data_components = [y, X]
#         if self.offset_col and self.offset_col in wide_df.columns:
#             data_components.append(wide_df[self.offset_col])

#         combined = pd.concat(data_components, axis=1).dropna()
#         if combined.empty:
#             return None

#         return {
#             'y': combined[self.target_col],
#             'X': combined[X.columns],
#             'offset': combined[self.offset_col] if self.offset_col else None,
#             'clusters': None
#         }

#     def _fit_model(self, y, X, offset=None, **kwargs):
#         model = sm.NegativeBinomial(y, X, loglike_method='nb2', offset=offset)
#         return model.fit(maxiter=2000, method='bfgs', disp=0)


# class ClusteredErrorRegression(BaseRegressionRunner):
#     """Clustered Negative Binomial Regression."""

#     def __init__(self, target_col, experimental_groups, cluster_col, offset_col=None):
#         super().__init__(target_col, experimental_groups, offset_col)
#         self.cluster_col = cluster_col

#     def prepare_group_data(
#         self, group_df, cat_predictors, bin_predictors, num_predictors, new_predictor, dims_to_exclude=None
#     ):
#         wide_df, score_cols = self._pivot_data(
#             group_df, cat_predictors, bin_predictors, num_predictors, new_predictor, extra_index_cols=[
#                 self.cluster_col]
#         )
#         if dims_to_exclude:
#             score_cols = [
#                 col for col in score_cols if col not in dims_to_exclude]

#         X = self._build_predictor_matrix(
#             wide_df, score_cols, cat_predictors, bin_predictors, num_predictors)
#         if X is None:
#             return None

#         y = wide_df[self.target_col]
#         clusters = wide_df[self.cluster_col]
#         data_components = [y, X, clusters]
#         if self.offset_col and self.offset_col in wide_df.columns:
#             data_components.append(wide_df[self.offset_col])

#         combined = pd.concat(data_components, axis=1).dropna()
#         if combined.empty:
#             return None

#         return {
#             'y': combined[self.target_col],
#             'X': combined[X.columns],
#             'clusters': combined[self.cluster_col],
#             'offset': combined[self.offset_col] if self.offset_col else None
#         }

#     def _fit_model(self, y, X, offset=None, **kwargs):
#         clusters = kwargs.get('clusters')
#         if clusters is None:
#             raise ValueError(
#                 "ClusteredErrorRegression requires 'clusters' argument.")
#         model = sm.NegativeBinomial(y, X, loglike_method='nb2', offset=offset)
#         return model.fit(cov_type='cluster', cov_kwds={'groups': clusters}, maxiter=2000, disp=0, use_t=True)


# # =============================================================================
# # OLS IMPLEMENTATIONS (LOG-LINEAR MODELS)
# # =============================================================================

# class StandardOLSRegression(StandardErrorRegression):
#     """Standard OLS Regression (Inherits data prep from StandardErrorRegression)."""

#     def _fit_model(self, y, X, offset=None, **kwargs):
#         # Note: OLS does not use the 'offset' parameter in the same way as GLM.
#         # We assume the user has included the log-offset as a predictor in X (via num_vars).
#         return sm.OLS(y, X).fit()


# class ClusteredOLSRegression(ClusteredErrorRegression):
#     """Clustered OLS Regression (Inherits data prep from ClusteredErrorRegression)."""

#     def _fit_model(self, y, X, offset=None, **kwargs):
#         clusters = kwargs.get('clusters')
#         if clusters is None:
#             raise ValueError(
#                 "ClusteredOLSRegression requires 'clusters' argument.")

#         # Fit OLS
#         results = sm.OLS(y, X).fit()

#         # Calculate Robust Clustered SEs post-hoc
#         # usage: get_robustcov_results(cov_type='cluster', groups=clusters)
#         return results.get_robustcov_results(cov_type='cluster', groups=clusters)


# # =============================================================================
# # FACTORY FUNCTION
# # =============================================================================

# def create_regression_runner(
#     target_col: str,
#     experimental_groups: List[str],
#     cluster_col: Optional[str] = None,
#     offset_col: Optional[str] = None,
#     model_type: str = 'negbin'  # Options: 'negbin', 'ols'
# ) -> BaseRegressionRunner:
#     """
#     Factory function to create the appropriate regression runner.

#     Args:
#         target_col: Name of dependent variable (Count for NegBin, Log-Transformed for OLS)
#         model_type: 'negbin' for Negative Binomial, 'ols' for Ordinary Least Squares
#     """

#     if model_type == 'ols':
#         if cluster_col:
#             return ClusteredOLSRegression(target_col, experimental_groups, cluster_col, offset_col)
#         return StandardOLSRegression(target_col, experimental_groups, offset_col)

#     # Default to Negative Binomial
#     if cluster_col:
#         return ClusteredErrorRegression(target_col, experimental_groups, cluster_col, offset_col)
#     return StandardErrorRegression(target_col, experimental_groups, offset_col)
