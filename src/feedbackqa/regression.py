from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


RegressionMode = Literal["rating", "disagreement", "aggregated", "verbosity"]


class FeedbackQARegressionRunner:
    """
    Fits clustered OLS models for FeedbackQA data grouped by experiment conditions.

    Categorical covariates
    ----------------------
    Pass column names via ``categorical_covariates``.  They are one-hot encoded
    (drop-first) automatically before any mode's data-preparation step, so the
    resulting dummy columns are added to the design matrix exactly like numeric
    covariates.  Example::

        runner = FeedbackQARegressionRunner(
            categorical_covariates=["length_bucket", "domain"],
        )

    The reference level (dropped category) is whichever label sorts first after
    ``pd.get_dummies(drop_first=True)`` — typically the alphabetically first
    label, e.g. ``"1. Short"`` for a ``length_bucket`` column.

    Modes
    -----
    "rating" (default)
        Predicts individual human ratings from LLM scores (+ optional entropy/covariates).
        score_cols are melted so each rater-input pair is one row.
        Clustered by input_id.

        Formula: human_rating ~ mean_rating_dim1 + mean_rating_dim2 + ... [+ entropy_* + covariates]

    "disagreement"
        Predicts absolute inter-rater disagreement from LLM scores (+ optional entropy/covariates).
        One row per input — no clustering needed.

        Formula: score_1_score_2_disagreement ~ mean_rating_dim1 + ...

    "aggregated"
        Predicts LLM scores from averaged human rating (+ covariates).
        score_cols are averaged into avg_human_rating. One row per input, no clustering.
        dep_var must be a pre-existing input-level column (e.g. mean_rating).

        Formula: mean_rating ~ avg_human_rating + covariates

    "verbosity"
        Predicts LLM scores from individual human ratings (+ covariates).
        dep_var must be a pre-existing input-level column (e.g. mean_rating).
        score_cols are melted into human_rating predictor. Clustered by input_id.
        No pivot — all variables are input-level, dimension is ignored.

        Formula: mean_rating ~ human_rating + covariates
    """

    def __init__(
        self,
        experimental_groups: Optional[List[str]] = None,
        prompt_hash_map: Optional[Mapping[str, str]] = None,
        dep_var: str = "raw_human_rating",
        llm_prediction_col: str = "mean_rating",
        covariates: Optional[List[str]] = None,
        categorical_covariates: Optional[List[str]] = None,
        entropy_col: str = "normalized_entropy",
        dimension_col: str = "dimension_name",
        model_col: str = "model_name",
        prompt_col: str = "prompt_id",
        input_col: str = "input_id",
        cluster_col: str = "input_id",
        score_cols: Sequence[str] = ("score_1", "score_2"),
        include_entropy: bool = True,
        mode: RegressionMode = "rating",
    ) -> None:
        self.experimental_groups = experimental_groups or [
            "model_name", "prompt_id"]
        self.prompt_hash_map = dict(prompt_hash_map or {})

        self.dep_var = dep_var
        self.llm_prediction_col = llm_prediction_col
        self.covariates = covariates or []
        self.categorical_covariates = categorical_covariates or []
        self.entropy_col = entropy_col
        self.dimension_col = dimension_col
        self.model_col = model_col
        self.prompt_col = prompt_col
        self.input_col = input_col
        self.cluster_col = cluster_col
        self.score_cols = list(score_cols)
        self.include_entropy = include_entropy
        self.mode = mode

        # populated by run_regression / run_custom_regression
        self.results: Dict[str, Any] = {}
        self.run_data: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_columns(columns) -> List[str]:
        """Collapse MultiIndex pivot columns to 'value_dim' strings."""
        flat = []
        for col in columns:
            if isinstance(col, tuple):
                parts = [str(c) for c in col if c not in (None, "")]
                flat.append("_".join(parts).strip("_"))
            else:
                flat.append(str(col))
        return flat

    def _infer_holistic(self, group_df: pd.DataFrame) -> bool:
        """Return True when the group contains only one dimension (holistic scoring)."""
        return len(group_df[self.dimension_col].dropna().unique()) == 1

    def _make_model_label(self, group_keys: Any) -> str:
        """
        Build a human-readable label for a group.
        For the default grouping [model_name, prompt_id]: "<model_name>_<prompt_description>"
        """
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        if (
            self.experimental_groups == [self.model_col, self.prompt_col]
            and len(group_keys) == 2
        ):
            model_name, prompt_id = group_keys
            prompt_desc = self.prompt_hash_map.get(prompt_id, str(prompt_id))
            return f"{model_name}_{prompt_desc}"

        parts = []
        for group_name, key in zip(self.experimental_groups, group_keys):
            if group_name == self.prompt_col:
                key = self.prompt_hash_map.get(key, str(key))
            parts.append(str(key))
        return "_".join(parts)

    def _fit_ols(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        clusters: Optional[pd.Series] = None,
    ):
        """OLS with optional cluster-robust standard errors."""
        X_const = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X_const)
        if clusters is not None:
            return model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
        return model.fit()

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------

    def _encode_categoricals(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        One-hot encode self.categorical_covariates (drop-first to avoid
        perfect multicollinearity) and return:
          - df with dummy columns appended
          - list of the new dummy column names

        Columns in categorical_covariates that are absent from df are
        silently skipped so the method is safe to call on any slice.
        """
        present = [c for c in self.categorical_covariates if c in df.columns]
        if not present:
            return df, []

        dummies = pd.get_dummies(
            df[present],
            columns=present,
            drop_first=True,
            dtype=float,
        )
        dummy_cols = dummies.columns.tolist()
        df = pd.concat([df.reset_index(drop=True),
                        dummies.reset_index(drop=True)], axis=1)
        return df, dummy_cols

    # ------------------------------------------------------------------
    # Data preparation — one method per mode
    # ------------------------------------------------------------------

    def _prepare_rating_data(
        self,
        df: pd.DataFrame,
        predictor_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Predicts human ratings from LLM scores.

        Holistic: one predictor column, melted over raters.
        Dimensional: pivot LLM scores (and optionally entropy) per dimension,
                     then melt over raters. Covariates added to pivot index
                     so they survive unprefixed.
        """
        is_holistic = self._infer_holistic(df)
        unique_dims = sorted(df[self.dimension_col].dropna().unique().tolist())

        if is_holistic:
            keep_cols = [self.input_col, self.model_col,
                         *self.score_cols, predictor_name]
            if self.include_entropy and self.entropy_col in df.columns:
                keep_cols.append(self.entropy_col)
            keep_cols += [c for c in self.covariates if c in df.columns]
            keep_cols = [c for c in dict.fromkeys(
                keep_cols) if c in df.columns]

            wide_df = df[keep_cols].drop_duplicates(
                subset=[self.input_col]).copy()

            iv_cols = [predictor_name]
            if self.include_entropy and self.entropy_col in wide_df.columns:
                iv_cols.append(self.entropy_col)
            iv_cols += [c for c in self.covariates if c in wide_df.columns]

        else:
            # covariates go into the index so they are not dimension-suffixed after pivot
            index_cols = [self.input_col, self.model_col, *self.score_cols]
            index_cols += [c for c in self.covariates if c in df.columns]
            index_cols = [c for c in dict.fromkeys(
                index_cols) if c in df.columns]

            value_cols = [predictor_name]
            if self.include_entropy and self.entropy_col in df.columns:
                value_cols.append(self.entropy_col)

            wide_df = (
                df.pivot_table(
                    index=index_cols,
                    columns=self.dimension_col,
                    values=value_cols,
                    aggfunc="mean",
                )
                .reset_index()
            )
            wide_df.columns = self._flatten_columns(wide_df.columns)

            iv_cols = [c for c in wide_df.columns if c.startswith(
                f"{predictor_name}_")]
            if self.include_entropy:
                iv_cols += [
                    c for c in wide_df.columns if c.startswith(f"{self.entropy_col}_")
                ]
            iv_cols += [c for c in self.covariates if c in wide_df.columns]

        if not iv_cols:
            return None

        score_cols_present = [
            c for c in self.score_cols if c in wide_df.columns]
        if not score_cols_present:
            return None

        id_vars = list(dict.fromkeys(
            c for c in [self.input_col, self.model_col, *iv_cols]
            if c in wide_df.columns
        ))

        long_df = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=score_cols_present,
            var_name="rater_id",
            value_name=self.dep_var,
        ).dropna(subset=[self.dep_var])

        if long_df.empty:
            return None

        clusters = long_df[self.cluster_col] if self.cluster_col in long_df.columns else None

        return {
            "y": long_df[self.dep_var],
            "X": long_df[iv_cols].copy(),
            "clusters": clusters,
            "iv_cols": iv_cols,
            "formula": f"{self.dep_var} ~ " + " + ".join(iv_cols),
            "wide_df": wide_df,
            "long_df": long_df,
            "is_holistic": is_holistic,
            "dimensions": unique_dims,
        }

    def _prepare_disagreement_data(
        self,
        df: pd.DataFrame,
        predictor_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Predicts absolute inter-rater disagreement from LLM scores.

        Requires exactly two score_cols. Disagreement = |score_1 - score_2|.
        One row per input — no clustering needed.
        """
        if len(self.score_cols) != 2:
            raise ValueError(
                f"Disagreement mode requires exactly 2 score_cols, got {self.score_cols}"
            )
        s1, s2 = self.score_cols

        is_holistic = self._infer_holistic(df)
        unique_dims = sorted(df[self.dimension_col].dropna().unique().tolist())
        dep_var = f"{s1}_{s2}_disagreement"

        if is_holistic:
            keep_cols = [self.input_col,
                         self.model_col, s1, s2, predictor_name]
            if self.include_entropy and self.entropy_col in df.columns:
                keep_cols.append(self.entropy_col)
            keep_cols += [c for c in self.covariates if c in df.columns]
            keep_cols = [c for c in dict.fromkeys(
                keep_cols) if c in df.columns]

            wide_df = df[keep_cols].drop_duplicates(
                subset=[self.input_col]).copy()

            iv_cols = [predictor_name]
            if self.include_entropy and self.entropy_col in wide_df.columns:
                iv_cols.append(self.entropy_col)
            iv_cols += [c for c in self.covariates if c in wide_df.columns]

        else:
            index_cols = [self.input_col, self.model_col, s1, s2]
            index_cols += [c for c in self.covariates if c in df.columns]
            index_cols = [c for c in dict.fromkeys(
                index_cols) if c in df.columns]

            value_cols = [predictor_name]
            if self.include_entropy and self.entropy_col in df.columns:
                value_cols.append(self.entropy_col)

            wide_df = (
                df.pivot_table(
                    index=index_cols,
                    columns=self.dimension_col,
                    values=value_cols,
                    aggfunc="mean",
                )
                .reset_index()
            )
            wide_df.columns = self._flatten_columns(wide_df.columns)

            iv_cols = [c for c in wide_df.columns if c.startswith(
                f"{predictor_name}_")]
            if self.include_entropy:
                iv_cols += [
                    c for c in wide_df.columns if c.startswith(f"{self.entropy_col}_")
                ]
            iv_cols += [c for c in self.covariates if c in wide_df.columns]

        if not iv_cols:
            return None

        if s1 not in wide_df.columns or s2 not in wide_df.columns:
            return None

        wide_df = wide_df.copy()
        wide_df[dep_var] = (wide_df[s1] - wide_df[s2]).abs()
        long_df = wide_df.dropna(subset=[dep_var, *iv_cols]).copy()

        if long_df.empty:
            return None

        return {
            "y": long_df[dep_var],
            "X": long_df[iv_cols].copy(),
            "clusters": None,
            "iv_cols": iv_cols,
            "formula": f"{dep_var} ~ " + " + ".join(iv_cols),
            "wide_df": wide_df,
            "long_df": long_df,
            "is_holistic": is_holistic,
            "dimensions": unique_dims,
            "dep_var": dep_var,  # overrides runner.dep_var for this run
        }

    def _prepare_aggregated_data(
        self,
        df: pd.DataFrame,
        predictor_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Predicts LLM scores from averaged human rating (+ covariates).

        score_cols are averaged into a single avg_human_rating predictor.
        One row per input — no clustering needed.
        dep_var must be a pre-existing input-level column (e.g. mean_rating).
        """
        if self.dep_var not in df.columns:
            raise ValueError(
                f"dep_var '{self.dep_var}' not found. For aggregated mode, "
                f"dep_var must be a pre-existing input-level column (e.g. 'mean_rating')."
            )

        is_holistic = self._infer_holistic(df)
        unique_dims = sorted(df[self.dimension_col].dropna().unique().tolist())
        avg_human_col = "avg_human_rating"

        # all variables are input-level — deduplicate, no pivot needed
        keep_cols = [self.input_col, self.model_col,
                     *self.score_cols, self.dep_var]
        keep_cols += [c for c in self.covariates if c in df.columns]
        keep_cols = [c for c in dict.fromkeys(keep_cols) if c in df.columns]

        wide_df = df[keep_cols].drop_duplicates(subset=[self.input_col]).copy()
        score_cols_present = [
            c for c in self.score_cols if c in wide_df.columns]
        wide_df[avg_human_col] = wide_df[score_cols_present].mean(axis=1)

        iv_cols = [avg_human_col]
        iv_cols += [c for c in self.covariates if c in wide_df.columns]

        long_df = wide_df.dropna(subset=[self.dep_var, *iv_cols]).copy()
        if long_df.empty:
            return None

        return {
            "y": long_df[self.dep_var],
            "X": long_df[iv_cols].copy(),
            "clusters": None,
            "iv_cols": iv_cols,
            "formula": f"{self.dep_var} ~ " + " + ".join(iv_cols),
            "wide_df": wide_df,
            "long_df": long_df,
            "is_holistic": is_holistic,
            "dimensions": unique_dims,
        }

    def _prepare_verbosity_data(
        self,
        df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """
        Predicts LLM scores from individual human ratings (+ covariates).

        dep_var must be a pre-existing input-level column (e.g. mean_rating).
        score_cols are melted into a single human_rating predictor.
        One row per rater per input — clustered by input_id.
        No pivot — all variables are input-level, dimension_col is ignored.

        Formula: mean_rating ~ human_rating + covariates
        """
        if self.dep_var not in df.columns:
            raise ValueError(
                f"dep_var '{self.dep_var}' not found. For verbosity mode, "
                f"dep_var must be a pre-existing input-level column (e.g. 'mean_rating')."
            )

        is_holistic = self._infer_holistic(df)
        unique_dims = sorted(df[self.dimension_col].dropna().unique().tolist())

        keep_cols = [self.input_col, self.model_col,
                     *self.score_cols, self.dep_var]
        keep_cols += [c for c in self.covariates if c in df.columns]
        keep_cols = [c for c in dict.fromkeys(keep_cols) if c in df.columns]

        wide_df = df[keep_cols].drop_duplicates(subset=[self.input_col]).copy()

        score_cols_present = [
            c for c in self.score_cols if c in wide_df.columns]
        if not score_cols_present:
            return None

        id_vars = [self.input_col, self.model_col, self.dep_var]
        id_vars += [c for c in self.covariates if c in wide_df.columns]
        id_vars = [c for c in dict.fromkeys(id_vars) if c in wide_df.columns]

        long_df = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=score_cols_present,
            var_name="rater_id",
            value_name="human_rating",
        ).dropna(subset=[self.dep_var, "human_rating"])

        if long_df.empty:
            return None

        iv_cols = ["human_rating"]
        iv_cols += [c for c in self.covariates if c in long_df.columns]

        clusters = long_df[self.cluster_col] if self.cluster_col in long_df.columns else None

        return {
            "y": long_df[self.dep_var],
            "X": long_df[iv_cols].copy(),
            "clusters": clusters,
            "iv_cols": iv_cols,
            "formula": f"{self.dep_var} ~ " + " + ".join(iv_cols),
            "wide_df": wide_df,
            "long_df": long_df,
            "is_holistic": is_holistic,
            "dimensions": unique_dims,
        }

    # ------------------------------------------------------------------
    # Public preparation entry point
    # ------------------------------------------------------------------

    def prepare_group_data(
        self,
        group_df: pd.DataFrame,
        predictor_name: Optional[str] = None,
        dims_to_exclude: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Dispatch to the correct preparation method based on self.mode.

        Returns a dict with keys:
            y, X, clusters, iv_cols, formula,
            wide_df, long_df, is_holistic, dimensions
        Returns None when the slice is unusable.
        """
        predictor_name = predictor_name or self.llm_prediction_col
        df = group_df.copy()

        if dims_to_exclude:
            df = df[~df[self.dimension_col].isin(dims_to_exclude)].copy()
        if df.empty:
            return None

        # Encode categorical covariates and temporarily append dummy columns
        # to self.covariates so every _prepare_* method picks them up.
        df, dummy_cols = self._encode_categoricals(df)
        _original_covariates = self.covariates
        if dummy_cols:
            self.covariates = self.covariates + dummy_cols

        try:
            if self.mode == "disagreement":
                result = self._prepare_disagreement_data(df, predictor_name)
            elif self.mode == "aggregated":
                result = self._prepare_aggregated_data(df, predictor_name)
            elif self.mode == "verbosity":
                result = self._prepare_verbosity_data(df)
            else:
                result = self._prepare_rating_data(df, predictor_name)
        finally:
            # Always restore the original covariates list
            self.covariates = _original_covariates

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored results and run data."""
        self.results = {}
        self.run_data = {}

    def run_regression(
        self,
        df: pd.DataFrame,
        predictor_name: Optional[str] = None,
        dims_to_exclude: Optional[List[str]] = None,
        reset: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit all grouped regressions and store them in self.results / self.run_data.

        Parameters
        ----------
        df              : analysis DataFrame (long format for rating/disagreement modes;
                          any format for verbosity/aggregated since no pivot is needed)
        predictor_name  : override the default llm_prediction_col
        dims_to_exclude : dimension names to drop before fitting
        reset           : clear previous results before running (default True)
        """
        if reset:
            self.reset()

        predictor_name = predictor_name or self.llm_prediction_col

        for group_keys, group_df in df.groupby(self.experimental_groups):
            data = self.prepare_group_data(
                group_df, predictor_name, dims_to_exclude)
            if data is None:
                print(
                    f"Skipping {group_keys}: data preparation returned nothing.")
                continue

            y, X = data["y"], data["X"]
            if len(y) < (X.shape[1] + 2):
                print(
                    f"Skipping {group_keys}: not enough observations ({len(y)}).")
                continue

            try:
                result = self._fit_ols(y, X, data["clusters"])
                label = self._make_model_label(group_keys)
                self.results[label] = result
                self.run_data[label] = data
                metric = (
                    f"AIC={result.aic:.2f}"
                    if hasattr(result, "aic")
                    else f"R²={result.rsquared:.3f}"
                )
                print(f"✓ {label} | n={result.nobs:.0f} | {metric}")
            except Exception as exc:
                print(f"Error fitting {group_keys}: {exc}")

        return self.results

    def run_custom_regression(
        self,
        run_label: str,
        selected_vars: List[str],
        new_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Refit a stored run with a custom subset of predictors.

        Useful for manual variable selection or sensitivity checks without
        re-running prepare_group_data.
        """
        if run_label not in self.run_data:
            raise KeyError(f"Unknown run label: {run_label!r}")
        if not selected_vars:
            raise ValueError(
                "Provide at least one predictor in selected_vars.")

        base = self.run_data[run_label]
        long_df = base["long_df"].copy()

        missing = [v for v in selected_vars if v not in long_df.columns]
        if missing:
            raise ValueError(f"Variables not found in long_df: {missing}")

        # disagreement mode stores its own dep_var key; all others use self.dep_var
        dep_var = base.get("dep_var", self.dep_var)
        y = long_df[dep_var]
        X = long_df[selected_vars].copy()

        # preserve original clustering decision
        clusters = base["clusters"]
        if clusters is not None:
            # re-align cluster series to current long_df index
            clusters = (
                long_df[self.cluster_col]
                if self.cluster_col in long_df.columns
                else None
            )

        result = self._fit_ols(y, X, clusters)
        label = new_label or f"{run_label}__{'_+_'.join(selected_vars)}"

        self.results[label] = result
        self.run_data[label] = {
            **base,
            "y": y,
            "X": X,
            "clusters": clusters,
            "iv_cols": selected_vars,
            "formula": f"{dep_var} ~ " + " + ".join(selected_vars),
        }

        metric = (
            f"AIC={result.aic:.2f}"
            if hasattr(result, "aic")
            else f"R²={result.rsquared:.3f}"
        )
        print(f"✓ Custom: {label} | n={result.nobs:.0f} | {metric}")
        return self.results

    def compute_vifs(
        self,
        df: Optional[pd.DataFrame] = None,
        run_labels: Optional[List[str]] = None,
        threshold: float = 5.0,
    ) -> pd.DataFrame:
        """
        Compute Variance Inflation Factors for each predictor in each group.

        Uses the design matrix X already built by prepare_group_data, so
        encoding, pivoting, and melting are handled exactly as in the regression.
        No need to re-specify covariates or dimensions.

        Parameters
        ----------
        df          : Raw DataFrame to prepare on the fly. When omitted the
                      method reuses X from the stored run_data (i.e. after
                      run_regression() has been called). Providing df lets you
                      inspect VIFs before fitting any models.
        run_labels  : Subset of labels to compute VIFs for. Only relevant when
                      df is None; ignored otherwise.
        threshold   : Flag predictors whose VIF exceeds this value (default 5.0).

        Returns
        -------
        pd.DataFrame with columns:
            group, predictor, vif, flagged
        Sorted by group then vif descending.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # ------------------------------------------------------------------
        # Collect (label -> X) pairs from stored run_data or fresh preparation
        # ------------------------------------------------------------------
        if df is not None:
            groups: Dict[str, pd.DataFrame] = {}
            for group_keys, group_df in df.groupby(self.experimental_groups):
                label = self._make_model_label(group_keys)
                data = self.prepare_group_data(group_df)
                if data is not None:
                    groups[label] = data["X"]
        else:
            if not self.run_data:
                raise RuntimeError(
                    "No stored runs found. Call run_regression() first, "
                    "or pass df= to compute VIFs on the fly."
                )
            labels = run_labels if run_labels is not None else list(
                self.run_data.keys())
            groups = {label: self.run_data[label]["X"] for label in labels}

        # ------------------------------------------------------------------
        # Compute VIF per group
        # ------------------------------------------------------------------
        rows = []
        for label, X in groups.items():
            X = X.dropna()

            if X.shape[1] < 2:
                # VIF is undefined with a single predictor — record NaN
                for col in X.columns:
                    rows.append({"group": label, "predictor": col,
                                 "vif": float("nan"), "flagged": False})
                continue

            # statsmodels VIF expects the constant included in the matrix
            X_const = sm.add_constant(X, has_constant="add")
            predictor_cols = [c for c in X_const.columns if c != "const"]
            X_arr = X_const.values.astype(float)

            for col in predictor_cols:
                idx = X_const.columns.tolist().index(col)
                vif_val = float(variance_inflation_factor(X_arr, idx))
                rows.append({
                    "group": label,
                    "predictor": col,
                    "vif": round(vif_val, 3),
                    "flagged": vif_val > threshold,
                })

        return (
            pd.DataFrame(rows, columns=[
                         "group", "predictor", "vif", "flagged"])
            .sort_values(["group", "vif"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def summarize(self) -> pd.DataFrame:
        """Return a DataFrame of fit statistics for all stored results, sorted by adj R²."""
        rows = [
            {
                "label": label,
                "mode": self.mode,
                "nobs": getattr(r, "nobs", np.nan),
                "r_squared": getattr(r, "rsquared", np.nan),
                "adj_r_squared": getattr(r, "rsquared_adj", np.nan),
                "aic": getattr(r, "aic", np.nan),
                "bic": getattr(r, "bic", np.nan),
                "condition_number": getattr(r, "condition_number", np.nan),
            }
            for label, r in self.results.items()
        ]
        return pd.DataFrame(rows).sort_values("adj_r_squared", ascending=False)
