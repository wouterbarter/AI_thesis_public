from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import Markdown, clear_output, display
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import ipywidgets as widgets
except ImportError:  # pragma: no cover
    widgets = None


class FeedbackQARegressionRunner:
    """
    Regression runner for FeedbackQA-style experiments.

    Design goals:
    - Mirror the workflow of the user's existing regression runner
    - Keep the notebook interface high-level
    - Store fitted models in `self.results`
    - Store per-run diagnostics in `self.diagnostics`
    - Store prepared run data in `self.run_data` for custom refits
    """

    def __init__(
        self,
        experimental_groups: Optional[List[str]] = None,
        prompt_hash_map: Optional[Mapping[str, str]] = None,
        dep_var: str = "raw_human_rating",
        llm_prediction_col: str = "mean_rating",
        entropy_col: str = "normalized_entropy",
        dimension_col: str = "dimension_name",
        model_col: str = "model_name",
        prompt_col: str = "prompt_id",
        input_col: str = "input_id",
        cluster_col: str = "input_id",
        score_cols: Sequence[str] = ("score_1", "score_2"),
        include_entropy: bool = True,
    ) -> None:
        self.experimental_groups = experimental_groups or [
            "model_name", "prompt_id"]
        self.prompt_hash_map = dict(prompt_hash_map or {})

        self.dep_var = dep_var
        self.llm_prediction_col = llm_prediction_col
        self.entropy_col = entropy_col
        self.dimension_col = dimension_col
        self.model_col = model_col
        self.prompt_col = prompt_col
        self.input_col = input_col
        self.cluster_col = cluster_col
        self.score_cols = list(score_cols)
        self.include_entropy = include_entropy

        self.results: Dict[str, Any] = {}
        self.diagnostics: Dict[str, Dict[str, Any]] = {}
        self.run_data: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------------
    # Core helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _flatten_columns(columns) -> List[str]:
        flat_cols = []
        for col in columns:
            if isinstance(col, tuple):
                parts = [str(c) for c in col if c not in (None, "")]
                flat_cols.append("_".join(parts).strip("_"))
            else:
                flat_cols.append(str(col))
        return flat_cols

    def _infer_holistic(self, group_df: pd.DataFrame) -> bool:
        unique_dims = group_df[self.dimension_col].dropna().unique()
        return len(unique_dims) == 1

    def _make_model_label(self, group_keys: Any) -> str:
        """
        Build human-readable label from grouped keys.

        Default behavior for ['model_name', 'prompt_id']:
            <model_name>_<prompt_description>
        """
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        if len(group_keys) == 2 and self.experimental_groups == [self.model_col, self.prompt_col]:
            model_name, prompt_id = group_keys
            prompt_desc = self.prompt_hash_map.get(prompt_id, str(prompt_id))
            return f"{model_name}_{prompt_desc}"

        label_parts = []
        for group_name, key in zip(self.experimental_groups, group_keys):
            if group_name == self.prompt_col:
                key = self.prompt_hash_map.get(key, str(key))
            label_parts.append(f"{group_name}={key}")

        return " | ".join(label_parts)

    def prepare_group_data(
        self,
        group_df: pd.DataFrame,
        predictor_name: Optional[str] = None,
        dims_to_exclude: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare a single grouped dataset for regression.

        Returns a dict with:
            y, X, clusters, iv_cols, formula,
            wide_df, long_df, is_holistic, dimensions
        """
        predictor_name = predictor_name or self.llm_prediction_col
        df = group_df.copy()

        if dims_to_exclude:
            df = df[~df[self.dimension_col].isin(dims_to_exclude)].copy()

        if df.empty:
            return None

        is_holistic = self._infer_holistic(df)
        unique_dims = sorted(df[self.dimension_col].dropna().unique().tolist())

        if is_holistic:
            keep_cols = [
                self.input_col,
                self.model_col,
                *self.score_cols,
                predictor_name,
            ]
            keep_cols = [c for c in keep_cols if c in df.columns]

            wide_df = (
                df[keep_cols]
                .drop_duplicates(subset=[self.input_col])
                .copy()
            )

            iv_cols = [predictor_name]

        else:
            index_cols = [self.input_col, self.model_col, *self.score_cols]
            index_cols = [c for c in index_cols if c in df.columns]

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

            iv_cols = [
                col for col in wide_df.columns
                if col.startswith(f"{predictor_name}_")
            ]

        if not iv_cols:
            return None

        score_cols_present = [
            c for c in self.score_cols if c in wide_df.columns]
        if not score_cols_present:
            return None

        id_vars = [c for c in [self.input_col,
                               self.model_col, *iv_cols] if c in wide_df.columns]

        long_df = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=score_cols_present,
            var_name="rater_id",
            value_name=self.dep_var,
        ).dropna(subset=[self.dep_var])

        if long_df.empty:
            return None

        y = long_df[self.dep_var]
        X = sm.add_constant(long_df[iv_cols], has_constant="add")
        clusters = long_df[self.cluster_col] if self.cluster_col in long_df.columns else None
        formula = f"{self.dep_var} ~ " + " + ".join(iv_cols)

        return {
            "y": y,
            "X": X,
            "clusters": clusters,
            "iv_cols": iv_cols,
            "formula": formula,
            "wide_df": wide_df,
            "long_df": long_df,
            "is_holistic": is_holistic,
            "dimensions": unique_dims,
        }

    def _fit_model(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        offset: Optional[pd.Series] = None,
        clusters: Optional[pd.Series] = None,
    ):
        """
        Fit clustered OLS.

        `offset` is ignored; it is only here to keep parity with the user's
        broader regression framework.
        """
        model = sm.OLS(y, X)

        if clusters is not None:
            res = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": clusters},
            )
        else:
            res = model.fit()

        return res

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------

    def _compute_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        vif_df = pd.DataFrame(
            {
                "variable": X.columns,
                "VIF": [
                    variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])
                ],
            }
        )
        return vif_df

    def _compute_correlations(
        self,
        long_df: pd.DataFrame,
        iv_cols: List[str],
    ) -> pd.DataFrame:
        corr_cols = iv_cols + [self.dep_var]
        corr_cols = [c for c in corr_cols if c in long_df.columns]
        return long_df[corr_cols].corr(numeric_only=True)

    def _compute_standardized_condition_number(
        self,
        long_df: pd.DataFrame,
        iv_cols: List[str],
    ) -> float:
        X = long_df[iv_cols].copy().dropna()

        if X.empty:
            return np.nan

        std = X.std(ddof=0).replace(0, np.nan)
        X_std = (X - X.mean()) / std
        X_std = X_std.dropna(axis=1)

        if X_std.shape[1] == 0:
            return np.nan

        return float(np.linalg.cond(X_std.to_numpy()))

    def _fit_bivariate_models(
        self,
        long_df: pd.DataFrame,
        iv_cols: List[str],
    ) -> Dict[str, Any]:
        models = {}

        for iv in iv_cols:
            y = long_df[self.dep_var]
            X = sm.add_constant(long_df[[iv]], has_constant="add")
            clusters = long_df[self.cluster_col] if self.cluster_col in long_df.columns else None

            model = self._fit_model(y=y, X=X, clusters=clusters)
            models[iv] = model

        return models

    @staticmethod
    def _summarize_bivariate_models(models: Dict[str, Any]) -> pd.DataFrame:
        rows = []

        for iv, model in models.items():
            rows.append(
                {
                    "variable": iv,
                    "coef": model.params.get(iv, np.nan),
                    "std_err": model.bse.get(iv, np.nan),
                    "p_value": model.pvalues.get(iv, np.nan),
                    "r_squared": getattr(model, "rsquared", np.nan),
                    "adj_r_squared": getattr(model, "rsquared_adj", np.nan),
                    "aic": getattr(model, "aic", np.nan),
                    "bic": getattr(model, "bic", np.nan),
                    "nobs": getattr(model, "nobs", np.nan),
                }
            )

        return pd.DataFrame(rows).sort_values("r_squared", ascending=False)

    @staticmethod
    def _build_coef_table(model: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": model.params,
                "std_err": model.bse,
                "z_or_t": getattr(model, "tvalues", pd.Series(index=model.params.index, dtype=float)),
                "p_value": model.pvalues,
                "ci_lower": model.conf_int()[0],
                "ci_upper": model.conf_int()[1],
            }
        )

    def _build_metrics_table(
        self,
        label: str,
        model: Any,
        standardized_condition_number: float,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "label": label,
                    "nobs": getattr(model, "nobs", np.nan),
                    "r_squared": getattr(model, "rsquared", np.nan),
                    "adj_r_squared": getattr(model, "rsquared_adj", np.nan),
                    "aic": getattr(model, "aic", np.nan),
                    "bic": getattr(model, "bic", np.nan),
                    "raw_condition_number": getattr(model, "condition_number", np.nan),
                    "standardized_condition_number": standardized_condition_number,
                }
            ]
        )

    def _compute_diagnostics(
        self,
        label: str,
        model: Any,
        data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        X = data_dict["X"]
        long_df = data_dict["long_df"]
        iv_cols = data_dict["iv_cols"]

        vif_df = self._compute_vif(X)
        corr_df = self._compute_correlations(long_df=long_df, iv_cols=iv_cols)
        standardized_condition_number = self._compute_standardized_condition_number(
            long_df=long_df,
            iv_cols=iv_cols,
        )

        bivariate_models = self._fit_bivariate_models(
            long_df=long_df, iv_cols=iv_cols)
        bivariate_summary = self._summarize_bivariate_models(bivariate_models)
        coef_table = self._build_coef_table(model)
        metrics_table = self._build_metrics_table(
            label=label,
            model=model,
            standardized_condition_number=standardized_condition_number,
        )

        return {
            "metrics": metrics_table,
            "coefficients": coef_table,
            "vif": vif_df,
            "correlations": corr_df,
            "bivariate_summary": bivariate_summary,
            "bivariate_models": bivariate_models,
        }

    # ---------------------------------------------------------------------
    # Public run methods
    # ---------------------------------------------------------------------

    def run_regression(
        self,
        df: pd.DataFrame,
        predictor_name: Optional[str] = None,
        dims_to_exclude: Optional[List[str]] = None,
        collect_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """
        Main execution loop for FeedbackQA-based LLM regressions.
        """
        predictor_name = predictor_name or self.llm_prediction_col
        grouped = df.groupby(self.experimental_groups)

        for group_keys, group_df in grouped:
            data_dict = self.prepare_group_data(
                group_df=group_df,
                predictor_name=predictor_name,
                dims_to_exclude=dims_to_exclude,
            )

            if data_dict is None:
                print(f"Skipping {group_keys}: Data preparation failed.")
                continue

            y = data_dict.get("y")
            X = data_dict.get("X")

            if y is None or X is None:
                print(f"Skipping {group_keys}: Missing y or X data.")
                continue

            if len(y) < (X.shape[1] + 2):
                print(f"Skipping {group_keys}: Insufficient observations.")
                continue

            clusters = data_dict.get("clusters")

            try:
                res = self._fit_model(
                    y=y,
                    X=X,
                    clusters=clusters,
                )

                label = self._make_model_label(group_keys)
                self.results[label] = res
                self.run_data[label] = data_dict

                if collect_diagnostics:
                    self.diagnostics[label] = self._compute_diagnostics(
                        label=label,
                        model=res,
                        data_dict=data_dict,
                    )

                metric = (
                    f"AIC={res.aic:.2f}"
                    if hasattr(res, "aic")
                    else f"R2={res.rsquared:.3f}"
                )
                print(
                    f"✓ Model fitted: {label} | {res.nobs:.0f} observations | {metric}")

            except Exception as e:
                print(f"Error fitting model for {group_keys}: {e}")

        return self.results

    def run_custom_model(
        self,
        run_label: str,
        selected_vars: List[str],
        new_label: Optional[str] = None,
        collect_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """
        Refit a stored run with a custom subset of IVs.
        """
        if run_label not in self.run_data:
            raise KeyError(f"Run label not found: {run_label}")

        if not selected_vars:
            raise ValueError("Please provide at least one predictor.")

        base_data = self.run_data[run_label]
        long_df = base_data["long_df"].copy()

        missing_vars = [v for v in selected_vars if v not in long_df.columns]
        if missing_vars:
            raise ValueError(
                f"Selected variables not found in data: {missing_vars}")

        y = long_df[self.dep_var]
        X = sm.add_constant(long_df[selected_vars], has_constant="add")
        clusters = long_df[self.cluster_col] if self.cluster_col in long_df.columns else None

        res = self._fit_model(y=y, X=X, clusters=clusters)

        fitted_label = new_label or f"{run_label}__custom__{' + '.join(selected_vars)}"

        data_dict = {
            **base_data,
            "y": y,
            "X": X,
            "clusters": clusters,
            "iv_cols": selected_vars,
            "formula": f"{self.dep_var} ~ " + " + ".join(selected_vars),
        }

        self.results[fitted_label] = res
        self.run_data[fitted_label] = data_dict

        if collect_diagnostics:
            self.diagnostics[fitted_label] = self._compute_diagnostics(
                label=fitted_label,
                model=res,
                data_dict=data_dict,
            )

        metric = (
            f"AIC={res.aic:.2f}"
            if hasattr(res, "aic")
            else f"R2={res.rsquared:.3f}"
        )
        print(
            f"✓ Custom model fitted: {fitted_label} | {res.nobs:.0f} observations | {metric}")

        return self.results

    # ---------------------------------------------------------------------
    # Summary helpers
    # ---------------------------------------------------------------------

    def summarize_results(self) -> pd.DataFrame:
        rows = []

        for label, model in self.results.items():
            rows.append(
                {
                    "label": label,
                    "nobs": getattr(model, "nobs", np.nan),
                    "r_squared": getattr(model, "rsquared", np.nan),
                    "adj_r_squared": getattr(model, "rsquared_adj", np.nan),
                    "aic": getattr(model, "aic", np.nan),
                    "bic": getattr(model, "bic", np.nan),
                    "condition_number": getattr(model, "condition_number", np.nan),
                }
            )

        return pd.DataFrame(rows).sort_values("adj_r_squared", ascending=False)

    def summarize_diagnostics(self) -> pd.DataFrame:
        rows = []

        for label, diag in self.diagnostics.items():
            metrics = diag.get("metrics")
            if metrics is None or metrics.empty:
                continue
            rows.append(metrics.iloc[0].to_dict())

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values("adj_r_squared", ascending=False)

    # ---------------------------------------------------------------------
    # Interactive widgets
    # ---------------------------------------------------------------------

    @staticmethod
    def _display_any(obj: Any) -> None:
        if isinstance(obj, pd.DataFrame):
            display(obj)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key == "bivariate_models":
                    continue
                display(Markdown(f"**{key}**"))
                FeedbackQARegressionRunner._display_any(value)
        elif hasattr(obj, "summary"):
            print(obj.summary())
        else:
            display(obj)

    def interactive_results_selector(
        self,
        data_dict: Optional[Mapping[str, Any]] = None,
        description: str = "Select option:",
        width: str = "700px",
    ):
        if widgets is None:
            raise ImportError(
                "ipywidgets is required for interactive selectors.")

        source = dict(data_dict or self.results)

        if not source:
            print("No results to display.")
            return

        dropdown = widgets.Dropdown(
            options=list(source.keys()),
            description=description,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=width),
        )

        output = widgets.Output()

        def render(selected_key: str) -> None:
            with output:
                clear_output(wait=True)
                display(Markdown(f"### {selected_key}"))
                self._display_any(source[selected_key])

        def on_change(change):
            if change["name"] == "value" and change["new"] is not None:
                render(change["new"])

        dropdown.observe(on_change, names="value")
        display(dropdown, output)
        render(dropdown.value)

    def make_custom_model_widget(
        self,
        run_label: str,
        description: str = "Variables:",
        rows: int = 12,
        width: str = "85%",
    ):
        if widgets is None:
            raise ImportError(
                "ipywidgets is required for custom model widgets.")

        if run_label not in self.run_data:
            raise KeyError(f"Run label not found: {run_label}")

        iv_cols = self.run_data[run_label]["iv_cols"]

        variable_selector = widgets.SelectMultiple(
            options=iv_cols,
            value=tuple(iv_cols),
            rows=rows,
            description=description,
            layout=widgets.Layout(width=width),
        )

        label_box = widgets.Text(
            value=f"{run_label}__custom",
            description="Label:",
            layout=widgets.Layout(width=width),
        )

        run_button = widgets.Button(
            description="Run Custom Model",
            button_style="success",
            icon="play",
        )

        output_area = widgets.Output()

        def run_custom_regression(_):
            with output_area:
                clear_output(wait=True)
                selected_vars = list(variable_selector.value)

                if not selected_vars:
                    print("⚠️ Please select at least one variable.")
                    return

                new_label = label_box.value.strip() or None

                self.run_custom_model(
                    run_label=run_label,
                    selected_vars=selected_vars,
                    new_label=new_label,
                    collect_diagnostics=True,
                )

                print("\nModel summary:\n")
                print(self.results[new_label].summary())

        run_button.on_click(run_custom_regression)

        ui = widgets.VBox(
            [variable_selector, label_box, run_button, output_area])
        display(ui)


# -------------------------------------------------------------------------
# Module-level convenience wrappers
# -------------------------------------------------------------------------

def interactive_regression_results_selector(
    data_dict: Mapping[str, Any],
    description: str = "Select option:",
    width: str = "700px",
):
    """
    Generic interactive selector for:
    - fitted model results
    - diagnostics dicts
    - summary DataFrames
    """
    runner = FeedbackQARegressionRunner()
    runner.interactive_results_selector(
        data_dict=data_dict,
        description=description,
        width=width,
    )
