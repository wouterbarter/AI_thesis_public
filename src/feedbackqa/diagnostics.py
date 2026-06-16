# from __future__ import annotations
# import warnings

# from typing import Any, Dict, List, Optional

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor


# # ---------------------------------------------------------------------------
# # Individual diagnostic functions
# # Each operates on raw data so they can be called outside of a runner context.
# # ---------------------------------------------------------------------------

# def compute_vif(X: pd.DataFrame, add_constant: bool = True) -> pd.DataFrame:
#     """
#     Variance inflation factors for all columns in X.

#     Parameters
#     ----------
#     X : predictor matrix (no constant)
#     add_constant : prepend a constant column before computing VIF
#     """
#     X_work = X.dropna().copy()
#     if add_constant:
#         X_work = sm.add_constant(X_work, has_constant="add")
#     return pd.DataFrame(
#         {
#             "variable": X_work.columns,
#             "VIF": [
#                 variance_inflation_factor(X_work.values, i)
#                 for i in range(X_work.shape[1])
#             ],
#         }
#     )


# def compute_correlations(
#     long_df: pd.DataFrame,
#     iv_cols: List[str],
#     dep_var: str = "raw_human_rating",
# ) -> pd.DataFrame:
#     """Pearson correlation matrix for predictors and the dependent variable."""
#     cols = [c for c in [*iv_cols, dep_var] if c in long_df.columns]
#     return long_df[cols].corr(numeric_only=True)


# def compute_standardized_condition_number(X: pd.DataFrame) -> float:
#     """
#     Condition number of the standardised predictor matrix.

#     More informative than the raw condition number for detecting
#     multicollinearity independently of variable scaling.
#     """
#     X_work = X.dropna().copy()
#     if X_work.empty:
#         return np.nan

#     std = X_work.std(ddof=0).replace(0, np.nan)
#     X_std = ((X_work - X_work.mean()) / std).dropna(axis=1)
#     if X_std.shape[1] == 0:
#         return np.nan

#     return float(np.linalg.cond(X_std.to_numpy()))


# def fit_bivariate_models(
#     long_df: pd.DataFrame,
#     iv_cols: List[str],
#     dep_var: str = "raw_human_rating",
#     cluster_col: str = "input_id",
# ) -> Dict[str, Any]:
#     models: Dict[str, Any] = {}
#     clusters = long_df[cluster_col] if cluster_col in long_df.columns else None

#     for iv in iv_cols:
#         # skip zero-variance predictors — OLS is undefined and R² would be meaningless
#         if long_df[iv].nunique() <= 1:
#             continue

#         y = long_df[dep_var]
#         X = sm.add_constant(long_df[[iv]], has_constant="add")
#         model = sm.OLS(y, X)

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", RuntimeWarning)
#             res = (
#                 model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
#                 if clusters is not None
#                 else model.fit()
#             )
#         models[iv] = res

#     return models


# def summarize_bivariate_models(models: Dict[str, Any]) -> pd.DataFrame:
#     """
#     Compact summary table for a dict returned by fit_bivariate_models.
#     Sorted by R² descending.
#     """
#     rows = []
#     for iv, model in models.items():
#         # rsquared is computed lazily upon access, so we must catch warnings here
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", RuntimeWarning)
#             rows.append({
#                 "variable": iv,
#                 "coef": model.params.get(iv, np.nan),
#                 "std_err": model.bse.get(iv, np.nan),
#                 "p_value": model.pvalues.get(iv, np.nan),
#                 "r_squared": getattr(model, "rsquared", np.nan),
#                 "adj_r_squared": getattr(model, "rsquared_adj", np.nan),
#                 "aic": getattr(model, "aic", np.nan),
#                 "bic": getattr(model, "bic", np.nan),
#                 "nobs": getattr(model, "nobs", np.nan),
#             })

#     if not rows:
#         return pd.DataFrame()

#     return pd.DataFrame(rows).sort_values("r_squared", ascending=False)


# def make_coefficient_table(result: Any) -> pd.DataFrame:
#     """Full coefficient table with CIs for a fitted statsmodels result."""
#     ci = result.conf_int()
#     return pd.DataFrame(
#         {
#             "coef": result.params,
#             "std_err": result.bse,
#             "t": getattr(result, "tvalues", pd.Series(index=result.params.index, dtype=float)),
#             "p_value": result.pvalues,
#             "ci_lower": ci[0],
#             "ci_upper": ci[1],
#         }
#     )


# # ---------------------------------------------------------------------------
# # Runner-level helpers (operate on a FeedbackQARegressionRunner instance)
# # ---------------------------------------------------------------------------

# def collect_run_diagnostics(
#     result: Any,
#     run_data: Dict[str, Any],
#     dep_var: str = "raw_human_rating",
#     cluster_col: str = "input_id",
# ) -> Dict[str, Any]:
#     """
#     Full diagnostic bundle for a single fitted run.
#     """
#     long_df = run_data["long_df"]
#     X = run_data["X"]
#     iv_cols = run_data["iv_cols"]

#     # if the outcome has no variance in this group, bivariate OLS is meaningless
#     if long_df[dep_var].nunique() <= 1:
#         biv_models = {}
#         biv_summary = pd.DataFrame()
#     else:
#         biv_models = fit_bivariate_models(
#             long_df, iv_cols, dep_var, cluster_col)
#         biv_summary = summarize_bivariate_models(biv_models)

#     # Wrap metrics evaluation to catch lazy-loading divide-by-zero warnings
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
#         metrics = pd.DataFrame([{
#             "nobs": getattr(result, "nobs", np.nan),
#             "r_squared": getattr(result, "rsquared", np.nan),
#             "adj_r_squared": getattr(result, "rsquared_adj", np.nan),
#             "aic": getattr(result, "aic", np.nan),
#             "bic": getattr(result, "bic", np.nan),
#             "raw_condition_number": getattr(result, "condition_number", np.nan),
#             "standardized_condition_number": compute_standardized_condition_number(X),
#         }])

#     return {
#         "metrics": metrics,
#         "coefficients": make_coefficient_table(result),
#         "vif": compute_vif(X),
#         "correlations": compute_correlations(long_df, iv_cols, dep_var),
#         "bivariate_summary": biv_summary,
#         "bivariate_models": biv_models,
#     }


# def collect_all_diagnostics(runner) -> Dict[str, Dict[str, Any]]:
#     """
#     Run collect_run_diagnostics for every label in runner.results.

#     Parameters
#     ----------
#     runner : FeedbackQARegressionRunner instance

#     Returns
#     -------
#     dict mapping label → diagnostic bundle
#     """
#     return {
#         label: collect_run_diagnostics(
#             result=result,
#             run_data=runner.run_data[label],
#             dep_var=runner.dep_var,
#             cluster_col=runner.cluster_col,
#         )
#         for label, result in runner.results.items()
#     }


# def extract_diagnostic_table(
#     diagnostics: Dict[str, Dict[str, Any]],
#     key: str,
# ) -> Dict[str, pd.DataFrame]:
#     """
#     Slice a single diagnostic table out of the full diagnostics dict.

#     Example
#     -------
#     vif_tables  = extract_diagnostic_table(diagnostics, "vif")
#     corr_tables = extract_diagnostic_table(diagnostics, "correlations")
#     biv_tables  = extract_diagnostic_table(diagnostics, "bivariate_summary")
#     """
#     return {
#         label: bundle[key]
#         for label, bundle in diagnostics.items()
#         if key in bundle
#     }


# def overview(diagnostics: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
#     """
#     Stack all per-run diagnostic tables into single DataFrames for side-by-side comparison.

#     Returns a dict with keys: metrics, coefficients, vif, bivariate_summary, correlations
#     Each DataFrame has a 'label' column identifying the run.
#     """
#     def stack(key: str) -> pd.DataFrame:
#         frames = []
#         for label, bundle in diagnostics.items():
#             if key not in bundle:
#                 continue
#             df = bundle[key].copy()
#             df.insert(0, "label", label)
#             frames.append(df)
#         return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

#     return {
#         "metrics":           stack("metrics"),
#         "coefficients":      stack("coefficients"),
#         "vif":               stack("vif"),
#         "bivariate_summary": stack("bivariate_summary"),
#         "correlations":      stack("correlations"),
#     }


from __future__ import annotations
from typing import Optional, List
from typing import Any, Dict, Union
import warnings

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Utility: wrap any callable to suppress RuntimeWarning (divide by zero etc.)
# ---------------------------------------------------------------------------

def _safe(fn, *args, fallback=np.nan, **kwargs):
    """Call fn(*args, **kwargs), returning fallback on RuntimeWarning or exception."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return fn(*args, **kwargs)
        except Exception:
            return fallback


def _safe_getattr(obj, attr):
    """getattr that suppresses RuntimeWarning on lazy-computed statsmodels properties."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return getattr(obj, attr, np.nan)
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# Individual diagnostic functions
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame, add_constant: bool = True) -> pd.DataFrame:
    """
    Variance inflation factors for all columns in X.

    Zero-variance columns are dropped before computation — they would
    cause a divide-by-zero inside variance_inflation_factor and are
    uninformative for multicollinearity anyway.
    """
    X_work = X.dropna().copy()

    # drop zero-variance columns (constant after dummification, etc.)
    varying = X_work.columns[X_work.nunique() > 1]
    X_work = X_work[varying]

    if X_work.empty:
        return pd.DataFrame(columns=["variable", "VIF"])

    if add_constant:
        X_work = sm.add_constant(X_work, has_constant="add")

    rows = []
    for i, col in enumerate(X_work.columns):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                vif = variance_inflation_factor(X_work.values, i)
            except Exception:
                vif = np.nan
        rows.append({"variable": col, "VIF": vif})

    return pd.DataFrame(rows)


def compute_correlations(
    long_df: pd.DataFrame,
    iv_cols: List[str],
    dep_var: str = "raw_human_rating",
) -> pd.DataFrame:
    """Pearson correlation matrix for predictors and the dependent variable."""
    cols = [c for c in [*iv_cols, dep_var] if c in long_df.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return long_df[cols].corr(numeric_only=True)


def compute_standardized_condition_number(X: pd.DataFrame) -> float:
    """
    Condition number of the standardised predictor matrix.

    More informative than the raw condition number for detecting
    multicollinearity independently of variable scaling.
    """
    X_work = X.dropna().copy()
    if X_work.empty:
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            std = X_work.std(ddof=0).replace(0, np.nan)
            X_std = ((X_work - X_work.mean()) / std).dropna(axis=1)
            if X_std.shape[1] == 0:
                return np.nan
            return float(np.linalg.cond(X_std.to_numpy()))
        except Exception:
            return np.nan


def fit_bivariate_models(
    long_df: pd.DataFrame,
    iv_cols: List[str],
    dep_var: str = "raw_human_rating",
    cluster_col: str = "input_id",
) -> Dict[str, Any]:
    """
    One OLS model per predictor, each regressed alone on dep_var.

    Skips zero-variance predictors and zero-variance outcomes silently.
    """
    models: Dict[str, Any] = {}

    if long_df[dep_var].nunique() <= 1:
        return models  # outcome has no variance — all bivariate R² would be 0/0

    clusters = long_df[cluster_col] if cluster_col in long_df.columns else None

    for iv in iv_cols:
        if long_df[iv].nunique() <= 1:
            continue  # zero-variance predictor

        y = long_df[dep_var]
        X = sm.add_constant(long_df[[iv]], has_constant="add")
        model = sm.OLS(y, X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                res = (
                    model.fit(cov_type="cluster", cov_kwds={
                              "groups": clusters})
                    if clusters is not None
                    else model.fit()
                )
                models[iv] = res
            except Exception:
                continue

    return models


def summarize_bivariate_models(models: Dict[str, Any]) -> pd.DataFrame:
    """
    Compact summary table for a dict returned by fit_bivariate_models.
    Sorted by R² descending.
    """
    if not models:
        return pd.DataFrame()

    rows = []
    for iv, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                rows.append({
                    "variable": iv,
                    "coef": model.params.get(iv, np.nan),
                    "std_err": model.bse.get(iv, np.nan),
                    "p_value": model.pvalues.get(iv, np.nan),
                    "r_squared": _safe_getattr(model, "rsquared"),
                    "adj_r_squared": _safe_getattr(model, "rsquared_adj"),
                    "aic": _safe_getattr(model, "aic"),
                    "bic": _safe_getattr(model, "bic"),
                    "nobs": _safe_getattr(model, "nobs"),
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("r_squared", ascending=False)


def make_coefficient_table(result: Any) -> pd.DataFrame:
    """Full coefficient table with CIs for a fitted statsmodels result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            ci = result.conf_int()
            return pd.DataFrame({
                "coef": result.params,
                "std_err": result.bse,
                "t": getattr(
                    result, "tvalues",
                    pd.Series(index=result.params.index, dtype=float)
                ),
                "p_value": result.pvalues,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            })
        except Exception:
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Runner-level helpers
# ---------------------------------------------------------------------------

def collect_run_diagnostics(
    result: Any,
    run_data: Dict[str, Any],
    dep_var: str = "raw_human_rating",
    cluster_col: str = "input_id",
) -> Dict[str, Any]:
    """
    Full diagnostic bundle for a single fitted run.

    All statsmodels property accesses that may trigger divide-by-zero
    (rsquared, aic, condition_number, etc.) are guarded via _safe_getattr.
    """
    long_df = run_data["long_df"]
    X = run_data["X"]
    iv_cols = run_data["iv_cols"]

    biv_models = fit_bivariate_models(long_df, iv_cols, dep_var, cluster_col)
    biv_summary = summarize_bivariate_models(biv_models)

    metrics = pd.DataFrame([{
        "nobs":                        _safe_getattr(result, "nobs"),
        "r_squared":                   _safe_getattr(result, "rsquared"),
        "adj_r_squared":               _safe_getattr(result, "rsquared_adj"),
        "pseudo_r_squared":            _safe_getattr(result, "prsquared"),
        "aic":                         _safe_getattr(result, "aic"),
        "bic":                         _safe_getattr(result, "bic"),
        "llf":                         _safe_getattr(result, "llf"),
        "raw_condition_number":        _safe_getattr(result, "condition_number"),
        "standardized_condition_number": compute_standardized_condition_number(X),
    }])

    return {
        "metrics":          metrics,
        "coefficients":     make_coefficient_table(result),
        "vif":              compute_vif(X),
        "correlations":     compute_correlations(long_df, iv_cols, dep_var),
        "bivariate_summary": biv_summary,
        "bivariate_models": biv_models,
    }


def _normalize_label_to_str(label: Any) -> str:
    """
    Converts any label (tuple, string, etc.) into a clean, flat string.
    Example: ('Gemma 4 Formative',) -> 'Gemma 4 Formative'
    """
    if isinstance(label, tuple):
        # Join elements if multiple, otherwise take the first
        return " ".join(str(x) for x in label).strip()
    return str(label).strip()


def collect_all_diagnostics(runner) -> Dict[str, Dict[str, Any]]:
    """
    Collects diagnostics, ensuring all keys are clean strings.
    """
    diagnostics = {}

    for original_label, result in runner.results.items():
        # Use the original key to get the data
        data = runner.run_data[original_label]

        # Create a clean string version for the dictionary key
        clean_key = _normalize_label_to_str(original_label)

        diagnostics[clean_key] = collect_run_diagnostics(
            result=result,
            run_data=data,
            dep_var=runner.dep_var,
            cluster_col=runner.cluster_col,
        )

    return diagnostics


# def collect_all_diagnostics(runner) -> Dict[str, Dict[str, Any]]:
#     """
#     Run collect_run_diagnostics for every label in runner.results.

#     Works with both FeedbackQARegressionRunner and BaseRegressionRunner
#     subclasses, as long as the runner exposes:
#         runner.results      : Dict[str, fitted result]
#         runner.run_data     : Dict[str, data dict with 'long_df', 'X', 'iv_cols']
#         runner.dep_var      : str
#         runner.cluster_col  : str
#     """
#     return {
#         label: collect_run_diagnostics(
#             result=result,
#             run_data=runner.run_data[label],
#             dep_var=runner.dep_var,
#             cluster_col=runner.cluster_col,
#         )
#         for label, result in runner.results.items()
#     }


def extract_diagnostic_table(
    diagnostics: Dict[str, Dict[str, Any]],
    key: str,
) -> Dict[str, pd.DataFrame]:
    """
    Slice a single diagnostic table out of the full diagnostics dict.

    Example
    -------
    vif_tables  = extract_diagnostic_table(diagnostics, "vif")
    corr_tables = extract_diagnostic_table(diagnostics, "correlations")
    biv_tables  = extract_diagnostic_table(diagnostics, "bivariate_summary")
    """
    return {
        label: bundle[key]
        for label, bundle in diagnostics.items()
        if key in bundle
    }


def overview(diagnostics: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Stack all per-run diagnostic tables into single DataFrames for side-by-side comparison.

    Returns a dict with keys: metrics, coefficients, vif, bivariate_summary, correlations
    Each DataFrame has a 'label' column identifying the run.
    """
    def stack(key: str) -> pd.DataFrame:
        frames = []
        for label, bundle in diagnostics.items():
            if key not in bundle:
                continue
            df = bundle[key].copy()
            if df.empty:
                continue
            df.insert(0, "label", label)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {
        "metrics":           stack("metrics"),
        "coefficients":      stack("coefficients"),
        "vif":               stack("vif"),
        "bivariate_summary": stack("bivariate_summary"),
        "correlations":      stack("correlations"),
    }


def generate_crossvalidation_alignment_table(cv_mean: pd.DataFrame, cv_entr: pd.DataFrame, human_base: pd.Series) -> pd.DataFrame:
    """
    Merges EV-only and EV+Entropy cross-validation results and appends the human psychometric ceiling.
    """
    # Isolate alignment metrics and apply specification suffixes
    metrics = ['RMSE', 'Pearson_r', 'Spearman_rho']
    df_base = cv_mean[metrics].add_suffix('_EV')
    df_ext = cv_entr[metrics].add_suffix('_EV+Entr')

    # Join specifications on model label
    df_comp = df_base.join(df_ext)

    # Format human baseline to match the comparative structure
    human_row = {
        'RMSE_EV': human_base['RMSE'],
        'RMSE_EV+Entr': np.nan,
        'Pearson_r_EV': human_base['Pearson_r'],
        'Pearson_r_EV+Entr': np.nan,
        'Spearman_rho_EV': human_base['Spearman_rho'],
        'Spearman_rho_EV+Entr': np.nan
    }
    human_df = pd.DataFrame([human_row], index=['Human-Human Baseline'])

    # Concatenate and enforce side-by-side metric ordering
    ordered_cols = [
        'RMSE_EV', 'RMSE_EV+Entr',
        'Pearson_r_EV', 'Pearson_r_EV+Entr',
        'Spearman_rho_EV', 'Spearman_rho_EV+Entr'
    ]
    return pd.concat([df_comp, human_df])[ordered_cols]


def generate_feedbackqa_cv_latex_table(
    cv_mean: pd.DataFrame,
    human_base: Optional[pd.Series] = None,
    model_order: Optional[List[str]] = None,
    condition_order: Optional[List[str]] = None,
    caption: str = "Cross-Validated Convergent Validity Performance on FeedbackQA",
    label: str = "tab:feedbackqa_cv",
    round_digits: int = 3
) -> str:
    """
    Creates a thesis-ready LaTeX table for FeedbackQA cross-validation results.

    The table reports EV-only model performance and optionally appends the
    human-human baseline.

    Expected input:
        cv_mean index: labels like "Gemma 4_Formative"
        cv_mean columns: RMSE, Pearson_r, Spearman_rho, Kendall_tau, optionally nobs
    """

    required_metrics = ["RMSE", "Pearson_r", "Spearman_rho", "Kendall_tau"]

    missing = [m for m in required_metrics if m not in cv_mean.columns]
    if missing:
        raise ValueError(
            f"Missing required metric columns: {missing}. "
            "Rerun evaluate_models() after adding Kendall_tau."
        )

    df = cv_mean.copy().reset_index()
    df = df.rename(columns={"label": "Group", "index": "Group"})

    if "Group" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Group"})

    # Split labels like "Qwen 3.5_Formative" into Model and Condition
    split_labels = df["Group"].astype(str).str.split("_", n=1, expand=True)
    df["Model"] = split_labels[0]
    df["Condition"] = split_labels[1].fillna("")

    df = df[["Model", "Condition"] + required_metrics]

    # Append human-human baseline
    if human_base is not None:
        human_row = {
            "Model": "Human-Human",
            "Condition": "Baseline",
            "RMSE": human_base.get("RMSE", np.nan),
            "Pearson_r": human_base.get("Pearson_r", np.nan),
            "Spearman_rho": human_base.get("Spearman_rho", np.nan),
            "Kendall_tau": human_base.get("Kendall_tau", np.nan),
        }
        df = pd.concat([df, pd.DataFrame([human_row])], ignore_index=True)

    # Default ordering
    if model_order is None:
        model_order = ["Gemma 4", "Llama 3.2",
                       "Qwen 3", "Qwen 3.5", "Human-Human"]

    if condition_order is None:
        condition_order = [
            "Holistic Naive",
            "Holistic Informed",
            "Formative",
            "Baseline"
        ]

    df["Model"] = pd.Categorical(
        df["Model"], categories=model_order, ordered=True)
    df["Condition"] = pd.Categorical(
        df["Condition"], categories=condition_order, ordered=True)

    df = df.sort_values(["Model", "Condition"]).reset_index(drop=True)

    # Format numeric values
    for col in required_metrics:
        df[col] = df[col].map(
            lambda x: f"{x:.{round_digits}f}" if pd.notnull(x) else "--"
        )

    # Rename columns for LaTeX
    table_df = df.rename(columns={
        "RMSE": r"RMSE $\downarrow$",
        "Pearson_r": r"Pearson's $r$ $\uparrow$",
        "Spearman_rho": r"Spearman's $\rho$ $\uparrow$",
        "Kendall_tau": r"Kendall's $\tau$ $\uparrow$"
    })

    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="llcccc",
        caption=caption,
        label=label
    )

    note = (
        "\n\\begin{flushleft}\n"
        "\\footnotesize\n"
        "\\textit{Note.} Values report out-of-fold cross-validated performance "
        "for EV-only specifications. RMSE measures prediction error, while "
        "Pearson's $r$, Spearman's $\\rho$, and Kendall's $\\tau$ measure "
        "alignment with human quality judgments. The human-human baseline "
        "compares the two available human annotations and is not a fitted model.\n"
        "\\end{flushleft}\n"
    )

    latex = latex.replace("\\end{table}", note + "\\end{table}")

    return latex
