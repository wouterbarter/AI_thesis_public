# Verbosity bias

from __future__ import annotations
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import numpy as np
from typing import Any, Dict, List, Optional
import warnings
from scipy.stats import spearmanr
from typing import Optional
import itertools


def compute_verbosity_correlations(
    final_df: pd.DataFrame,
    score_column: str = 'mean_rating',
    input_length_column: str = 'raw_input_length',
    prompt_filter: Optional[str] = 'Formative',
    model_name_col: Optional[str] = 'model_name_clean',
    dimension_name_col: Optional[str] = 'dimension_name_clean'
) -> pd.DataFrame:
    """
    Computes the Spearman correlation between input length and scores.
    Returns a DataFrame containing the raw correlation data, p-values, and sample sizes.
    """
    # 1. Filter the dataframe if a specific prompt_name is requested
    if prompt_filter:
        if 'prompt_name' not in final_df.columns:
            print("Warning: 'prompt_name' column not found. Processing entire DataFrame.")
            df_working = final_df
        else:
            df_working = final_df[final_df['prompt_name'] == prompt_filter]
    else:
        df_working = final_df

    # 2. Calculate correlation per model AND per subdimension
    results = []

    # Grouping by model and dimension
    for (model, dimension), group in df_working.groupby([model_name_col, dimension_name_col]):
        valid_data = group[[input_length_column, score_column]].dropna()

        if len(valid_data) > 1:
            corr, p_val = spearmanr(
                valid_data[input_length_column], valid_data[score_column]
            )
            results.append({
                'model_name': model,
                'dimension_name': dimension,
                'verbosity_corr': corr,
                'p_value': p_val,
                'n_samples': len(valid_data)
            })

    # 3. Return as a clean dataframe for easy viewing/exporting
    return pd.DataFrame(results)


def compute_length_correlations(
    df: pd.DataFrame,
    text_cols: list,
    rating_col: str = 'mean_rating',
    group_col: str = 'dimension_name_clean',
    model_filter: str = None,
    model_col: str = 'model_name',
    compute_deltas: bool = True
) -> pd.DataFrame:
    """
    Computes Pearson correlations between LLM ratings and text length variants,
    grouped by a specific dimension (e.g., formative criteria). 
    Optionally calculates the delta between text column correlations.

    Args:
        df: The main input DataFrame.
        text_cols: List of string column names containing text length metrics 
                   (e.g., ['log_requirements_word_count', 'log_text_word_count']).
        rating_col: The column containing the LLM rating.
        group_col: The column to group by (usually the psychometric dimension).
        model_filter: Specific model string to filter by (e.g., 'Qwen 3.5 Formative').
                      If None, runs on the entire DataFrame.
        model_col: The column used for filtering the model.
        compute_deltas: If True, calculates pairwise differences between the text_cols.

    Returns:
        A sorted pandas DataFrame with correlations and deltas.
    """
    # 1. Filter dataframe if a specific model is requested
    if model_filter:
        working_df = df[df[model_col] == model_filter].copy()
    else:
        working_df = df.copy()

    results = []

    # 2. Group by the specified dimension
    grouped = working_df.groupby(group_col, observed=True)

    for name, group in grouped:
        row = {group_col: name}
        for col in text_cols:
            # Calculate correlation; pandas corr() automatically handles NaNs
            corr = group[rating_col].corr(group[col])
            row[f'Corr_with_{col}'] = corr
        results.append(row)

    results_df = pd.DataFrame(results)

    # 3. Compute pairwise deltas if requested
    if compute_deltas and len(text_cols) > 1:
        # Create all unique pairs from the provided text_cols list
        pairs = list(itertools.combinations(text_cols, 2))
        for col1, col2 in pairs:
            delta_col_name = f'Delta ({col1} minus {col2})'
            results_df[delta_col_name] = results_df[f'Corr_with_{col1}'] - \
                results_df[f'Corr_with_{col2}']

    # 4. Sort by the first text column's correlation to establish a logical hierarchy
    if text_cols:
        first_col_name = f'Corr_with_{text_cols[0]}'
        results_df = results_df.sort_values(
            by=first_col_name, ascending=False).reset_index(drop=True)

    return results_df

# RUNNER DIAGNOSTICS


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


def collect_all_diagnostics(runner) -> Dict[str, Dict[str, Any]]:
    """
    Run collect_run_diagnostics for every label in runner.results.
    Labels are normalized to clean strings to prevent downstream parsing errors.
    """
    def clean(l):
        if isinstance(l, tuple):
            return " ".join(str(x) for x in l)
        return str(l)

    return {
        clean(label): collect_run_diagnostics(
            result=result,
            run_data=runner.run_data[label],  # This relies on labels matching
            dep_var=runner.dep_var,
            cluster_col=runner.cluster_col,
        )
        for label, result in runner.results.items()
    }


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


def style_correlation_matrix(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Applies a clean coolwarm background gradient to the correlation and delta 
    columns of the resulting dataframe for easy visual diagnostic scanning.
    """
    # Identify numeric columns that we calculated
    target_cols = [col for col in df.columns if col.startswith(
        'Corr_with_') or col.startswith('Delta')]

    return df.style.background_gradient(cmap='coolwarm', subset=target_cols).format(precision=3)


def compact_vif_threshold_summary(
    vif_df: pd.DataFrame,
    group_col: str = "group",
    vif_col: str = "vif",
    threshold: float = 5.0,
    high_threshold: float = 10.0
) -> pd.DataFrame:
    """
    Produces a minimal VIF summary:
    max VIF, number above 5, number above 10, and interpretation.
    """

    df = vif_df.copy()
    df[vif_col] = pd.to_numeric(df[vif_col], errors="coerce")

    out = (
        df.dropna(subset=[vif_col])
        .groupby(group_col)
        .agg(
            **{
                "Max VIF": (vif_col, "max"),
                "Mean VIF": (vif_col, "mean"),
                f"N VIF > {threshold:g}": (vif_col, lambda x: int((x > threshold).sum())),
                f"N VIF > {high_threshold:g}": (vif_col, lambda x: int((x > high_threshold).sum())),
            }
        )
        .reset_index()
        .rename(columns={group_col: "Group"})
    )

    def interpret(row):
        if row[f"N VIF > {high_threshold:g}"] > 0:
            return "Substantial"
        if row[f"N VIF > {threshold:g}"] > 0:
            return "Moderate"
        return "Limited"

    out["Multicollinearity"] = out.apply(interpret, axis=1)

    out["Max VIF"] = out["Max VIF"].map(lambda x: f"{x:.2f}")
    out["Mean VIF"] = out["Mean VIF"].map(lambda x: f"{x:.2f}")

    return out
