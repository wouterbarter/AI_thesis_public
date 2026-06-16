# import pandas as pd
# import numpy as np


# def rating_entropy_distribution(df, model_col='model_name_clean', dim_col = 'dimension_name_clean', rating_col='mean_rating', entropy_col='normalized_entropy'):
#     """Computes scale utilization moments and the integer proximity index to diagnose quantization."""

#     # Calculate absolute distance to nearest integer
#     df['int_dist'] = (df[rating_col] - df[rating_col].round()).abs()

#     stats = df.groupby(model_col).agg({
#         rating_col: ['mean', 'std'],
#         entropy_col: ['mean', 'std'],
#         'int_dist': ['mean']
#     })

#     table_df = pd.DataFrame(index=stats.index)
#     table_df.index.name = 'Model'

#     table_df['Rating: Mean (SD)'] = (
#         stats[rating_col]['mean'].round(2).map('{:.2f}'.format) + " (" +
#         stats[rating_col]['std'].round(2).map('{:.2f}'.format) + ")"
#     )
#     table_df['Entropy: Mean (SD)'] = (
#         stats[entropy_col]['mean'].round(2).map('{:.2f}'.format) + " (" +
#         stats[entropy_col]['std'].round(2).map('{:.2f}'.format) + ")"
#     )

#     table_df['Mean Distance to Integer'] = stats['int_dist']['mean'].round(
#         2).map('{:.2f}'.format)

#     return table_df
from typing import Tuple
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def rating_entropy_distribution(
    df,
    model_col='model_name_clean',
    dim_col='dimension_name_clean',
    rating_col='mean_rating',
    entropy_col='normalized_entropy'
):
    """
    Computes model-level diagnostics by first calculating moments within each
    model x dimension cell, then averaging those moments across dimensions.

    This avoids inflating the rating SD by pooling together dimensions with
    different average rating levels.
    """

    df = df.copy()

    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    df[entropy_col] = pd.to_numeric(df[entropy_col], errors='coerce')

    # ------------------------------------------------------------
    # Step 1: Compute moments within each model x dimension cell
    # ------------------------------------------------------------
    dim_stats = (
        df
        .groupby([model_col, dim_col])
        .agg(
            rating_mean=(rating_col, 'mean'),
            rating_sd=(rating_col, 'std'),
            entropy_mean=(entropy_col, 'mean'),
            entropy_sd=(entropy_col, 'std')
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # Step 2: Average the dimension-level moments across dimensions
    # ------------------------------------------------------------
    model_stats = (
        dim_stats
        .groupby(model_col)
        .agg(
            rating_mean=('rating_mean', 'mean'),
            rating_sd=('rating_sd', 'mean'),
            entropy_mean=('entropy_mean', 'mean'),
            entropy_sd=('entropy_sd', 'mean')
        )
    )

    table_df = pd.DataFrame(index=model_stats.index)
    table_df.index.name = 'Model'

    table_df['Rating: Mean (SD)'] = (
        model_stats['rating_mean'].round(2).map('{:.2f}'.format) + " (" +
        model_stats['rating_sd'].round(2).map('{:.2f}'.format) + ")"
    )

    table_df['Entropy: Mean (SD)'] = (
        model_stats['entropy_mean'].round(2).map('{:.2f}'.format) + " (" +
        model_stats['entropy_sd'].round(2).map('{:.2f}'.format) + ")"
    )

    return table_df


# VERBOSITY BIAS


def verbosity_bias_table(
    df: pd.DataFrame,
    sections: Dict[str, str],
    models: Optional[List[str]] = None,
    model_col: str = "model_name_clean",
    dim_col: str = "dimension_name_clean",
    rating_col: str = "mean_rating",
    p_lower: Tuple[float, float] = (0.50, 0.75),
    p_upper: Tuple[float, float] = (0.75, 1.00),
    min_abs_d: Optional[float] = None,
    sort_by: str = "Absolute Cohen's d",
    ascending: bool = False,
    round_digits: int = 2
) -> pd.DataFrame:
    """
    Computes verbosity-bias diagnostics using Cohen's d between two word-count brackets.

    By default, compares Q3 against Q4:
        lower bracket: 50th-75th percentile
        upper bracket: 75th-100th percentile

    Parameters
    ----------
    df:
        Evaluation dataframe.

    sections:
        Dictionary mapping section labels to word-count columns.
        Example:
            {
                "Question": "question_word_count",
                "Answer": "answer_word_count"
            }

    models:
        Optional list of models to include. If None, uses all models.

    model_col:
        Column identifying model architecture.

    dim_col:
        Column identifying evaluation criterion/dimension.

    rating_col:
        Rating column, usually expected-value rating.

    p_lower:
        Quantile interval for the lower comparison bracket.

    p_upper:
        Quantile interval for the upper comparison bracket.

    min_abs_d:
        If provided, keeps only rows where abs(Cohen's d) >= min_abs_d.
        Useful for reporting only non-negligible verbosity effects.

    sort_by:
        Column used for sorting.

    ascending:
        Sort direction.

    round_digits:
        Number of decimals for reported statistics.

    Returns
    -------
    pd.DataFrame with one row per model x dimension x text section.
    """

    df = df.copy()

    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

    if models is None:
        models = list(df[model_col].dropna().unique())

    results = []

    for model in models:
        model_df = df[df[model_col] == model]

        for dim in model_df[dim_col].dropna().unique():
            dim_df = model_df[model_df[dim_col] == dim]

            if dim_df.empty:
                continue

            for section_name, text_col in sections.items():

                if text_col not in dim_df.columns:
                    raise KeyError(
                        f"Column '{text_col}' not found in dataframe.")

                tmp = dim_df[[rating_col, text_col]].dropna().copy()

                if tmp.empty:
                    continue

                tmp[text_col] = pd.to_numeric(tmp[text_col], errors="coerce")
                tmp = tmp.dropna(subset=[text_col, rating_col])

                if tmp.empty:
                    continue

                lower_min = tmp[text_col].quantile(p_lower[0])
                lower_max = tmp[text_col].quantile(p_lower[1])
                upper_min = tmp[text_col].quantile(p_upper[0])
                upper_max = tmp[text_col].quantile(p_upper[1])

                lower_ratings = tmp[
                    (tmp[text_col] >= lower_min) &
                    (tmp[text_col] <= lower_max)
                ][rating_col]

                upper_ratings = tmp[
                    (tmp[text_col] > upper_min) &
                    (tmp[text_col] <= upper_max)
                ][rating_col]

                n_lower = len(lower_ratings)
                n_upper = len(upper_ratings)

                if n_lower > 1 and n_upper > 1:
                    var_lower = lower_ratings.var(ddof=1)
                    var_upper = upper_ratings.var(ddof=1)

                    pooled_sd = np.sqrt(
                        ((n_lower - 1) * var_lower + (n_upper - 1) * var_upper)
                        / (n_lower + n_upper - 2)
                    )

                    raw_shift = upper_ratings.mean() - lower_ratings.mean()

                    if pooled_sd > 0:
                        cohens_d = raw_shift / pooled_sd
                        abs_d = abs(cohens_d)
                    else:
                        cohens_d = np.nan
                        abs_d = np.nan
                else:
                    raw_shift = np.nan
                    cohens_d = np.nan
                    abs_d = np.nan

                results.append({
                    "Model": model,
                    "Dimension": dim,
                    "Text Section": section_name,
                    "Mean Q3": lower_ratings.mean() if n_lower > 0 else np.nan,
                    "Mean Q4": upper_ratings.mean() if n_upper > 0 else np.nan,
                    "Raw Shift": raw_shift,
                    "Cohen's d": cohens_d,
                    "Absolute Cohen's d": abs_d,
                    "N Q3": n_lower,
                    "N Q4": n_upper,
                })

    out = pd.DataFrame(results)

    if out.empty:
        return out

    out = out.dropna(subset=["Cohen's d", "Absolute Cohen's d"])

    if min_abs_d is not None:
        out = out[out["Absolute Cohen's d"] >= min_abs_d]

    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)

    numeric_cols = [
        "Mean Q3",
        "Mean Q4",
        "Raw Shift",
        "Cohen's d",
        "Absolute Cohen's d"
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].round(round_digits)

    return out.reset_index(drop=True)


def summarize_verbosity_bias(
    bias_df: pd.DataFrame,
    model_col: str = "Model",
    abs_d_col: str = "Absolute Cohen's d",
    small_threshold: float = 0.20,
    medium_threshold: float = 0.50,
    large_threshold: float = 0.80,
    round_digits: int = 2,
    combine_medium_large: bool = False
) -> pd.DataFrame:
    """
    Summarizes verbosity-bias diagnostics by model using mutually exclusive
    Cohen's d bins.

    Bins:
        Negligible: |d| < small_threshold
        Small: small_threshold <= |d| < medium_threshold
        Medium: medium_threshold <= |d| < large_threshold
        Large: |d| >= large_threshold
    """

    df = bias_df.copy()
    df[abs_d_col] = pd.to_numeric(df[abs_d_col], errors="coerce")
    df = df.dropna(subset=[model_col, abs_d_col])

    def summarize_group(x):
        return pd.Series({
            "Mean $|d|$": x.mean(),
            "Max $|d|$": x.max(),
            "N": x.count(),
            "Negligible": ((x < small_threshold)).sum(),
            "Small": ((x >= small_threshold) & (x < medium_threshold)).sum(),
            "Medium": ((x >= medium_threshold) & (x < large_threshold)).sum(),
            "Large": ((x >= large_threshold)).sum(),
        })

    summary = (
        df.groupby(model_col)[abs_d_col]
        .apply(summarize_group)
        .unstack()
        .reset_index()
        .rename(columns={model_col: "Model"})
    )

    summary["N"] = summary["N"].astype(int)
    for col in ["Negligible", "Small", "Medium", "Large"]:
        summary[col] = summary[col].astype(int)

    summary["Mean $|d|$"] = summary["Mean $|d|$"].map(
        lambda x: f"{x:.{round_digits}f}"
    )
    summary["Max $|d|$"] = summary["Max $|d|$"].map(
        lambda x: f"{x:.{round_digits}f}"
    )

    if combine_medium_large:
        summary["Medium/Large"] = summary["Medium"] + summary["Large"]
        summary = summary[
            ["Model", "Mean $|d|$", "Max $|d|$", "N",
                "Negligible", "Small", "Medium/Large"]
        ]
    else:
        summary = summary[
            ["Model", "Mean $|d|$", "Max $|d|$", "N",
                "Negligible", "Small", "Medium", "Large"]
        ]

    return summary
