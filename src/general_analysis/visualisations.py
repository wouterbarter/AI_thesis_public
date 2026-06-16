from typing import List, Optional, Tuple
from typing import List, Optional
from typing import Optional, List, Dict
from typing import Optional, Tuple, List, Dict
from typing import Optional, List
import math
from typing import Optional, Tuple
from typing import Optional, Union
import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from typing import Optional
from pandas.io.formats.style import Styler


def plot_verbosity_bias(final_df: pd.DataFrame, score_column: str = 'mean_rating', input_length_column: str = 'raw_input_length', save_path: Optional[str] = None) -> plt.Axes:
    """
    Calculates and plots the Spearman correlation between input length and the LLM's 
    assigned score to analyze verbosity bias across models and prompt architectures.

    Args:
        final_df (pd.DataFrame): The main dataframe containing the evaluations.
        score_column (str): The column representing the LLM's continuous score (e.g., 'expected_value').
        save_path (str, optional): If provided, saves the plot to this file path.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """

    # 1. Ensure prompt_name column exists
    if 'prompt_name' not in final_df.columns:
        print("prompt_name column not specified")
        return

    # 2. Calculate the Spearman correlation for each model and prompt architecture
    results = []

    # We assume final_df already has 'model_name' and 'prompt_name' properly formatted
    # based on your previous analyses.
    for (model, prompt), group in final_df.groupby(['model_name', 'prompt_name']):
        # Drop missing values to prevent scipy from throwing errors
        valid_data = group[[input_length_column, score_column]].dropna()

        if len(valid_data) > 1:
            corr, p_val = spearmanr(
                valid_data[input_length_column], valid_data[score_column])
            results.append({
                'model_name': model,
                'prompt_name': prompt,
                'verbosity_corr': corr
            })

    df_bias = pd.DataFrame(results)

    # 3. Plotting Setup
    # Enforce the specific ordering (reverse alphabetical)
    prompt_order = ['Holistic Naive', 'Holistic Informed', 'Formative']

    plt.figure(figsize=(10, 6))

    # Generate the grouped bar plot
    ax = sns.barplot(
        data=df_bias,
        x='model_name',
        y='verbosity_corr',
        hue='prompt_name',
        hue_order=prompt_order,
        palette='Set2'  # Keeping the same color palette as the EV regression plot
    )

    # Formatting and aesthetics
    ax.set_title('Verbosity Bias: Correlation between Input Length and LLM Score',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Name', fontsize=12)
    ax.set_ylabel('Spearman Correlation (ρ)', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=15)

    # Move legend outside the main plot area
    plt.legend(title='Prompt Architecture',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add horizontal grid lines behind the bars
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Handle output routing
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return ax


def plot_verbosity_breakdown(
    corr_df: pd.DataFrame,
    dim_map_clean: Optional[Dict[str, str]] = None,
    plot_order: Optional[List[str]] = None,
    group_by: str = 'model',  # Toggle between 'model' and 'dimension'
    save_path: Optional[str] = None
) -> plt.Axes:
    """
    Takes a computed correlation DataFrame and generates a grouped bar chart.
    Can group the bars either by Model (default) or by Dimension.
    """
    if corr_df.empty:
        print("The provided correlation DataFrame is empty. Nothing to plot.")
        return None

    plot_df = corr_df.copy()

    # 1. Map raw dimension names to clean names
    if dim_map_clean:
        plot_df['dimension_name'] = plot_df['dimension_name'].map(
            dim_map_clean).fillna(plot_df['dimension_name'])

    # 2. Filter available dimensions to maintain strict order
    available_dims = None
    if plot_order:
        plot_df = plot_df[plot_df['dimension_name'].isin(plot_order)]
        available_dims = [
            dim for dim in plot_order if dim in plot_df['dimension_name'].unique()]

    if plot_df.empty:
        print("WARNING: DataFrame is empty after filtering by plot_order.")
        return None

    # 3. Dynamic grouping logic
    if group_by == 'model':
        x_var = 'model_name'
        hue_var = 'dimension_name'
        x_order = None  # Let Seaborn handle model order, or pass a specific list if you have one
        hue_order = available_dims  # Enforce dimension order in the legend
        legend_title = 'Dimension'
        rot = 15
        ha = 'center'
    elif group_by == 'dimension':
        x_var = 'dimension_name'
        hue_var = 'model_name'
        x_order = available_dims  # Enforce dimension order on the X-axis
        # Alphabetical model order in the legend
        hue_order = sorted(plot_df['model_name'].unique())
        legend_title = 'Model Name'
        rot = 45  # Steeper rotation since dimension names are longer
        ha = 'right'
    else:
        raise ValueError("group_by must be either 'model' or 'dimension'")

    # 4. Plotting Setup
    # Make it slightly wider if grouping by dimension to fit the text
    fig_width = 16 if group_by == 'dimension' else 14
    plt.figure(figsize=(fig_width, 6))

    ax = sns.barplot(
        data=plot_df,
        x=x_var,
        y='verbosity_corr',
        hue=hue_var,
        order=x_order,
        hue_order=hue_order,
        palette='Set3'
    )

    # 5. Formatting and aesthetics
    title_suffix = "by Model" if group_by == 'model' else "by Dimension"
    ax.set_title(f'Verbosity Correlation Breakdown {title_suffix}',
                 fontsize=14, fontweight='bold')

    ax.set_xlabel(legend_title, fontsize=12)
    ax.set_ylabel('Spearman Correlation (ρ) with Length', fontsize=12)

    # Apply dynamic X-axis label rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rot, ha=ha)

    # Move legend completely outside the plot area
    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc='upper left')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return ax


def plot_formative_verbosity_breakdown(final_df: pd.DataFrame, score_column: str = 'mean_rating', input_length_column: str = 'raw_input_length', breakdown_col: str = 'Formative', save_path: Optional[str] = None) -> plt.Axes:
    """
    Unpacks the 'Formative' architecture to show verbosity correlation 
    across its individual subdimensions.
    """
    # 1. Ensure prompt_name column exists
    if 'prompt_name' not in final_df.columns:
        print("prompt_name column not specified")
        return

    # 2. Filter ONLY for Formative evaluations
    formative_df = final_df[final_df['prompt_name'] == breakdown_col]

    # 3. Calculate correlation per model AND per subdimension
    results = []
    # Assuming your subdimensions are stored in a column like 'dimension_name'
    for (model, dimension), group in formative_df.groupby(['model_name', 'dimension_name']):
        valid_data = group[[input_length_column, score_column]].dropna()

        if len(valid_data) > 1:
            corr, p_val = spearmanr(
                valid_data[input_length_column], valid_data[score_column])
            results.append({
                'model_name': model,
                'dimension_name': dimension,
                'verbosity_corr': corr
            })

    df_breakdown = pd.DataFrame(results)

    # 4. Plotting Setup
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=df_breakdown,
        x='model_name',
        y='verbosity_corr',
        hue='dimension_name',
        palette='Set3'  # A lighter palette to differentiate it from the main architecture plots
    )

    # Formatting and aesthetics
    ax.set_title('Verbosity Correlation Breakdown: Formative Subdimensions',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Name', fontsize=12)
    ax.set_ylabel('Spearman Correlation (ρ) with Length', fontsize=12)

    plt.xticks(rotation=15)
    plt.legend(title='Formative Subdimension',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return ax


def plot_absolute_bias_spread(df,
                              sections,
                              models=['Gemma 4', 'Llama 3.2',
                                      'Qwen 3', 'Qwen 3.5'],
                              dim_col='dimension_name_clean',
                              rating_col='mean_rating',
                              # Expects a list/tuple of [min_quantile, max_quantile]
                              p_lower=[0.0, 0.25],
                              # Expects a list/tuple of [min_quantile, max_quantile]
                              p_upper=[0.75, 1.0]):

    results = []
    dimensions = df[dim_col].dropna().unique()

    for m in models:
        for dim in dimensions:
            sub = df[(df['model_name_clean'] == m) & (df[dim_col] == dim)]
            if sub.empty:
                continue

            for section_name, text_col in sections.items():

                # 1. Calculate the word count thresholds based on the requested quantile ranges
                lower_bound_min = sub[text_col].quantile(p_lower[0])
                lower_bound_max = sub[text_col].quantile(p_lower[1])

                upper_bound_min = sub[text_col].quantile(p_upper[0])
                upper_bound_max = sub[text_col].quantile(p_upper[1])

                # 2. Subset the data strictly within these specific brackets
                short_texts = sub[(sub[text_col] >= lower_bound_min) & (
                    sub[text_col] <= lower_bound_max)][rating_col]
                long_texts = sub[(sub[text_col] >= upper_bound_min) & (
                    sub[text_col] <= upper_bound_max)][rating_col]

                n1, n2 = len(long_texts), len(short_texts)

                # 3. Calculate Absolute Cohen's d
                if n1 > 1 and n2 > 1:
                    var1, var2 = long_texts.var(), short_texts.var()
                    pooled_sd = np.sqrt(
                        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                    raw_shift = long_texts.mean() - short_texts.mean()

                    absolute_cohens_d = np.abs(
                        raw_shift / pooled_sd) if pooled_sd > 0 else np.nan

                    cohens_d = (
                        raw_shift / pooled_sd) if pooled_sd > 0 else np.nan
                else:
                    cohens_d = np.nan

                results.append({
                    'Model': m,
                    'Dimension': dim,
                    'Text Section': section_name,
                    'Absolute Cohen\'s d': absolute_cohens_d,
                    'Cohen\'s d': cohens_d
                })

    res_df = pd.DataFrame(results).dropna()

    # --- Plotting Setup ---
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set3", n_colors=len(models))

    # The Boxplot
    ax = sns.boxplot(data=res_df, x='Text Section', y='Absolute Cohen\'s d', hue='Model',
                     palette=palette, showfliers=False, width=0.6, boxprops={'alpha': 0.7})

    # The Strip Plot
    sns.stripplot(data=res_df, x='Text Section', y='Absolute Cohen\'s d', hue='Model',
                  palette=palette, dodge=True, linewidth=1, edgecolor='gray', alpha=0.8, size=5, ax=ax)

    # Threshold Lines
    ax.axhline(0.8, color='red', linestyle='--',
               linewidth=1.5, alpha=0.7, zorder=0)
    ax.text(-0.45, 0.82, 'Large Effect (|d| = 0.8)',
            color='red', fontsize=10, style='italic')

    ax.axhline(0.5, color='orange', linestyle='--',
               linewidth=1.5, alpha=0.7, zorder=0)
    ax.text(-0.45, 0.52, 'Moderate Effect (|d| = 0.5)',
            color='orange', fontsize=10, style='italic')

    # Dynamic Title showing the exact comparison brackets
    bracket_str = f"Quantiles [{p_lower[0]}-{p_lower[1]}] vs. [{p_upper[0]}-{p_upper[1]}]"
    ax.set_title(f"Distribution of Absolute Bias Magnitude (|d|)\nComparison Bracket: {bracket_str}",
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(r"Absolute Cohen's $d$ ($|d|$)",
                  fontsize=13, fontweight='bold')
    ax.set_xlabel("Text Section Evaluated", fontsize=13, fontweight='bold')

    # Fix legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(models)], labels[:len(models)],
              title='Model', loc='upper right')

    plt.tight_layout()
    plt.show()

    return res_df


def plot_verbosity_grid(df: pd.DataFrame,
                        models: List[str],
                        dimensions: Optional[List[str]] = None,
                        model_col: str = 'model_name_clean',
                        dim_col: str = 'dimension_name_clean',
                        x_col: str = 'raw_input_word_count',
                        y_col: str = 'mean_rating',
                        log_x: bool = True,
                        lower_truncate_pct: float = 0.02,
                        upper_truncate_pct: float = 0.98,
                        main_title: str = None,
                        x_label: str = "Total Word Count",
                        y_label: str = "LLM Rating",
                        y_min: float = 0.5,
                        y_max: float = 7.5,
                        save_path: str = None):
    """
    Generates a matrix facet grid comparing models (rows) across dimensions (columns).
    Stripped of correlation metrics to emphasize non-linear LOESS trajectories.
    Dynamically scales to any rating system (e.g., 1-4, 1-7, 1-100).
    """

    if dimensions is None:
        subset = df[df[model_col].isin(models)]
        dimensions = sorted(subset[dim_col].dropna().unique())

    n_models = len(models)
    n_dims = len(dimensions)

    if n_dims == 0 or n_models == 0:
        print("⚠️ Not enough data to plot. Check your model names and dimension lists.")
        return

    fig, axes = plt.subplots(nrows=n_models, ncols=n_dims,
                             figsize=(4 * n_dims, 4 * n_models),
                             sharey=True, sharex=True, squeeze=False)

    for i, model in enumerate(models):
        for j, dim in enumerate(dimensions):
            ax = axes[i, j]

            dim_data = df[(df[model_col] == model) & (
                df[dim_col] == dim) & (df[x_col] > 0)].copy()

            if dim_data.empty:
                ax.set_visible(False)
                continue

            lower_cutoff = dim_data[x_col].quantile(lower_truncate_pct)
            upper_cutoff = dim_data[x_col].quantile(upper_truncate_pct)

            plot_df = dim_data[(dim_data[x_col] >= lower_cutoff) &
                               (dim_data[x_col] <= upper_cutoff)].copy()

            # cutoff = dim_data[x_col].quantile(truncate_pct)
            # plot_df = dim_data[dim_data[x_col] <= cutoff].copy()

            # Plot Scatter & LOESS (Cleaned of text boxes)
            sns.regplot(data=plot_df, x=x_col, y=y_col,
                        scatter_kws={'alpha': 0.2,
                                     's': 15, 'color': '#34495e'},
                        line_kws={'color': '#e74c3c', 'lw': 3},
                        lowess=True, ax=ax)

            if log_x:
                ax.set_xscale('log')

            ax.set_xlabel("")
            ax.set_ylabel("")

            # Apply dynamic Y-axis limits
            ax.set_ylim(y_min, y_max)

            # Top Row: Show Dimension Names
            if i == 0:
                clean_dim = str(dim).replace('\n', ' ')
                ax.set_title(clean_dim, fontweight='bold', fontsize=13, pad=10)

            # Bottom Row: Show X-Axis Labels
            if i == n_models - 1:
                final_x_label = f"{x_label} (Log Scale)" if log_x else x_label
                ax.set_xlabel(final_x_label, fontweight='bold', fontsize=11)

            # Left Column: Show Vertical Model Names & Y-Axis
            if j == 0:
                ax.annotate(model, xy=(-0.35, 0.5), xycoords='axes fraction',
                            rotation=90, va='center', ha='center',
                            fontweight='bold', fontsize=16)

                # Apply dynamic Y-axis label
                ax.set_ylabel(y_label, fontsize=11)

    final_main_title = main_title if main_title else "Information Satiation by Model"
    plt.suptitle(final_main_title, fontsize=22, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.subplots_adjust(left=0.08)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Grid plot saved to {save_path}")
    else:
        plt.show()


def plot_verbosity_comparison_twocols(df: pd.DataFrame,
                                      models: list,
                                      model_col: str = 'model_name_clean',
                                      dim_col: str = 'dimension_name_clean',
                                      left_x_col: str = 'requirements_word_count',
                                      right_x_col: str = 'text_word_count',
                                      y_col: str = 'mean_rating',
                                      log_x: bool = False,
                                      truncate_pct: float = 0.98,
                                      main_title: str = None,
                                      left_x_label: str = "Requirements Words",
                                      right_x_label: str = "Body Words",
                                      save_path: str = None):
    """
    Generates a 4-column facet grid to directly compare two models side-by-side.
    Layout per row: [M1 Req] [M1 Body] | [M2 Req] [M2 Body]
    """
    if len(models) != 2:
        raise ValueError(
            "Please provide exactly two model names in the 'models' list.")

    m1, m2 = models

    # 1. Filter for the two models
    subset = df[df[model_col].isin(models)].copy()

    # 2. Get dimensions present in BOTH models to ensure a fair comparison
    dims_m1 = set(subset[subset[model_col] == m1][dim_col].dropna().unique())
    dims_m2 = set(subset[subset[model_col] == m2][dim_col].dropna().unique())
    dimensions = sorted(list(dims_m1.intersection(dims_m2)))
    n_dims = len(dimensions)

    if n_dims == 0:
        print("⚠️ No common dimensions found between the two models.")
        return

    # 3. Create Grid: n_dims rows, 4 columns. Make it extra wide (22 inches)
    fig, axes = plt.subplots(nrows=n_dims, ncols=4,
                             figsize=(22, 5 * n_dims), sharey='row')

    if n_dims == 1:
        axes = [axes]

    # 4. Loop through each dimension and plot
    for i, dim in enumerate(dimensions):
        clean_dim_name = str(dim).replace('\n', ' - ')

        # Loop through the 2 models for this dimension
        for m_idx, m_name in enumerate(models):
            dim_data = subset[(subset[dim_col] == dim) & (
                subset[model_col] == m_name)].copy()

            # col_offset is 0 for Model 1, and 2 for Model 2
            col_offset = m_idx * 2
            ax_left = axes[i][col_offset]
            ax_right = axes[i][col_offset + 1]

            # Truncate outliers
            left_cutoff = dim_data[left_x_col].quantile(truncate_pct)
            right_cutoff = dim_data[right_x_col].quantile(truncate_pct)

            plot_df = dim_data[(dim_data[left_x_col] <= left_cutoff) &
                               (dim_data[right_x_col] <= right_cutoff)].copy()

            # ==========================================
            # Plot 1: Requirements (Red)
            # ==========================================
            sns.regplot(data=plot_df, x=left_x_col, y=y_col,
                        scatter_kws={'alpha': 0.2, 's': 15}, line_kws={'color': 'red', 'lw': 3},
                        lowess=True, ax=ax_left)

            r_left, p_left = spearmanr(
                plot_df[left_x_col], plot_df[y_col], nan_policy='omit')

            if log_x:
                ax_left.set_xscale('symlog')

            ax_left.set_title(
                f"[{m_name}]\n{clean_dim_name} - Req.", fontweight='bold', fontsize=12)
            final_left_x = f"{left_x_label} (Log)" if log_x else left_x_label
            ax_left.set_xlabel(final_left_x)

            # Only show Y-axis label on the far left plot of the row
            if col_offset == 0:
                ax_left.set_ylabel("LLM Rating (1-7)", fontweight='bold')
            else:
                ax_left.set_ylabel("")

            ax_left.text(0.05, 0.95, f"r: {r_left:.3f}\np: {p_left:.3f}",
                         transform=ax_left.transAxes, fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # ==========================================
            # Plot 2: Body (Blue)
            # ==========================================
            sns.regplot(data=plot_df, x=right_x_col, y=y_col,
                        scatter_kws={'alpha': 0.2, 's': 15}, line_kws={'color': 'blue', 'lw': 3},
                        lowess=True, ax=ax_right)

            r_right, p_right = spearmanr(
                plot_df[right_x_col], plot_df[y_col], nan_policy='omit')

            if log_x:
                ax_right.set_xscale('symlog')

            ax_right.set_title(
                f"[{m_name}]\n{clean_dim_name} - Body", fontweight='bold', fontsize=12)
            final_right_x = f"{right_x_label} (Log)" if log_x else right_x_label
            ax_right.set_xlabel(final_right_x)
            ax_right.set_ylabel("")

            ax_right.text(0.05, 0.95, f"r: {r_right:.3f}\np: {p_right:.3f}",
                          transform=ax_right.transAxes, fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5. Global Aesthetics
    final_main_title = main_title if main_title else f"Verbosity Diagnostics: {models[0]} vs. {models[1]}"
    plt.suptitle(final_main_title, fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 6. Save or Show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Comparison plot saved to {save_path}")
    else:
        plt.show()


def plot_verbosity_ceilings(df: pd.DataFrame,
                            model_name: str,
                            model_col: str = 'model_name_clean',
                            dim_col: str = 'dimension_name_clean',
                            left_x_col: str = 'requirements_word_count',
                            right_x_col: str = 'text_word_count',
                            log_x: bool = False,
                            truncate_pct: float = 0.98,  # Drops the extreme tail for clean LOESS
                            main_title: str = None,
                            left_title_suffix: str = "Requirements Length vs. LLM Rating",
                            right_title_suffix: str = "Body Length vs. LLM Rating",
                            left_x_label: str = "Requirements Word Count",
                            right_x_label: str = "Body Word Count",
                            save_path: str = None):
    """
    Generates a facet grid of LOESS curves to diagnose Verbosity Bias ceilings.
    Dynamically maps x-axis columns to support multiple datasets (e.g., Barter, FeedbackQA).
    """
    # 1. Filter for the specific model
    subset = df[df[model_col] == model_name].copy()

    # 2. Get all unique dimensions for this model and sort them
    dimensions = sorted(subset[dim_col].dropna().unique())
    n_dims = len(dimensions)

    if n_dims == 0:
        print(
            f"⚠️ No dimensions found for model: {model_name} in column: {model_col}")
        return

    # 3. Create the Grid: n_dims rows, 2 columns
    fig, axes = plt.subplots(nrows=n_dims, ncols=2,
                             figsize=(14, 5 * n_dims), sharey=True)

    if n_dims == 1:
        axes = [axes]

    # 4. Loop through each dimension and plot
    for i, dim in enumerate(dimensions):
        dim_data = subset[subset[dim_col] == dim].copy()

        # Calculate truncation thresholds to prevent sparse-tail LOESS hallucinations
        left_cutoff = dim_data[left_x_col].quantile(truncate_pct)
        right_cutoff = dim_data[right_x_col].quantile(truncate_pct)

        plot_df = dim_data[(dim_data[left_x_col] <= left_cutoff) &
                           (dim_data[right_x_col] <= right_cutoff)].copy()

        ax_left = axes[i][0]
        ax_right = axes[i][1]

        # ==========================================
        # Plot 1: Left Graph
        # ==========================================
        sns.regplot(data=plot_df, x=left_x_col, y='mean_rating',
                    scatter_kws={'alpha': 0.2, 's': 15}, line_kws={'color': 'red', 'lw': 3},
                    lowess=True, ax=ax_left)

        # Correlation
        r_left, p_left = spearmanr(
            plot_df[left_x_col], plot_df['mean_rating'], nan_policy='omit')

        if log_x:
            # symlog safely handles zeros if they exist
            ax_left.set_xscale('symlog')

        clean_dim_name = str(dim).replace('\n', ' - ')
        ax_left.set_title(
            f"{clean_dim_name}\n{left_title_suffix}", fontweight='bold', fontsize=12)

        final_left_x_label = f"{left_x_label} (Log Scale)" if log_x else left_x_label
        ax_left.set_xlabel(final_left_x_label)
        ax_left.set_ylabel("LLM Rating (1-7)", fontweight='bold')

        # Add correlation box
        ax_left.text(0.05, 0.95, f"Spearman r: {r_left:.3f}\np-val: {p_left:.3f}",
                     transform=ax_left.transAxes, fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ==========================================
        # Plot 2: Right Graph
        # ==========================================
        sns.regplot(data=plot_df, x=right_x_col, y='mean_rating',
                    scatter_kws={'alpha': 0.2, 's': 15}, line_kws={'color': 'blue', 'lw': 3},
                    lowess=True, ax=ax_right)

        # Correlation
        r_right, p_right = spearmanr(
            plot_df[right_x_col], plot_df['mean_rating'], nan_policy='omit')

        if log_x:
            ax_right.set_xscale('symlog')

        ax_right.set_title(
            f"{clean_dim_name}\n{right_title_suffix}", fontweight='bold', fontsize=12)

        final_right_x_label = f"{right_x_label} (Log Scale)" if log_x else right_x_label
        ax_right.set_xlabel(final_right_x_label)
        ax_right.set_ylabel("")

        # Add correlation box
        ax_right.text(0.05, 0.95, f"Spearman r: {r_right:.3f}\np-val: {p_right:.3f}",
                      transform=ax_right.transAxes, fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5. Global Aesthetics
    final_main_title = main_title if main_title else f"Verbosity Ceiling Diagnostics: {model_name}"
    plt.suptitle(final_main_title, fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()

    # 6. Save or Show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()


def compute_multimodality(logits_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the number of local maxima for a 2D array of probabilities or logits.

    Parameters:
    logits_matrix (np.ndarray): A 2D array of shape (N, C) where C is the number of ordinal classes.

    Returns:
    np.ndarray: A 1D array of length N containing the integer count of peaks per row.
    """
    # 1. Pad the left and right edges with -infinity
    padded = np.pad(
        logits_matrix,
        pad_width=((0, 0), (1, 1)),
        mode='constant',
        constant_values=-np.inf
    )

    # 2. A point is a local maximum if it is strictly greater than BOTH neighbors
    is_greater_than_left = padded[:, 1:-1] > padded[:, :-2]
    is_greater_than_right = padded[:, 1:-1] > padded[:, 2:]

    # 3. Find where both conditions are True
    is_peak = is_greater_than_left & is_greater_than_right

    # 4. Count the peaks per row
    return is_peak.sum(axis=1)


def append_multimodality_flags(df: pd.DataFrame, logits_col: str = 'sorted_logits') -> pd.DataFrame:
    """
    Extracts the logits matrix from a dataframe, computes multimodality, 
    and appends 'peak_count' and 'is_multimodal' columns to the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    logits_col (str): The name of the column containing the logits/probabilities.

    Returns:
    pd.DataFrame: The dataframe with the new multimodality columns added.
    """
    # 1. Flatten the entire matrix at once safely
    logits_matrix = np.array(df[logits_col].tolist())

    if logits_matrix.ndim == 3:
        logits_matrix = logits_matrix.squeeze(axis=1)

    # 2. Compute the peaks and create binary flag
    df['peak_count'] = compute_multimodality(logits_matrix)
    df['is_multimodal'] = df['peak_count'] > 1

    return df


def generate_multimodality_report(df: pd.DataFrame, group_col: str = 'analysis_group') -> Styler:
    """
    Generates a stylized statistical summary of multimodality failure rates.
    Assumes 'append_multimodality_flags' has already been run on the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing 'is_multimodal' and the grouping column.
    group_col (str): The column name to group the statistics by.

    Returns:
    Styler: A styled pandas DataFrame ready for Jupyter Notebook rendering.
    """
    if 'is_multimodal' not in df.columns:
        raise ValueError(
            "Column 'is_multimodal' not found. Please run append_multimodality_flags() first.")

    # 1. Group and Aggregate
    multimodal_stats = (
        df.groupby(group_col)['is_multimodal']
        .agg(
            Failure_Rate='mean',
            Std_Dev='std',
            Total_Evaluations='count'
        )
        .sort_values('Failure_Rate', ascending=False)
    )

    # 2. Style the output
    styled_stats = multimodal_stats.style.format({
        'Failure_Rate': '{:.2%}'.format,
        'Std_Dev': '{:.4f}'.format,
        'Total_Evaluations': '{:,}'.format
    }).background_gradient(subset=['Failure_Rate'], cmap='Reds')

    return styled_stats


def plot_multimodality_rates(df_stats: pd.DataFrame, prompt_order, group_col: str = 'analysis_group_named', save_path: Optional[str] = None) -> plt.Axes:
    """
    Plots the multimodality failure rate across models and prompt architectures.

    Args:
        df_stats (pd.DataFrame): The unstyled dataframe returned by the groupby aggregation 
                                 (must contain the 'Failure_Rate' column as a float between 0 and 1).
        group_col (str): The column or index name containing the group labels.
        save_path (str, optional): If provided, saves the plot to this file path.
    """
    # 1. Handle whether the group_col is in the index or a standard column
    if group_col in df_stats.index.names:
        df_plot = df_stats.reset_index()
    else:
        df_plot = df_stats.copy()

    # 2. Helper to parse the label (Format: Model_Prompt_Hash)
    def parse_group(label):
        parts = str(label).split('_')
        model = parts[0]
        prompt = parts[1] if len(parts) > 1 else 'Unknown'
        return pd.Series([model, prompt])

    df_plot[['model_name', 'prompt_name']
            ] = df_plot[group_col].apply(parse_group)

    # 3. Plotting Setup
    # prompt_order = ['Holistic Naive', 'Holistic Informed', 'Formative']

    plt.figure(figsize=(10, 6))

    # 4. Generate the grouped bar plot
    ax = sns.barplot(
        data=df_plot,
        x='model_name',
        y='Failure_Rate',
        hue='prompt_name',
        hue_order=prompt_order,
        # Using a distinct palette (reds/blues) to differentiate from the EV/Verbosity charts
        palette='Set1'
    )

    # 5. Formatting and aesthetics
    ax.set_title('Multimodality Rate by Model and Prompt Architecture',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Name', fontsize=12)
    ax.set_ylabel('Multimodality Rate (% of Evaluations)', fontsize=12)

    # Format Y-axis correctly as percentages (assuming Failure_Rate is a float 0.0 - 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.xticks(rotation=15)
    plt.legend(title='Prompt Architecture',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return ax


def plot_formative_multimodality_breakdown(final_df: pd.DataFrame, breakdown_col: str = 'Formative', save_path: Optional[str] = None) -> plt.Axes:
    """
    Unpacks the 'Formative' architecture to show the multimodality failure rate 
    across its individual subdimensions.

    Args:
        final_df (pd.DataFrame): The main dataframe containing evaluations. 
                                 Must contain 'is_multimodal', 'model_name', 
                                 'prompt_name', and 'dimension_name'.
        save_path (str, optional): If provided, saves the plot to this file path.
    """
    # 1. Ensure the multimodality flag exists
    if 'is_multimodal' not in final_df.columns:
        raise ValueError(
            "Column 'is_multimodal' not found. Please run append_multimodality_flags() first.")

    # 2. Filter ONLY for Formative evaluations
    formative_df = final_df[final_df['prompt_name'] == breakdown_col].copy()

    # 3. Calculate failure rate per model AND per subdimension
    df_breakdown = (
        formative_df.groupby(['model_name', 'dimension_name'])['is_multimodal']
        .mean()
        .reset_index()
        .rename(columns={'is_multimodal': 'Failure_Rate'})
    )

    # 4. Plotting Setup
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=df_breakdown,
        x='model_name',
        y='Failure_Rate',
        hue='dimension_name',
        palette='Set3'  # Lighter palette to distinguish from high-level architecture plots
    )

    # 5. Formatting and aesthetics
    ax.set_title(f'Multimodality Rate Breakdown: {breakdown_col} Subdimensions',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Name', fontsize=12)
    ax.set_ylabel('Multimodality Rate (% of Evaluations)', fontsize=12)

    # Format Y-axis as percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.xticks(rotation=15)
    plt.legend(title='Formative Subdimension',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return ax


def compute_multimodal_severity(probs_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the raw Multimodal Conflict Severity (MCS) for a matrix of probabilities.
    """
    # 1. Identify all peaks (same padding logic as before)
    padded = np.pad(probs_matrix, pad_width=((0, 0), (1, 1)),
                    mode='constant', constant_values=-np.inf)
    is_peak = (padded[:, 1:-1] > padded[:, :-2]
               ) & (padded[:, 1:-1] > padded[:, 2:])

    # 2. Find the primary peak (global maximum) for each row
    primary_peak_idx = np.argmax(probs_matrix, axis=1)

    # 3. Create a distance matrix: distance from every token to the primary peak
    num_tokens = probs_matrix.shape[1]
    token_indices = np.arange(num_tokens)
    distances = np.abs(token_indices[None, :] - primary_peak_idx[:, None])

    # 4. Isolate secondary peaks (it is a peak, AND it is not the primary peak)
    is_secondary_peak = is_peak & (
        token_indices[None, :] != primary_peak_idx[:, None])

    # 5. Calculate severity: Sum of (probability * distance) for all secondary peaks
    severity = np.sum(probs_matrix * distances * is_secondary_peak, axis=1)

    return severity


def append_multimodality_severity(df: pd.DataFrame, logits_col: str = 'sorted_logits') -> pd.DataFrame:
    """
    Converts logits to probabilities, calculates the Normalized MCS, 
    and appends it to the dataframe.
    """
    # 1. Extract logits safely
    logits_matrix = np.array(df[logits_col].tolist())
    if logits_matrix.ndim == 3:
        logits_matrix = logits_matrix.squeeze(axis=1)

    # 2. Apply Softmax to ensure we are working with true probabilities
    exp_logits = np.exp(
        logits_matrix - np.max(logits_matrix, axis=1, keepdims=True))
    probs_matrix = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 3. Calculate Raw Severity
    raw_severity = compute_multimodal_severity(probs_matrix)

    # 4. Normalize (Max severity = 50% mass * max possible distance)
    max_distance = probs_matrix.shape[1] - 1
    max_possible_severity = 0.5 * max_distance

    df['multimodal_severity_score'] = raw_severity / max_possible_severity

    # Keep the boolean flag for backwards compatibility with your older plots,
    # but base it on a reasonable noise threshold (e.g., > 1% severity)
    df['is_multimodal'] = df['multimodal_severity_score'] > 0.01

    return df


def generate_multimodal_severity_report(df: pd.DataFrame, group_col: str = 'analysis_group_named') -> Styler:
    """
    Generates a stylized statistical summary using the continuous severity metric.
    """
    if 'multimodal_severity_score' not in df.columns:
        raise ValueError(
            "Column not found. Run append_multimodality_severity() first.")

    # Group and aggregate the continuous score
    severity_stats = (
        df.groupby(group_col)['multimodal_severity_score']
        .agg(
            Mean_Severity='mean',
            Max_Severity='max',  # Helpful to see if a group ever hits 100% split
            Total_Evaluations='count'
        )
        .sort_values('Mean_Severity', ascending=False)
    )

    # Style the output
    styled_stats = severity_stats.style.format({
        'Mean_Severity': '{:.2%}'.format,
        'Max_Severity': '{:.2%}'.format,
        'Total_Evaluations': '{:,}'.format
    }).background_gradient(subset=['Mean_Severity'], cmap='Purples')

    return styled_stats


def plot_severity_bar_chart(final_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Axes:
    """
    Plots the continuous Multimodal Conflict Severity (MCS) across models and architectures.
    Replaces the old binary failure rate plot.
    """
    if 'multimodal_severity_score' not in final_df.columns:
        raise ValueError(
            "Column not found. Run append_multimodality_severity() first.")

    prompt_order = ['Holistic Naive', 'Holistic Informed', 'Formative']

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=final_df,
        x='model_name',
        y='multimodal_severity_score',
        hue='prompt_name',
        hue_order=prompt_order,
        palette='Purples'  # Switched to purples to match your styled dataframe!
    )

    ax.set_title('Average Multimodal Conflict Severity (MCS) by Architecture',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Name', fontsize=12)
    ax.set_ylabel('Mean Severity Score (%)', fontsize=12)

    # Format Y-axis correctly as percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.xticks(rotation=15)
    plt.legend(title='Prompt Architecture',
               bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return ax


def plot_epistemic_valley(
    final_df: pd.DataFrame,
    target_model: str,
    target_prompt: str = 'Formative',
    breakdown_by_dimension: bool = False,
    save_path: Optional[str] = None
) -> Union[plt.Axes, sns.FacetGrid]:
    """
    Plots a 2D density landscape (KDE) of Expected Value vs. Multimodal Severity.
    Visually proves that severe multimodality concentrates perfectly in the middle of the rating scale.

    Args:
        final_df: The main evaluations dataframe.
        target_model: The specific model to analyze (e.g., 'Qwen/Qwen3.5-4B').
        target_prompt: The prompt architecture (default 'Formative').
        breakdown_by_dimension: If True, splits the plot into a 2x2 grid for each subdimension.
        save_path: Path to save the image.
    """
    # 1. Filter out absolute zeroes to isolate where actual conflict is happening
    subset = final_df[
        (final_df['model_name'] == target_model) &
        (final_df['prompt_name'] == target_prompt) &
        (final_df['multimodal_severity_score'] > 0.01)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No multimodal data found for {target_model} under {target_prompt}.")

    # --- MODE 1: Grid Breakdown by Dimension ---
    if breakdown_by_dimension and 'dimension_name' in subset.columns:
        # Create a FacetGrid density plot
        g = sns.displot(
            data=subset,
            x='mean_rating',
            y='multimodal_severity_score',
            col='dimension_name',
            col_wrap=2,
            kind='kde',
            fill=True,
            cmap='rocket_r',
            levels=10,
            thresh=0.05,
            height=4.5,
            aspect=1.2
        )

        # Overlay the scatter plot onto the grid
        g.map_dataframe(
            sns.scatterplot,
            x='mean_rating',
            y='multimodal_severity_score',
            color='black',
            alpha=0.15,
            s=10
        )

        # Format axes for all subplots
        for ax in g.axes.flat:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.axvline(2.5, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Expected Value (EV)', fontsize=12)
            ax.set_ylabel('Multimodal Severity', fontsize=12)

        g.set_titles(col_template="{col_name}", size=12, fontweight='bold')
        g.fig.suptitle(
            f'The Epistemic Valley by Dimension\nModel: {target_model}', fontsize=16, fontweight='bold', y=1.05)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()

        return g

    # --- MODE 2: Single Aggregated Plot (Original) ---
    else:
        plt.figure(figsize=(10, 6))

        ax = sns.kdeplot(
            data=subset,
            x='mean_rating',
            y='multimodal_severity_score',
            fill=True,
            cmap='rocket_r',
            levels=10,
            thresh=0.05
        )

        sns.scatterplot(
            data=subset,
            x='mean_rating',
            y='multimodal_severity_score',
            color='black',
            alpha=0.15,
            s=10,
            ax=ax
        )

        ax.set_title(
            f'The Epistemic Valley: Where Conflict Occurs\n{target_model} ({target_prompt})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Expected Value (EV)', fontsize=12)
        ax.set_ylabel('Multimodal Conflict Severity', fontsize=12)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.axvline(2.5, color='red', linestyle='--',
                   alpha=0.5, label='Scale Midpoint (2.5)')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()

        return ax


def plot_single_dimension_rating_distribution_by_model(
    final_df: pd.DataFrame,
    dimension_name: str,
    model_order: Optional[List[str]] = None,
    dimension_col: str = 'dimension_name_clean',
    model_col: str = 'model_name_clean',
    score_col: str = 'mean_rating',
    rating_range: Tuple[int, int] = (1, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    Generates a single violin plot for one selected dimension, with model architectures
    on the x-axis.

    Args:
        final_df:
            Main evaluations dataframe.

        dimension_name:
            The clean dimension name to filter on, e.g. 'Overall Quality (Naive)'.

        model_order:
            Optional list defining the left-to-right model order.

        dimension_col:
            Column containing clean dimension names.

        model_col:
            Column containing clean model names.

        score_col:
            Column containing the score to plot, e.g. 'mean_rating' or 'mode_rating'.

        rating_range:
            Tuple of rating scale minimum and maximum, e.g. (1, 4) or (1, 7).

        title:
            Optional custom title.

        save_path:
            Optional path to save the figure.

        figsize:
            Figure size.

    Returns:
        Matplotlib Figure object, or None if the filtered dataframe is empty.
    """

    df_plot = final_df.copy()

    # Filter to selected dimension
    df_plot = df_plot[df_plot[dimension_col] == dimension_name].copy()

    if df_plot.empty:
        print(
            f"WARNING: No rows found for {dimension_col} == '{dimension_name}'. "
            f"Check spelling or available values in final_df[{dimension_col!r}]."
        )
        return None

    # Determine model order
    if model_order is None:
        model_order = list(df_plot[model_col].dropna().unique())

    # Keep only models in order
    df_plot = df_plot[df_plot[model_col].isin(model_order)].copy()

    if df_plot.empty:
        print(
            "WARNING: df_plot is empty after applying model_order. "
            "Check that model_order matches the values in the model column."
        )
        return None

    y_min, y_max = rating_range
    y_ticks = list(range(y_min, y_max + 1))

    fig, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette("Set2", len(model_order))

    # Violin plot
    sns.violinplot(
        data=df_plot,
        x=model_col,
        y=score_col,
        hue=model_col,
        order=model_order,
        palette=palette,
        inner='box',
        cut=0,
        linewidth=1.5,
        legend=False,
        ax=ax
    )

    # Overlay mean line
    sns.boxplot(
        data=df_plot,
        x=model_col,
        y=score_col,
        hue=model_col,
        order=model_order,
        palette=palette,
        showmeans=True,
        meanline=True,
        meanprops={
            'color': 'white',
            'linewidth': 2.5,
            'linestyle': 'solid'
        },
        medianprops={'visible': False},
        whiskerprops={'visible': False},
        capprops={'visible': False},
        boxprops={'visible': False},
        showfliers=False,
        legend=False,
        ax=ax
    )

    ax.set_xlabel('')
    ax.set_ylabel(f'Rating ({y_min}-{y_max})', fontsize=12)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

    ax.set_xticklabels(model_order, rotation=30, ha='right')

    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    if title is None:
        title = f'Rating Distributions by Model: {dimension_name}'

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return fig


def plot_rating_distributions_by_architecture(
    final_df: pd.DataFrame,
    plot_order: List[str],
    dim_map_clean: Dict[str, str],
    score_col: str = 'mean_rating',
    rating_range: Tuple[int, int] = (1, 4),
    save_path: Optional[str] = None,
    file_format: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generates a 2x2 grid of violin plots where each subplot corresponds to a model.
    The distributions are grouped by Holistic, Macro-Formative, and Formative architectures.

    Args:
        final_df: The main evaluations dataframe.
        plot_order: List defining the left-to-right order of the categories.
        dim_map_clean: Dictionary mapping raw dimension names to plot labels.
        score_col: The column name for the scores (e.g., 'mean_rating' or 'expected_value').
        rating_range: A tuple of (min, max) for the rating scale (e.g., (1, 4) or (1, 7)).
        save_path: Path to save the image file.
    """
    df_plot = final_df.copy()

    # 1. Create the Hybrid X-Axis Category using the dictionary map
    def get_plot_category(row):
        raw_dim = row.get('dimension_name')
        # If the raw dimension is in our map, return the clean plot label
        return dim_map_clean.get(raw_dim, None)

    df_plot['plot_category'] = df_plot.apply(get_plot_category, axis=1)

    # 2. Filter for selected categories
    df_plot = df_plot[df_plot['plot_category'].isin(plot_order)]

    # Debugging check to ensure mapping worked
    if df_plot.empty:
        print("WARNING: df_plot is empty! Check that 'dimension_name' contains the exact keys defined in dim_map_clean.")
        return None

    # 3. Extract models
    models = df_plot['model_name_clean'].dropna().unique()

    # 4. Plotting Setup
    # Widened slightly for the new labels
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", len(plot_order))

    # Calculate Y-ticks based on the provided range
    y_min, y_max = rating_range
    y_ticks = list(range(y_min, y_max + 1))

    # 5. Generate each subplot
    for i, model in enumerate(models[:4]):
        ax = axes[i]
        model_data = df_plot[df_plot['model_name_clean'] == model]

        # Draw the violin plot
        sns.violinplot(
            data=model_data,
            x='plot_category',
            y=score_col,
            hue='plot_category',
            legend=False,
            order=plot_order,
            palette=palette,
            inner='box',
            cut=0,
            linewidth=1.5,
            ax=ax
        )

        # Overlay the mean line
        sns.boxplot(
            showmeans=True,
            meanline=True,
            meanprops={'color': 'white',
                       'linewidth': 2.5, 'linestyle': 'solid'},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            capprops={'visible': False},
            boxprops={'visible': False},
            showfliers=False,
            data=model_data,
            x='plot_category',
            y=score_col,
            hue='plot_category',
            legend=False,
            order=plot_order,
            ax=ax
        )

        ax.set_title(f"Model: {model}", fontsize=14, fontweight='bold')
        ax.set_xlabel('')

        # Handle Axis Ticks and Labels
        ax.set_xticks(range(len(plot_order)))
        ax.set_xticklabels(plot_order, rotation=45, ha='right')
        ax.set_yticks(y_ticks)

        if i % 2 == 0:
            ax.set_ylabel(f'Rating ({y_min}-{y_max})', fontsize=12)
        else:
            ax.set_ylabel('')

        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

    fig.suptitle(f'Rating Distributions ({y_min}-{y_max} Scale): Holistic vs Macro-Formative vs Formative',
                 fontsize=18, fontweight='bold', y=1.02)

    plt.tight_layout(h_pad=3.0)

    if save_path:
        if file_format is None:
            file_format = save_path.split(".")[-1].lower()

        fig.savefig(
            save_path,
            format=file_format,
            bbox_inches='tight',
            dpi=300
        )

        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()

    return fig


def _parse_label(label: str) -> pd.Series:
    """
    Parses a clean string like 'Gemma 4 Formative' or 'baseline'.
    """
    if str(label).lower() == 'baseline':
        return pd.Series(['Baseline', 'Baseline'])

    # rsplit splits from the right, capturing the last word as the prompt
    # "Gemma 4 Formative" -> ["Gemma 4", "Formative"]
    parts = str(label).rsplit(' ', 1)

    if len(parts) == 2:
        return pd.Series([parts[0], parts[1]])

    return pd.Series([label, 'N/A'])


def plot_ev_regression_r2(df_results: pd.DataFrame,
                          prompt_order: list[str] = [
                              'Holistic Naive', 'Holistic Informed', 'Formative'],
                          y_col='r_squared',
                          save_path: str = None) -> plt.Axes:
    """
    Generates a grouped bar chart comparing the R-squared values of EV regression
    across different models and prompt architectures.

    Args:
        df_results (pd.DataFrame): The dataframe containing 'label' and 'r_squared' columns.
        save_path (str, optional): If provided, saves the plot to this file path. 
                                   Otherwise, displays the plot interactively.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """
    # Create a copy to prevent SettingWithCopy warnings on the original dataframe
    df_plot = df_results.copy()

    # Apply the parsing logic to extract clean categories
    df_plot[['model_name', 'prompt_name']
            ] = df_plot['label'].apply(_parse_label)

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # Generate the grouped bar plot
    ax = sns.barplot(
        data=df_plot,
        x='model_name',
        y=y_col,
        hue='prompt_name',
        hue_order=prompt_order,  # Forces the desired order
        palette='Set2'
    )

    # Formatting and aesthetics
    ax.set_title('Predictive Power of EV Regression by Model and Architecture',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Name', fontsize=12)
    ax.set_ylabel('R-Squared ($R^2$)', fontsize=12)  # TODO change label

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=15)

    # Move legend outside the main plot area
    plt.legend(title='Prompt Architecture',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add horizontal grid lines behind the bars
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Handle output routing
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return ax


# # ---------------------------------------------------------
# # 1. MODIFIED BASE FUNCTION (Added `ax` support)
# # ---------------------------------------------------------
# # TODO: rewrite so it works for any two arbitrary variables

# def plot_disagreement_vs_y_var(
#     final_df: pd.DataFrame,
#     target_model: str = 'Qwen 3.5',
#     y_var: str = 'multimodal_severity_score',
#     ax: Optional[plt.Axes] = None
# ):
#     """
#     Plots some y_var across human disagreement levels,
#     broken down by dimension.
#     """
#     subset = final_df[
#         (final_df['model_name_clean'] == target_model)
#     ].copy()

#     # Check if we are plotting in a grid or standalone
#     is_standalone = ax is None
#     if is_standalone:
#         fig, ax = plt.subplots(figsize=(12, 7))

#     # Pointplot shows the mean and a confidence interval (error bar)
#     sns.pointplot(
#         data=subset,
#         x='human_disagreement',
#         y=y_var,
#         hue='analysis_group_full_clean',
#         palette='viridis',
#         # Distinct markers for each line
#         markers=['o', 's', 'D', '^', 'v', 'x'],
#         linestyles='-',
#         scale=1.2,
#         capsize=.1,
#         ax=ax
#     )

#     # Formatting
#     ax.set_title(f'Model: {target_model}', fontsize=14, fontweight='bold')
#     ax.set_xlabel('Human Disagreement (Absolute Delta)', fontsize=12)

#     if y_var == 'multimodal_severity_score':
#         y_lab = 'Mean Multimodal Severity (MCS)'
#     elif y_var == 'normalized_entropy':
#         y_lab = 'Mean Normalized Entropy'
#     else:
#         y_lab = y_var

#     ax.set_ylabel(y_lab, fontsize=12)
#     ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
#     ax.grid(axis='y', linestyle='--', alpha=0.5)

#     # Handle the legend cleanly
#     if is_standalone:
#         ax.legend(title='Dimension', bbox_to_anchor=(
#             1.05, 1), loc='upper left')
#         plt.tight_layout()
#         plt.show()
#     else:
#         # If in a grid, remove individual legends to avoid clutter
#         # The wrapper will create a master legend
#         if ax.get_legend() is not None:
#             ax.get_legend().remove()


# # ---------------------------------------------------------
# # 2. THE WRAPPER FUNCTION (Produces the 2x2 Grid)
# # ---------------------------------------------------------
# # TODO: rewrite so it works for any two arbitrary variables
# def plot_disagreement_grid(
#     final_df: pd.DataFrame,
#     y_var: str = 'multimodal_severity_score',
#     models: Optional[list] = None,
#     save_path: Optional[str] = None
# ) -> plt.Figure:
#     """
#     Generates a 2x2 grid calling `plot_disagreement_vs_y_var` for all 4 models.
#     """
#     # Extract the models (Make sure these are the exact string names in your data)
#     if not models:
#         models = final_df['model_name_clean'].dropna().unique()

#     if len(models) < 4:
#         print(
#             f"Warning: Found {len(models)} models, expected 4 for a complete 2x2 grid.")

#     # Create the 2x2 grid
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     axes = axes.flatten()

#     # Generate subplots
#     for i, model in enumerate(models[:4]):
#         # Call the base function and pass the specific grid axis to it
#         plot_disagreement_vs_y_var(
#             final_df, target_model=model, y_var=y_var, ax=axes[i])

#         # Clean up internal axis labels for a tighter grid look
#         if i % 2 != 0:
#             axes[i].set_ylabel('')  # Remove Y-label for right-column plots
#         if i < 2:
#             axes[i].set_xlabel('')  # Remove X-label for top-row plots

#     # Create a single Master Legend for the entire figure on the right side
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, title='Dimension / Prompt Group',
#                bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=11, title_fontsize=12)

#     # Add an overarching Super Title depending on the metric
#     metric_title = "Normalized Entropy" if y_var == 'normalized_entropy' else "Multimodal Conflict Severity (MCS)"
#     fig.suptitle(f'{metric_title} Scaling vs. Human Disagreement',
#                  fontsize=18, fontweight='bold', y=1.02)

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     # else:
#     #     plt.show()

#     return fig


# ---------------------------------------------------------
# 1. THE BASE FUNCTION (Now handles arbitrary X and Y)
# ---------------------------------------------------------

def plot_variable_interaction(
    final_df: pd.DataFrame,
    target_model: str,
    x_var: str = 'human_disagreement',
    y_var: str = 'multimodal_severity_score',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Base function: Plots any y_var across any x_var for a single model.
    """
    # 1. Prepare data
    subset = final_df[final_df['model_name_clean'] == target_model].copy()

    # 2. Setup Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Create the Plot
    sns.pointplot(
        data=subset,
        x=x_var,
        y=y_var,
        hue='analysis_group_full_clean',
        palette='viridis',
        markers=['o', 's', 'D', '^', 'v', 'x'],
        linestyles='-',
        scale=1.2,
        capsize=.1,
        ax=ax
    )

    # 4. Dynamic Label Formatting
    def format_label(var_name):
        labels = {
            'multimodal_severity_score': 'Mean Multimodal Severity (MCS)',
            'normalized_entropy': 'Mean Normalized Entropy',
            'human_disagreement': 'Human Disagreement (Absolute Delta)',
            'expected_value': 'Expected Value (Score)'
        }
        # Fallback: replace underscores with spaces and capitalize
        return labels.get(var_name, var_name.replace('_', ' ').title())

    ax.set_title(f'Model: {target_model}', fontsize=14, fontweight='bold')
    ax.set_xlabel(format_label(x_var), fontsize=11)
    ax.set_ylabel(format_label(y_var), fontsize=11)

    # Only apply percentage formatting to variables that need it
    if y_var in ['multimodal_severity_score', 'normalized_entropy']:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Remove individual legends
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    return ax

# ---------------------------------------------------------
# 2. THE WRAPPER FUNCTION (Dynamic Grid Size)
# ---------------------------------------------------------


def plot_variable_grid(
    final_df: pd.DataFrame,
    x_var: str = 'human_disagreement',
    y_var: str = 'multimodal_severity_score',
    models: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Generates a dynamic grid calling `plot_variable_interaction` for the given models.
    Adapts perfectly to 1, 2, 3, or 4+ models.
    """
    # Extract models if not provided
    if not models:
        models = final_df['model_name_clean'].dropna().unique().tolist()

    n_models = len(models)
    if n_models == 0:
        raise ValueError("No models found to plot.")

    # Calculate dynamic grid dimensions (max 2 columns)
    ncols = min(2, n_models)
    nrows = math.ceil(n_models / 2)

    # Create the figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))

    # Ensure `axes` is always a 1D array for easy iteration
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Generate subplots
    for i, model in enumerate(models):
        plot_variable_interaction(
            final_df, target_model=model, x_var=x_var, y_var=y_var, ax=axes[i])

        # Clean up internal axis labels for a tighter grid look
        if i % ncols != 0:
            axes[i].set_ylabel('')  # Remove Y-label for right-column plots

        # Only remove the X-label if there is another plot directly below it!
        # (This prevents missing X-labels if there are exactly 3 models)
        if (i + ncols) < n_models:
            axes[i].set_xlabel('')

    # Turn off any completely empty subplots (e.g., the 4th slot if we only have 3 models)
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    # Create a single Master Legend using the first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Dimension / Prompt Group',
               bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=11, title_fontsize=12)

    # Dynamic Super Title Helper
    def format_title(var_name):
        titles = {
            'multimodal_severity_score': 'MCS',
            'normalized_entropy': 'Normalized Entropy',
            'human_disagreement': 'Human Disagreement'
        }
        return titles.get(var_name, var_name.replace('_', ' ').title())

    fig.suptitle(f'{format_title(y_var)} vs. {format_title(x_var)}',
                 fontsize=18, fontweight='bold', y=1.05 if nrows == 1 else 1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return None

    # return fig


# BASELINE_SENTINEL = "BASELINE"


# def plot_metric_bars(
#     runner,
#     metric:      str = "pseudo_r_squared",
#     model_col:   str = None,
#     prompt_col:  str = None,
#     plot_order:  list = None,
#     plot_delta:  bool = False,  # <-- NEW: Plot difference from baseline
#     figsize:     tuple = None,
# ) -> plt.Figure:
#     """
#     Bar chart of any fit metric from runner.summarize(), grouped by model
#     then split by prompt_id within each group.
#     """

#     METRIC_LABELS = {
#         "pseudo_r_squared": "Pseudo R²",
#         "adj_r_squared":    "Adj. R²",
#         "r_squared":        "R²",
#         "aic":              "AIC",
#         "bic":              "BIC",
#         "llf":              "Log-likelihood",
#     }

#     METRIC_HIGHER_IS_BETTER = {
#         "pseudo_r_squared": True,
#         "adj_r_squared":    True,
#         "r_squared":        True,
#         "aic":              False,
#         "bic":              False,
#         "llf":              True,
#     }

#     model_col = model_col or runner.experimental_groups[0]
#     prompt_col = prompt_col or runner.experimental_groups[1]

#     # Create a copy so we don't accidentally modify the underlying summary dataframe
#     df = runner.summarize().copy()

#     if metric not in df.columns:
#         raise ValueError(f"Metric '{metric}' not in summarize() output. "
#                          f"Available: {[c for c in df.columns if c not in ('label','result','record')]}")

#     # ── Split baseline from real runs ────────────────────────────────────
#     is_baseline = df[model_col] == BASELINE_SENTINEL
#     baseline_df = df[is_baseline]
#     runs_df = df[~is_baseline].dropna(subset=[metric])

#     baseline_val = (
#         baseline_df[metric].iloc[0]
#         if not baseline_df.empty else None
#     )

#     higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
#     metric_label = METRIC_LABELS.get(metric, metric)

#     # ── Handle Delta Calculation ─────────────────────────────────────────
#     if plot_delta and baseline_val is not None:
#         runs_df[metric] = runs_df[metric] - baseline_val
#         original_baseline_val = baseline_val
#         baseline_val = 0.0  # The baseline reference line is now flat at 0

#         # Update labels to reflect it's a delta
#         metric_label = f"Δ {metric_label}"
#         if higher_is_better:
#             direction_note = "↑ positive is better"
#         else:
#             direction_note = "↓ negative is better"

#         baseline_legend_label = f"baseline ({original_baseline_val:.2f} normalized to 0)"
#     else:
#         direction_note = "↑ higher is better" if higher_is_better else "↓ lower is better"
#         baseline_legend_label = f"baseline ({baseline_val:.3f})" if baseline_val is not None else "baseline"

#     # ── Derive ordered axes ──────────────────────────────────────────────
#     unique_prompts = list(runs_df[prompt_col].unique())

#     if plot_order:
#         hue_order = [p for p in plot_order if p in unique_prompts] + \
#                     [p for p in unique_prompts if p not in plot_order]
#     else:
#         hue_order = unique_prompts

#     n_models = runs_df[model_col].nunique()
#     n_prompts = len(hue_order)

#     # ── Layout ───────────────────────────────────────────────────────────
#     # Slightly wider default width to accommodate the external legend
#     figsize = figsize or (max(7, n_models * n_prompts * 0.9 + 2.5), 4.5)
#     fig, ax = plt.subplots(figsize=figsize)

#     # ── Seaborn Barplot ──────────────────────────────────────────────────
#     sns.barplot(
#         data=runs_df,
#         x=model_col,
#         y=metric,
#         hue=prompt_col,
#         hue_order=hue_order,
#         alpha=0.85,
#         ax=ax,
#         errorbar=None,
#         zorder=3
#     )

#     # ── Baseline reference line ──────────────────────────────────────────
#     if baseline_val is not None:
#         ax.axhline(
#             baseline_val,
#             color="dimgray", linestyle="--", linewidth=1.4,
#             zorder=4, label=baseline_legend_label,
#         )

#     # ── Axes cosmetics ───────────────────────────────────────────────────
#     ax.set_ylabel(metric_label, fontsize=11)
#     ax.set_xlabel(model_col, fontsize=11)

#     ax.yaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7, zorder=0)
#     ax.set_axisbelow(True)
#     sns.despine(ax=ax)

#     ax.set_title(
#         f"{metric_label} by {model_col} and {prompt_col}   "
#         f"({direction_note})",
#         fontsize=12, pad=10,
#     )

#     # ── Fix Legend Positioning ───────────────────────────────────────────
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(
#         handles=handles, labels=labels,
#         title=prompt_col,
#         fontsize=9, title_fontsize=9,
#         frameon=True, framealpha=0.9,
#         # Move legend outside the plot area
#         loc="upper left",
#         bbox_to_anchor=(1.02, 1)
#     )

#     # tight_layout will automatically account for the external legend
#     fig.tight_layout()
#     # return fig


BASELINE_SENTINEL = "BASELINE"


def plot_metric_bars(
    runner=None,
    df:          pd.DataFrame = None,
    metric:      str = "pseudo_r_squared",
    model_col:   str = None,
    prompt_col:  str = None,
    plot_order:  list = None,
    plot_delta:  bool = False,
    figsize:     tuple = None,
):
    """
    Bar chart of any fit metric from runner.summarize() or a direct DataFrame, 
    grouped by model then split by prompt_id within each group.
    """
    if runner is None and df is None:
        raise ValueError("Must provide either 'runner' or 'df'.")

    # If df is provided directly, we require model_col and prompt_col
    if df is None:
        df = runner.summarize().copy()
        model_col = model_col or runner.experimental_groups[0]
        prompt_col = prompt_col or runner.experimental_groups[1]
    else:
        df = df.copy()
        if not model_col or not prompt_col:
            raise ValueError(
                "If providing 'df', must specify 'model_col' and 'prompt_col'.")

    METRIC_LABELS = {
        "pseudo_r_squared": "Pseudo R²",
        "adj_r_squared":    "Adj. R²",
        "r_squared":        "R²",
        "aic":              "AIC",
        "bic":              "BIC",
        "llf":              "Log-likelihood",
    }

    METRIC_HIGHER_IS_BETTER = {
        "pseudo_r_squared": True,
        "adj_r_squared":    True,
        "r_squared":        True,
        "aic":              False,
        "bic":              False,
        "llf":              True,
    }

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not in DataFrame. Available: {list(df.columns)}")

    higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
    metric_label = METRIC_LABELS.get(metric, metric)

    # ── Split baseline from real runs ────────────────────────────────────
    is_baseline = df[model_col] == BASELINE_SENTINEL
    baseline_df = df[is_baseline]
    runs_df = df[~is_baseline].dropna(subset=[metric])

    baseline_val = baseline_df[metric].iloc[0] if not baseline_df.empty else None

    # ── Handle Delta Calculation ─────────────────────────────────────────
    if plot_delta:
        # If we have a baseline row, subtract it from all runs
        if baseline_val is not None:
            runs_df[metric] = runs_df[metric] - baseline_val
            original_baseline_val = baseline_val
            baseline_legend_label = f"baseline ({original_baseline_val:.2f} normalized to 0)"
        else:
            # If no baseline row exists, assume the user passed a pre-computed Delta DataFrame
            baseline_legend_label = "0 reference (no difference)"

        baseline_val = 0.0  # The baseline reference line is flat at 0

        # Update labels to reflect it's a delta
        metric_label = f"Δ {metric_label}"
        if higher_is_better:
            direction_note = "↑ positive is better"
        else:
            direction_note = "↓ negative is better"

    else:
        direction_note = "↑ higher is better" if higher_is_better else "↓ lower is better"
        baseline_legend_label = f"baseline ({baseline_val:.3f})" if baseline_val is not None else "baseline"

    # ── Derive ordered axes ──────────────────────────────────────────────
    unique_prompts = list(runs_df[prompt_col].unique())

    if plot_order:
        hue_order = [p for p in plot_order if p in unique_prompts] + \
                    [p for p in unique_prompts if p not in plot_order]
    else:
        hue_order = unique_prompts

    n_models = runs_df[model_col].nunique()
    n_prompts = len(hue_order)

    # ── Layout ───────────────────────────────────────────────────────────
    figsize = figsize or (max(7, n_models * n_prompts * 0.9 + 2.5), 4.5)
    fig, ax = plt.subplots(figsize=figsize)

    # ── Seaborn Barplot ──────────────────────────────────────────────────
    sns.barplot(
        data=runs_df,
        x=model_col,
        y=metric,
        hue=prompt_col,
        hue_order=hue_order,
        alpha=0.85,
        ax=ax,
        errorbar=None,
        zorder=3
    )

    # ── Baseline reference line ──────────────────────────────────────────
    if baseline_val is not None:
        ax.axhline(
            baseline_val,
            color="dimgray", linestyle="--", linewidth=1.4,
            zorder=4, label=baseline_legend_label,
        )

    # ── Axes cosmetics ───────────────────────────────────────────────────
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_xlabel(model_col, fontsize=11)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    ax.set_title(
        f"{metric_label} by {model_col} and {prompt_col}\n"
        f"({direction_note})",
        fontsize=12, pad=10,
    )

    # ── Fix Legend Positioning ───────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles=handles, labels=labels,
            title=prompt_col,
            fontsize=9, title_fontsize=9,
            frameon=True, framealpha=0.9,
            loc="upper left",
            bbox_to_anchor=(1.02, 1)
        )

    fig.tight_layout()
    return fig


def plot_single_dimension_rating_distribution_by_model_pdf(
    final_df: pd.DataFrame,
    dimension_name: str,
    save_path: str,
    model_order: Optional[List[str]] = None,
    dimension_col: str = 'dimension_name_clean',
    model_col: str = 'model_name_clean',
    score_col: str = 'mean_rating',
    rating_range: Tuple[int, int] = (1, 4),
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Generates a single-dimension violin plot with model architectures on the x-axis
    and saves it as a PDF.

    Args:
        final_df:
            Main evaluations dataframe.

        dimension_name:
            Clean dimension name to filter on, e.g. 'Overall Quality (Naive)'.

        save_path:
            Output path. Should end in '.pdf'.

        model_order:
            Optional list defining left-to-right model order.

        dimension_col:
            Column containing clean dimension names.

        model_col:
            Column containing clean model names.

        score_col:
            Column containing the score to plot, e.g. 'mean_rating'.

        rating_range:
            Rating scale minimum and maximum, e.g. (1, 4) or (1, 7).

        title:
            Optional custom title.

        figsize:
            Figure size.

        dpi:
            Export resolution. For vector PDF this mostly affects rasterized elements.

        show:
            Whether to display the plot in the notebook after saving.

    Returns:
        Matplotlib Figure object, or None if the filtered dataframe is empty.
    """

    if not save_path.lower().endswith(".pdf"):
        raise ValueError("save_path should end with '.pdf'.")

    df_plot = final_df.copy()

    # Filter to selected dimension
    df_plot = df_plot[df_plot[dimension_col] == dimension_name].copy()

    if df_plot.empty:
        print(
            f"WARNING: No rows found for {dimension_col} == '{dimension_name}'. "
            f"Check spelling or inspect final_df[{dimension_col!r}].unique()."
        )
        return None

    # Determine model order
    if model_order is None:
        model_order = list(df_plot[model_col].dropna().unique())

    # Keep only selected models
    df_plot = df_plot[df_plot[model_col].isin(model_order)].copy()

    if df_plot.empty:
        print(
            "WARNING: df_plot is empty after applying model_order. "
            "Check that model_order matches the values in the model column."
        )
        return None

    y_min, y_max = rating_range
    y_ticks = list(range(y_min, y_max + 1))

    fig, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette("Set2", len(model_order))

    sns.violinplot(
        data=df_plot,
        x=model_col,
        y=score_col,
        hue=model_col,
        order=model_order,
        palette=palette,
        inner='box',
        cut=0,
        linewidth=1.5,
        legend=False,
        ax=ax
    )

    # Overlay mean line
    sns.boxplot(
        data=df_plot,
        x=model_col,
        y=score_col,
        hue=model_col,
        order=model_order,
        palette=palette,
        showmeans=True,
        meanline=True,
        meanprops={
            'color': 'white',
            'linewidth': 2.5,
            'linestyle': 'solid'
        },
        medianprops={'visible': False},
        whiskerprops={'visible': False},
        capprops={'visible': False},
        boxprops={'visible': False},
        showfliers=False,
        legend=False,
        ax=ax
    )

    ax.set_xlabel('')
    ax.set_ylabel(f'Expected-value rating ({y_min}-{y_max})', fontsize=12)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30, ha='right')

    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
