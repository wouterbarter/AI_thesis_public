import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_calibration_correlation_comparisons(df_corr: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Generates two stacked bar charts comparing the Spearman correlation (ρ) 
    between human disagreement and entropy across models and prompt architectures.

    Args:
        df_corr (pd.DataFrame): The dataframe containing 'model_name', 
                                'prompt_name', and 'calibration_corr' columns.
        save_path (str, optional): If provided, saves the plot to this file path. 
                                   Otherwise, displays the plot interactively.

    Returns:
        matplotlib.figure.Figure: The figure object containing the subplots.
    """

    prompt_order = ['Holistic Naive', 'Holistic Informed', 'Formative']

    # Set up the matplotlib figure (two subplots stacked vertically)
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # --- Plot 1: Grouped by Model, Hue by Prompt Name ---
    sns.barplot(
        data=df_corr,
        x='model_name',
        y='calibration_corr',
        hue='prompt_name',
        hue_order=prompt_order,  # Controls the order of the sub-bars
        ax=axes[0],
        palette='viridis',
        capsize=0.1  # Adds the little caps to the error bars
    )
    axes[0].set_title('Calibration Correlation Grouped by Model',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model Name', fontsize=12)
    axes[0].set_ylabel('Spearman Correlation (ρ)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend(title='Prompt Architecture',
                   bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].set_axisbelow(True)

    # --- Plot 2: Grouped by Prompt Name, Hue by Model Name ---
    sns.barplot(
        data=df_corr,
        x='prompt_name',
        y='calibration_corr',
        hue='model_name',
        order=prompt_order,  # Controls the order of the main x-axis categories
        ax=axes[1],
        palette='magma',
        capsize=0.1
    )
    axes[1].set_title(
        'Calibration Correlation Grouped by Prompt Architecture', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Prompt Architecture', fontsize=12)
    axes[1].set_ylabel('Spearman Correlation (ρ)', fontsize=12)
    axes[1].legend(title='Model Name', bbox_to_anchor=(
        1.05, 1), loc='upper left')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].set_axisbelow(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Handle output routing
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    # return fig


def _parse_label(label: str) -> pd.Series:
    """
    Private helper function to extract the model name and prompt architecture 
    from the raw label string.
    """
    parts = str(label).split('_', 1)
    model_name = parts[0]

    if len(parts) > 1:
        prompt_desc = parts[1].lower()
        if 'formative' in prompt_desc:
            prompt_name = 'Formative'
        elif 'holistic informed' in prompt_desc:
            prompt_name = 'Holistic Informed'
        elif 'holistic naive' in prompt_desc:
            prompt_name = 'Holistic Naive'
        else:
            prompt_name = 'Unknown'
    else:
        prompt_name = 'Unknown'

    return pd.Series([model_name, prompt_name])


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


def plot_entropy_disagreement_violin_grid(df):
    """
    Generates a 2x2 grid of violin plots comparing normalized_entropy 
    and human_disagreement across different models and prompts.
    """
    # sns.catplot automatically builds a FacetGrid based on the 'col' parameter
    g = sns.catplot(
        data=df,
        x='human_disagreement',
        y='normalized_entropy',
        hue='prompt_name',
        col='model_name',       # Creates a new subplot for each model
        # Forces the subplots into a 2-column grid (2x2)
        col_wrap=2,
        kind='violin',          # Specifies the type of plot
        palette='husl',
        inner='quartile',       # Shows 25th, 50th, and 75th percentiles
        cut=0,                  # Truncates the violin tails at data limits
        density_norm='width',   # Keeps violin widths consistent
        height=5,               # Height of each individual subplot (in inches)
        aspect=1.2,             # Width-to-height ratio of each subplot
        sharey=True             # Shares the Y-axis scale across all plots for accurate comparison
    )

    # Clean up the axis labels and titles
    g.set_axis_labels("Human Disagreement (Ordinal)", "Normalized Entropy")
    g.set_titles(col_template="Model: {col_name}", size=13)

    # Add a main title for the entire figure
    g.fig.suptitle('Density Mass of Entropy across Human Disagreement',
                   y=1.05, fontsize=16, fontweight='bold')

    # Adjust layout to ensure the global title and legend don't overlap
    g.tight_layout()
    plt.show()
