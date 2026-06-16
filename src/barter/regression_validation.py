from typing import Optional, List, Dict
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Dict, Any, Optional, List, Union
from statsmodels.stats.outliers_influence import variance_inflation_factor


class PredictiveValidityExplorer:
    """
    Modular toolkit for mapping psychometric predictions against behavioral KPIs.
    Exposes atomic methods for dynamic, on-demand data extraction and visualization.
    """

    def __init__(self, runner):
        self.runner = runner
        self.target_col = runner.dep_var

    # =============================================================================
    # DATA EXTRACTION
    # =============================================================================

    def get_observation_data(
        self,
        label: str,
        merge_cols: Optional[List[str]] = None,
        source_df: Optional[pd.DataFrame] = None,
        include_ratings: bool = True  # <-- New parameter to toggle features
    ) -> pd.DataFrame:
        """
        Extracts row-wise actuals and predictions, computes localized residuals.
        Securely maps requested columns from either the internal long_df or an external source_df.
        Optionally joins the original feature ratings (X) used by the model.
        """
        if merge_cols is None:
            merge_cols = ["deal_id", "deal_text"]

        for rec in self.runner.records:
            if rec.label == label:
                if rec.y is not None and rec.y_pred is not None:
                    # 1. Initialize dataframe with strict index preservation
                    df = pd.DataFrame({
                        "actual": rec.y,
                        "predicted": rec.y_pred
                    }, index=rec.y.index).dropna()

                    # 3. Extract deal_id first (mandatory for external mapping)
                    if "deal_id" in rec.long_df.columns:
                        df["deal_id"] = rec.long_df["deal_id"]

                    # 4. Safely map requested columns
                    for col in merge_cols:
                        if col in df.columns:
                            # Already handled (e.g., deal_id or a feature in X)
                            continue

                        # Try internal long_df first
                        if col in rec.long_df.columns:
                            df[col] = rec.long_df[col]

                        # If missing, fallback to secure mapping via source_df (og_df)
                        elif source_df is not None and col in source_df.columns:
                            # Create a clean 1:1 mapping dictionary to preserve the exact index
                            mapping_series = source_df.drop_duplicates(
                                "deal_id").set_index("deal_id")[col]
                            df[col] = df["deal_id"].map(mapping_series)

                        else:
                            print(
                                f"Warning: '{col}' not found in long_df, X, or provided source_df.")

                    # 5. Compute Residuals
                    df["error"] = df["actual"] - df["predicted"]
                    df["squared_error"] = df["error"] ** 2
                    df["absolute_error"] = np.abs(df["error"])

                    # 6. Rank-Order Residuals & Displacements
                    # Calculate base ranks (using 'average' method to handle ties correctly)
                    df["actual_rank"] = df["actual"].rank()
                    df["predicted_rank"] = df["predicted"].rank()

                    # Absolute and Percentile rank metrics
                    n_obs = len(df)
                    df["actual_percentile"] = df["actual_rank"] / n_obs
                    df["predicted_percentile"] = df["predicted_rank"] / n_obs

                    df["rank_deviation"] = df["actual_rank"] - \
                        df["predicted_rank"]
                    df["abs_rank_displacement"] = np.abs(df["rank_deviation"])
                    # Normalized displacement as a percentage of the dataset size
                    df["normalized_displacement"] = df["abs_rank_displacement"] / \
                        (n_obs - 1) if n_obs > 1 else 0

                    # 2. Add the original model ratings (Features)
                    if include_ratings and hasattr(rec, 'X') and rec.X is not None:
                        # Find columns in X that aren't already in df to prevent collisions
                        x_cols = [
                            c for c in rec.X.columns if c not in df.columns]
                        df = df.join(rec.X[x_cols])

                    return df
        raise ValueError(
            f"No valid prediction data found for specification: {label}")

    # =============================================================================
    # METRIC COMPUTATION
    # =============================================================================

    def compute_rank_correlations(self) -> Dict[str, pd.Series]:
        """
        Computes the observation-level Spearman rank-order correlation for all models.
        Returns a dictionary mapping model labels to correlation statistics.
        """
        results = {}
        for label, obs_df in self.get_all_observation_data().items():
            if len(obs_df) > 1:
                rho, p_val = stats.spearmanr(
                    obs_df["actual"], obs_df["predicted"])
                results[label] = pd.Series(
                    {"spearman_rho": rho, "p_value": p_val, "n_obs": len(obs_df)})
            else:
                results[label] = pd.Series(
                    {"spearman_rho": np.nan, "p_value": np.nan, "n_obs": 0})
        return results

    def build_macro_summary(self) -> pd.DataFrame:
        """
        Synthesizes macro-level statsmodels fit metrics with computed micro-level correlations.
        """
        summary_df = self.runner.to_frame()
        if summary_df.empty:
            return summary_df

        summary_df = summary_df.set_index("label")
        corr_dict = self.compute_rank_correlations()

        # Inject micro-level rank correlations into the macro summary
        summary_df["spearman_rho"] = summary_df.index.map(
            lambda l: corr_dict.get(l, pd.Series()).get("spearman_rho", np.nan))
        summary_df["spearman_p_value"] = summary_df.index.map(
            lambda l: corr_dict.get(l, pd.Series()).get("p_value", np.nan))

        return summary_df

    def compute_rank_diagnostics(self, target_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Computes granular rank displacement statistics and quantile overlaps 
        to diagnose severe local misrankings.
        """
        labels = target_labels or [rec.label for rec in self.runner.records]
        diagnostics = []

        for label in labels:
            try:
                obs_df = self.get_observation_data(label)
                if obs_df.empty:
                    continue

                # Displacement stats
                norm_disp = obs_df["normalized_displacement"]

                # Quantile Overlaps (Top 10% and Bottom 10%)
                top_10_actual = obs_df["actual_percentile"] >= 0.90
                top_10_pred = obs_df["predicted_percentile"] >= 0.90
                bottom_10_actual = obs_df["actual_percentile"] <= 0.10
                bottom_10_pred = obs_df["predicted_percentile"] <= 0.10

                # Intersection over Actual (Recall in the extreme quantiles)
                top_10_overlap = (top_10_actual & top_10_pred).sum(
                ) / top_10_actual.sum() if top_10_actual.sum() > 0 else 0
                bottom_10_overlap = (bottom_10_actual & bottom_10_pred).sum(
                ) / bottom_10_actual.sum() if bottom_10_actual.sum() > 0 else 0

                diagnostics.append({
                    "Model": label,
                    "Mean Rank Error %": norm_disp.mean() * 100,
                    "Median Rank Error %": norm_disp.median() * 100,
                    "90th Pct Rank Error %": norm_disp.quantile(0.90) * 100,
                    "Max Rank Error %": norm_disp.max() * 100,
                    "Top 10% Overlap": top_10_overlap * 100,
                    "Bottom 10% Overlap": bottom_10_overlap * 100
                })
            except ValueError:
                continue

        return pd.DataFrame(diagnostics).set_index("Model")

   # =============================================================================
   # VISUALIZATION ENGINES
   # =============================================================================

    def plot_observation_level(
        self,
        x: str = "predicted",
        y: str = "actual",
        labels: Optional[List[str]] = None,
        source_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generates observation-level scatter plots for arbitrary vector pairs.
        Dynamically fetches external columns from source_df if x or y are not
        native residual/prediction vectors.
        """
        target_labels = labels or [rec.label for rec in self.runner.records]

        # 1. Dynamically identify which columns need to be merged from the source data
        native_cols = {
            "actual", "predicted", "error", "squared_error", "absolute_error",
            "actual_rank", "predicted_rank", "rank_deviation"
        }

        # Always keep deal_id for structural integrity
        merge_cols = ["deal_id"]
        if x not in native_cols and x not in merge_cols:
            merge_cols.append(x)
        if y not in native_cols and y not in merge_cols:
            merge_cols.append(y)

        # 2. Extract data safely, skipping missing labels without breaking the loop
        obs_data_dict = {}
        for l in target_labels:
            try:
                obs_data_dict[l] = self.get_observation_data(
                    label=l,
                    merge_cols=merge_cols,
                    source_df=source_df
                )
            except ValueError as e:
                print(f"Skipping {l}: {e}")
                continue

        figures = {}

        # 3. Generate figures
        for label, obs_df in obs_data_dict.items():
            if x not in obs_df.columns or y not in obs_df.columns:
                print(f"Skipping plot for {label}: Missing columns '{x}' or '{y}'. "
                      f"Ensure they exist in long_df or the provided source_df.")
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=obs_df, x=x, y=y,
                            alpha=0.4, edgecolor=None, ax=ax)

            # Identity Line for direct prediction comparison
            if x == "predicted" and y == "actual":
                min_val = min(obs_df[x].min(), obs_df[y].min())
                max_val = max(obs_df[x].max(), obs_df[y].max())
                ax.plot([min_val, max_val], [min_val, max_val],
                        color='red', linestyle='--', label='Perfect Fit')
                ax.legend()

            # Zero-baseline for residual and deviation plots
            elif "error" in y or "deviation" in y:
                ax.axhline(0, color='red', linestyle='--', label='Zero Error')
                ax.legend()

            # Dynamic formatting
            ax.set_title(
                f"Diagnostic: {label}\n{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}")
            ax.set_xlabel(x.replace('_', ' ').title())
            ax.set_ylabel(y.replace('_', ' ').title())
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()

            figures[label] = fig
            plt.close(fig)

        return figures

    def plot_macro_level(self, x: str, y: str) -> plt.Figure:
        """
        Generates a systemic scatter plot (scalar relationships across models).
        Flexibly maps any model-level metrics (e.g., x='mse', y='spearman_rho').
        Returns a single figure mapping the Pareto distribution of the specifications.
        """
        summary_df = self.build_macro_summary()

        if x not in summary_df.columns or y not in summary_df.columns:
            raise ValueError(
                f"Columns {x} and/or {y} not found in macro summary.")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(
            data=summary_df, x=x, y=y,
            scatter_kws={'alpha': 0.7, 's': 50},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax
        )

        for label, row in summary_df.iterrows():
            ax.text(row[x], row[y], f" {label}", fontsize=8, alpha=0.8)

        ax.set_title(f"Structural Validity: {y.upper()} vs {x.upper()}")
        ax.set_xlabel(x.replace('_', ' ').title())
        ax.set_ylabel(y.replace('_', ' ').title())
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()

        plt.close(fig)
        return fig

    def plot_rank_decile_matrix(
        self,
        target_labels: Optional[List[str]] = None
    ) -> Dict[str, plt.Figure]:
        """
        Plots rank-decile confusion matrices for evaluated models. 
        Severe failures appear as heavy mass in the off-diagonal corners 
        (e.g., actual decile 10 predicted as decile 1).
        Returns a dictionary mapping model labels to their respective figures.
        """
        labels = target_labels or [rec.label for rec in self.runner.records]
        figures = {}

        for label in labels:
            try:
                obs_df = self.get_observation_data(label)
                if obs_df.empty:
                    continue

                # Safely bin the pre-calculated percentiles into 10 deciles (1 to 10)
                bins = np.linspace(0, 1, 11)
                obs_df['actual_decile'] = pd.cut(
                    obs_df['actual_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
                )
                obs_df['predicted_decile'] = pd.cut(
                    obs_df['predicted_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
                )

                # Create the normalized confusion matrix (columns sum to 100%)
                cm = pd.crosstab(
                    obs_df['predicted_decile'],
                    obs_df['actual_decile'],
                    normalize='columns'
                ) * 100

                # Reverse the Y-axis so Decile 10 is at the top (standard visual flow)
                cm = cm.iloc[::-1]

                fig, ax = plt.subplots(figsize=(8, 7))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt=".1f",
                    cmap="Blues",
                    cbar_kws={'label': '% of Actual Decile'},
                    ax=ax
                )

                ax.set_title(f"Rank-Decile Confusion Matrix\n{label}", pad=15)
                ax.set_xlabel("Actual Decile (10 = Best Deals)")
                ax.set_ylabel("Predicted Decile (10 = Best Deals)")
                fig.tight_layout()

                figures[label] = fig
                plt.close(fig)  # Prevent inline display during generation

            except ValueError as e:
                print(f"Skipping {label}: {e}")
                continue

        return figures

    def plot_decile_improvement_matrix(
        self,
        baseline_label: str = "Baseline",
        target_labels: Optional[List[str]] = None
    ) -> Dict[str, plt.Figure]:
        """
        Plots the difference between a target model's confusion matrix and the baseline's.
        Blue (Positive) on the diagonal means the target model is more accurate.
        Red (Negative) in the off-diagonals means the target model makes fewer severe errors.
        """
        # 1. Compute Baseline Confusion Matrix
        try:
            base_df = self.get_observation_data(baseline_label)
            bins = np.linspace(0, 1, 11)
            base_df['actual_decile'] = pd.cut(
                base_df['actual_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
            )
            base_df['predicted_decile'] = pd.cut(
                base_df['predicted_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
            )
            base_cm = pd.crosstab(
                base_df['predicted_decile'], base_df['actual_decile'], normalize='columns'
            ) * 100
        except Exception as e:
            raise ValueError(
                f"Could not compute baseline matrix for '{baseline_label}': {e}")

        # 2. Iterate through target models
        labels = target_labels or [
            rec.label for rec in self.runner.records if rec.label != baseline_label]
        figures = {}

        for label in labels:
            try:
                obs_df = self.get_observation_data(label)
                if obs_df.empty:
                    continue

                obs_df['actual_decile'] = pd.cut(
                    obs_df['actual_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
                )
                obs_df['predicted_decile'] = pd.cut(
                    obs_df['predicted_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
                )

                target_cm = pd.crosstab(
                    obs_df['predicted_decile'], obs_df['actual_decile'], normalize='columns'
                ) * 100

                # 3. Compute Delta (Target - Baseline)
                delta_cm = target_cm - base_cm
                # Standard visual flow (10 at the top)
                delta_cm = delta_cm.iloc[::-1]

                fig, ax = plt.subplots(figsize=(8, 7))

                # Use a diverging colormap centered at zero
                sns.heatmap(
                    delta_cm,
                    annot=True,
                    fmt="+0.1f",
                    cmap="RdBu",  # Red for negative, Blue for positive
                    center=0,
                    cbar_kws={
                        'label': 'Percentage Point Difference vs Baseline'},
                    ax=ax
                )

                ax.set_title(
                    f"Incremental Rank-Decile Accuracy\n{label} vs {baseline_label}", pad=15)
                ax.set_xlabel("Actual Decile (10 = Best Deals)")
                ax.set_ylabel("Predicted Decile (10 = Best Deals)")
                fig.tight_layout()

                figures[label] = fig
                plt.close(fig)

            except ValueError as e:
                print(f"Skipping {label}: {e}")
                continue

        return figures

    def plot_decile_error_reduction(
        self,
        baseline_label: str = "baseline",
        target_labels: Optional[List[str]] = None
    ) -> Dict[str, plt.Figure]:
        """
        Plots the reduction in Mean Rank Error per actual decile.
        Bars pointing UP (Blue) mean the LLM reduced the error for that decile.
        """
        base_df = self.get_observation_data(baseline_label)
        bins = np.linspace(0, 1, 11)
        base_df['actual_decile'] = pd.cut(
            base_df['actual_percentile'], bins=bins, labels=range(1, 11))

        # Calculate how many deciles off the baseline was on average
        base_df['decile_error'] = np.abs(base_df['actual_decile'].astype(int) -
                                         pd.cut(base_df['predicted_percentile'], bins=bins, labels=range(1, 11)).astype(int))
        base_errors = base_df.groupby('actual_decile', observed=True)[
            'decile_error'].mean()

        labels = target_labels or [
            rec.label for rec in self.runner.records if rec.label != baseline_label]
        figures = {}

        for label in labels:
            try:
                obs_df = self.get_observation_data(label)
                obs_df['actual_decile'] = pd.cut(
                    obs_df['actual_percentile'], bins=bins, labels=range(1, 11))
                obs_df['decile_error'] = np.abs(obs_df['actual_decile'].astype(int) -
                                                pd.cut(obs_df['predicted_percentile'], bins=bins, labels=range(1, 11)).astype(int))
                target_errors = obs_df.groupby('actual_decile', observed=True)[
                    'decile_error'].mean()

                # Improvement = Baseline Error - Target Error (Positive is good)
                improvement = base_errors - target_errors

                fig, ax = plt.subplots(figsize=(10, 5))

                # Color bars based on value
                colors = ['#1f77b4' if val >
                          0 else '#d62728' for val in improvement]
                improvement.plot(kind='bar', ax=ax,
                                 color=colors, edgecolor='black')

                ax.axhline(0, color='black', linewidth=1.2)
                ax.set_title(
                    f"Average Rank Error Reduction by Decile\n{label} vs {baseline_label}", pad=15)
                ax.set_xlabel(
                    "Actual Deal Performance (Decile 10 = Viral Hits)")
                ax.set_ylabel("Improvement (Fewer Deciles Off)")

                # Add text annotations
                for p in ax.patches:
                    val = p.get_height()
                    ax.annotate(f"{val:+.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom' if val > 0 else 'top',
                                xytext=(0, 5 if val > 0 else -15), textcoords='offset points')

                plt.xticks(rotation=0)
                fig.tight_layout()
                figures[label] = fig
                plt.close(fig)
            except Exception as e:
                continue

        return figures

    # =============================================================================
    # STRUCTURAL DIAGNOSTICS
    # =============================================================================

    def compute_vifs(self) -> Dict[str, pd.DataFrame]:
        """
        Computes the Variance Inflation Factor (VIF) for the design matrices of 
        all evaluated specifications to diagnose severe multicollinearity.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary mapping model labels to their respective VIF DataFrames.
        """
        vif_results = {}

        for rec in self.runner.records:
            label = rec.label
            X = rec.X

            if X is None or X.empty:
                print(
                    f"Skipping VIF computation for {label}: Design matrix missing.")
                continue

            # Suppress division by zero warnings for perfectly collinear constants
            with np.errstate(divide='ignore'):
                vif_data = pd.DataFrame({
                    "feature": X.columns,
                    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                })

            # Filter out the intercept constraint and sort by severity
            vif_results[label] = (
                vif_data[vif_data["feature"] != "const"]
                .sort_values("VIF", ascending=False)
                .reset_index(drop=True)
            )

        return vif_results

    def compute_item_level_improvement_table(
        self,
        baseline_label: str = "Baseline",
        target_labels: Optional[List[str]] = None,
        severe_error_threshold: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        Computes item-level decile improvements (Target vs Baseline).
        Strictly merges on `deal_id` to prevent index-scrambling.
        """
        try:
            base_df = self.get_observation_data(baseline_label)
        except Exception as e:
            raise ValueError(
                f"Could not load baseline '{baseline_label}': {e}")

        bins = np.linspace(0, 1, 11)

        # Ensure deal_id is a column for strict merging
        if 'deal_id' not in base_df.columns:
            base_df = base_df.reset_index(names='deal_id')

        # 1. Baseline Calculations
        base_df['actual_decile'] = pd.cut(
            base_df['actual_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
        ).astype(float)

        base_df['base_pred_decile'] = pd.cut(
            base_df['predicted_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
        ).astype(float)

        base_df['base_error'] = (
            base_df['actual_decile'] - base_df['base_pred_decile']).abs()
        base_df['base_severe'] = (
            base_df['base_error'] >= severe_error_threshold).astype(int)

        # Keep only necessary baseline columns
        base_clean = base_df[['deal_id', 'actual_decile',
                              'base_error', 'base_severe']].copy()

        labels = target_labels or [
            rec.label for rec in self.runner.records if rec.label != baseline_label]
        results = {}

        for label in labels:
            try:
                tgt_df = self.get_observation_data(label)
                if tgt_df.empty:
                    continue

                if 'deal_id' not in tgt_df.columns:
                    tgt_df = tgt_df.reset_index(names='deal_id')

                # 2. Target Calculations
                tgt_df['tgt_pred_decile'] = pd.cut(
                    tgt_df['predicted_percentile'], bins=bins, labels=range(1, 11), include_lowest=True
                ).astype(float)

                tgt_clean = tgt_df[['deal_id', 'tgt_pred_decile']].copy()

                # 3. STRICT MERGE ON DEAL_ID
                merged = pd.merge(base_clean, tgt_clean,
                                  on='deal_id', how='inner')

                merged['tgt_error'] = (
                    merged['actual_decile'] - merged['tgt_pred_decile']).abs()
                merged['tgt_severe'] = (
                    merged['tgt_error'] >= severe_error_threshold).astype(int)

                # Delta: Positive means Target has LESS error than Baseline (Improvement)
                merged['delta'] = merged['base_error'] - merged['tgt_error']

                # 4. Group by actual decile
                grouped = merged.groupby('actual_decile', observed=True)

                stats = pd.DataFrame({
                    'N (Deals)': grouped.size(),
                    'Mean Δ': grouped['delta'].mean(),
                    'Median Δ': grouped['delta'].median(),
                    '% Improved': grouped.apply(lambda x: (x['delta'] > 0).mean() * 100),
                    '% Unchanged': grouped.apply(lambda x: (x['delta'] == 0).mean() * 100),
                    '% Worsened': grouped.apply(lambda x: (x['delta'] < 0).mean() * 100),
                    'Base Severe %': grouped['base_severe'].mean() * 100,
                    'Target Severe %': grouped['tgt_severe'].mean() * 100
                })

                stats['Severe Error Reduction (pp)'] = stats['Base Severe %'] - \
                    stats['Target Severe %']

                # 5. Add OVERALL summary row
                overall = pd.DataFrame({
                    'N (Deals)': len(merged),
                    'Mean Δ': merged['delta'].mean(),
                    'Median Δ': merged['delta'].median(),
                    '% Improved': (merged['delta'] > 0).mean() * 100,
                    '% Unchanged': (merged['delta'] == 0).mean() * 100,
                    '% Worsened': (merged['delta'] < 0).mean() * 100,
                    'Base Severe %': merged['base_severe'].mean() * 100,
                    'Target Severe %': merged['tgt_severe'].mean() * 100
                }, index=['OVERALL'])
                overall['Severe Error Reduction (pp)'] = overall['Base Severe %'] - \
                    overall['Target Severe %']

                stats = pd.concat([stats, overall])
                results[label] = stats.round(1)

            except Exception as e:
                print(f"Skipping {label}: {e}")

        return results
