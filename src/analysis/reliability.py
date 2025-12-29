import pandas as pd
import numpy as np
from .metrics import compute_entropy, calculate_reliability_gap

class ReliabilityAnalyzer:
    def __init__(self, df, group_cols, llm_rating_col = 'mode_rating', col_map=None):
        """
        Args:
            df: The main dataframe
            group_cols: List of columns to group by (['model', 'prompt', ...])
            col_map: Dictionary mapping standard names to your df columns
                     e.g. {'h1': 'score_1', 'h2': 'score_2', 'llm': 'score_llm'}
        """
        self.df = df.copy()
        self.group_cols = group_cols
        self.llm_rating_col = llm_rating_col
        
        # Default mapping if none provided
        self.cols = col_map or {
            'h1': 'score_1', 
            'h2': 'score_2', 
            'llm': llm_rating_col,
            'logits': 'sorted_logits'
        }

    def compute_reliability_gap(self, metric = 'kappa'):
        """
        Calculates the Reliability Gap using either Cohen's Kappa or Spearman correlation.
        
        Args:
            metric (str): 'kappa' (QWK) or 'spearman' (rho).
            n_bootstrap (int): Number of bootstrap iterations (0 for fast point estimate).
        """
        results = []
        if metric == "kappa" and self.llm_rating_col == "mean_rating":
            print("Using mean rating but using metric for categorical data. Automatically rounding to nearest int.")
            self.df[self.llm_rating_col] = self.df[self.llm_rating_col].round().astype(int)
            # llm = llm.round().astype(int)


        for name, group in self.df.groupby(self.group_cols):
            # Handle name being a tuple or single value
            if not isinstance(name, tuple): name = (name,)
            
            # Extract arrays
            h1 = group[self.cols['h1']]
            h2 = group[self.cols['h2']]
            llm = group[self.cols['llm']]
            
            gap, k_hh, k_llm = calculate_reliability_gap(h1, h2, llm, metric=metric)
            
            # Build result row
            row = dict(zip(self.group_cols, name))
            row.update({
                'metric_type': metric.capitalize(),
                f'{metric}_gap': gap,
                f'{metric}_hh': k_hh,
                f'{metric}_llm_avg': k_llm
            })
            results.append(row)
            
        return pd.DataFrame(results).sort_values(f'{metric}_gap')

    def analyze_calibration(self, disagreement_col = 'human_disagreement'):
        """
        Correlates Entropy (Uncertainty) with Disagreement (Difficulty).
        """
        # 1. Ensure entropy is calculated
        if 'entropy' not in self.df.columns:
            # Call internal method to calc entropy
            pass 



        # 2. Calculate Correlations per group
        results = []
        for name, group in self.df.groupby(self.group_cols):
            # Calculate score diff
            diff = group[disagreement_col]
            
            # Spearman Corr
            corr = group['entropy'].corr(diff, method='spearman')
            
            row = dict(zip(self.group_cols, name if isinstance(name, tuple) else (name,)))
            row['calibration_corr'] = corr
            results.append(row)
            
        return pd.DataFrame(results)