import pandas as pd
import ipywidgets as widgets
from scipy import stats
from IPython.display import display, clear_output, HTML
from typing import Dict, Any, Optional

class ExperimentRegistry:
    def __init__(self, results: Dict[Any, Any], prompt_map: Dict[str, str]):
        self.results = results
        self.prompt_map = prompt_map
        self._df_summary = None

    def _resolve_name(self, key: Any) -> str:
        """
        Simple lookup: stringifies the key and replaces the hash with the name.
        """
        # 1. Convert key to string (handles both tuples and stringified tuples)
        name = str(key)
        
        # 2. Find the prompt ID and swap it for the readable name
        for pid, label in self.prompt_map.items():
            if pid in name:
                name = name.replace(pid, label)
        
        # 3. (Optional) Strip Python tuple syntax for cleaner reading
        # Converts "('Qwen', 'Naive')" -> "Qwen | Naive"
        for char in ["('", "')", "('", "')"]: 
            name = name.replace(char, "")
        name = name.replace("', '", " | ").replace("', \"", " | ").replace("\", '", " | ")
            
        return name

    @property
    def summary(self) -> pd.DataFrame:
        if self._df_summary is not None:
            return self._df_summary

        rows = []
        for key, res in self.results.items():
            # Safety check for failed models
            if not hasattr(res, 'llf'): continue

            rows.append({
                'key': key,
                'readable_name': self._resolve_name(key),
                'n_obs': int(res.nobs),
                'llf': res.llf,
                'aic': res.aic,
                'converged': res.mle_retvals.get('converged', False) if hasattr(res, 'mle_retvals') else True
            })
        
        self._df_summary = pd.DataFrame(rows).sort_values('llf', ascending=False)
        return self._df_summary

    def llr_test(self, baseline_key: Any, candidate_key: Any, verbose: bool = True) -> Dict[str, Any]:
        """
        Performs Likelihood Ratio Test.
        """
        base_res = self.results.get(baseline_key)
        cand_res = self.results.get(candidate_key)
        
        if not base_res or not cand_res:
            return None # Or raise error
            
        # D = -2 * (LLF_restricted - LLF_unrestricted)
        D = -2 * (base_res.llf - cand_res.llf)
        
        # DF calculation
        df_diff = abs(cand_res.df_model - base_res.df_model)
        p_value = stats.chi2.sf(D, df_diff) if df_diff > 0 else 1.0
        
        # Significance stars
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''

        if verbose:
            print(f"--- Likelihood Ratio Test ---")
            print(f"Baseline:  {self._resolve_name(baseline_key)}")
            print(f"Candidate: {self._resolve_name(candidate_key)}")
            print(f"LR Statistic: {D:.4f} | P-Value: {p_value:.6e} {sig}")
            print(f"{'-'*30}\n")
            
        return {
            'D': D, 
            'p_value': p_value, 
            'df_diff': df_diff,
            'll_base': base_res.llf,
            'll_cand': cand_res.llf,
            'significance': sig
        }


    def run_all_pairwise_contrasts(self) -> pd.DataFrame:
        """
        Generates all possible pairwise comparisons within each Model Family.
        Automatically treats the lower-LLF model as the Baseline to ensure 
        valid Chi2 statistics.
        """
        from itertools import combinations
        
        # 1. Group keys by Model Family
        grouped_models = {}
        for key in self.results.keys():
            # Parse key to find Model Family
            if isinstance(key, str) and key.startswith('('):
                import ast
                real_key = ast.literal_eval(key)
            else:
                real_key = key
            
            model_name = real_key[0]
            if model_name not in grouped_models:
                grouped_models[model_name] = []
            grouped_models[model_name].append(key)

        rows = []

        # 2. Iterate over each family
        for model_name, keys in grouped_models.items():
            # Generate every unique pair of prompts for this model
            # e.g. (A, B), (A, C), (B, C)...
            for k1, k2 in combinations(keys, 2):
                
                res1 = self.results[k1]
                res2 = self.results[k2]
                
                # 3. Orient the pair: Lower LLF is Baseline, Higher is Candidate
                # This ensures D is positive.
                if res1.llf < res2.llf:
                    base_key, cand_key = k1, k2
                    base_res, cand_res = res1, res2
                else:
                    base_key, cand_key = k2, k1
                    base_res, cand_res = res2, res1

                # 4. Run Test
                stats_res = self.llr_test(base_key, cand_key, verbose=False)
                
                rows.append({
                    'Model Family': model_name,
                    'Worse Model': self._resolve_name(base_key),
                    'Better Model': self._resolve_name(cand_key),
                    'LL_Diff': stats_res['ll_cand'] - stats_res['ll_base'], # Always positive
                    'LR Statistic': stats_res['D'],
                    'p-value': stats_res['p_value'],
                    'Significance': stats_res['significance']
                })

        # Return sorted by the biggest wins
        return pd.DataFrame(rows).sort_values(['Model Family', 'LR Statistic'], ascending=[True, False])




# --- Viewer Function ---

def interactive_model_viewer(registry: ExperimentRegistry):
    """
    Dropdown selector to view model results.
    """
    # Create options list: [("Readable Name", raw_key), ...]
    options = [
        (f"{row.readable_name} [LLF: {row.llf:.2f}]", row.key) 
        for row in registry.summary.itertuples()
    ]

    dropdown = widgets.Dropdown(
        options=options,
        value=options[0][1] if options else None,
        description="Select Model:",
        layout=widgets.Layout(width='600px')
    )
    
    output = widgets.Output()

    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            with output:
                clear_output(wait=True)
                key = change['new']
                model = registry.results.get(key)
                
                print(f"Model: {registry._resolve_name(key)}")
                # Force HTML rendering to avoid LaTeX errors
                display(HTML(model.summary().as_html()))

    dropdown.observe(on_change)
    display(dropdown, output)
    
    # Trigger initial view
    if options:
        with output:
            key = options[0][1]
            print(f"Model: {registry._resolve_name(key)}")
            display(HTML(registry.results[key].summary().as_html()))