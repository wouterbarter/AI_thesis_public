from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import statsmodels.api as sm
import warnings


class EconometricFeatureSelector:
    """
    Algorithmic feature selection suite tailored for econometric pipelines.
    Rigidly locks theoretical baseline controls and expanded categorical dummy levels 
    while algorithmically pruning experimental signals using clustered standard errors.
    """

    def __init__(
        self,
        categoricals: List[str],
        continuous: List[str],
        binaries: List[str]
    ):
        self.categoricals = categoricals
        self.continuous = continuous
        self.binaries = binaries

    def _build_protected_list(self, X: pd.DataFrame) -> List[str]:
        """Identifies and shields all baseline control variables from deletion."""
        protected_vars = ['const'] if 'const' in X.columns else []

        for col in X.columns:
            if col in self.continuous or col in self.binaries:
                if col not in protected_vars:
                    protected_vars.append(col)

            for cat in self.categoricals:
                if col.startswith(f"{cat}_") or col.startswith(f"C({cat}"):
                    if col not in protected_vars:
                        protected_vars.append(col)

        return protected_vars

    def _backward_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float,
        protected: List[str],
        groups: pd.Series
    ) -> List[str]:
        """Executes backward elimination using clustered standard errors."""
        features = list(X.columns)
        print(f"[START] Starting Backward Elimination (alpha = {alpha})")
        print(
            f"[INFO] Safely locked {len(protected)} structural baseline levels/features.\n")

        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sm.NegativeBinomial(y, X[features])
                    # Evaluate iterations using the true clustered covariance matrix
                    result = model.fit(
                        method='bfgs',
                        maxiter=500,
                        cov_type='cluster',
                        cov_kwds={'groups': groups},
                        disp=False
                    )
            except Exception as e:
                print(
                    f"[WARNING] Convergence stalled during iteration. Halting. Error: {e}")
                break

            pvalues = result.pvalues
            chopping_block = pvalues.drop(
                labels=[f for f in protected if f in pvalues.index], errors='ignore')

            if chopping_block.empty:
                print(
                    "[SUCCESS] All remaining experimental features are statistically significant.")
                break

            max_p_value = chopping_block.max()
            worst_feature = chopping_block.idxmax()

            if max_p_value > alpha:
                print(
                    f"[-] Dropping '{worst_feature}' (clustered p-value: {max_p_value:.4f})")
                features.remove(worst_feature)
            else:
                print(
                    f"\n[SUCCESS] Optimization complete! Remaining experimental features satisfy alpha = {alpha}.")
                break

        return features

    def _forward_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float,
        protected: List[str],
        groups: pd.Series
    ) -> List[str]:
        """Executes forward step selection using clustered standard errors."""
        current_features = [f for f in protected if f in X.columns]
        candidates = [col for col in X.columns if col not in current_features]

        print(f"[START] Starting Forward Selection (alpha = {alpha})")
        print(
            f"[INFO] Initialized model with {len(current_features)} protected structural control features.")
        print(
            f"[INFO] Evaluating {len(candidates)} experimental candidate dimensions...\n")

        while candidates:
            best_p_value = 1.0
            best_feature = None

            for candidate in candidates:
                trial_features = current_features + [candidate]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = sm.NegativeBinomial(y, X[trial_features])
                        result = model.fit(
                            method='bfgs',
                            maxiter=500,
                            cov_type='cluster',
                            cov_kwds={'groups': groups},
                            disp=False
                        )

                    p_val = result.pvalues[candidate]

                    if p_val < best_p_value:
                        best_p_value = p_val
                        best_feature = candidate
                except Exception:
                    continue

            if best_feature and best_p_value < alpha:
                print(
                    f"[+] Adding '{best_feature}' (clustered p-value: {best_p_value:.4f})")
                current_features.append(best_feature)
                candidates.remove(best_feature)
            else:
                print(
                    f"\n[SUCCESS] Optimization complete! No remaining candidate features satisfy alpha = {alpha}.")
                break

        return current_features

    def fit(
        self,
        runner: Any,
        target_model_label: str,
        method: str = 'backward',
        alpha: float = 0.05
    ) -> Tuple[Any, List[str]]:
        """Extracts runner design matrices, runs selection, and fits the final clustered model."""
        target_rec = next(
            (rec for rec in runner.records if rec.label == target_model_label), None)

        if target_rec is None or target_rec.X is None or target_rec.y is None:
            raise ValueError(
                f"Could not extract a valid design matrix for spec label: '{target_model_label}'")

        X_full = target_rec.X
        y_full = target_rec.y

        # Directly leverage your clean, pre-aligned clusters Series
        if target_rec.clusters is None:
            raise ValueError(
                f"No cluster group vector found in the record for '{target_model_label}'.")
        groups = target_rec.clusters

        protected_features = self._build_protected_list(X_full)

        if method.lower() == 'backward':
            optimal_features = self._backward_elimination(
                X_full, y_full, alpha, protected_features, groups)
        elif method.lower() == 'forward':
            optimal_features = self._forward_selection(
                X_full, y_full, alpha, protected_features, groups)
        else:
            raise ValueError(
                f"Selection strategy '{method}' not recognized. Use 'forward' or 'backward'.")

        print("\n" + "="*60)
        print(
            f"FINAL OPTIMIZED MODEL: {target_model_label} ({method.upper()})")
        print("="*60)

        final_model = sm.NegativeBinomial(y_full, X_full[optimal_features])

        final_result = final_model.fit(
            method='bfgs',
            maxiter=500,
            cov_type='cluster',
            cov_kwds={'groups': groups},
            disp=False
        )

        return final_result, optimal_features
