import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults, GLMResultsWrapper
from statsmodels.discrete.discrete_model import NegativeBinomialResultsWrapper, CountResultsWrapper
from typing import List, Dict, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod


class BaseRegressionRunner(ABC):
    """Base class for regression analysis with robust error handling and alpha estimation."""

    def __init__(self, target_col: str, experimental_groups: List[str], offset_col: Optional[str] = None):
        self.target_col = target_col
        self.experimental_groups = experimental_groups
        # Column containing pre-computed log(offset)
        self.offset_col = offset_col
        self.results = {}

    def _pivot_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        # The LLM metric to pivot (None for baseline)
        score_col: Optional[str],
        extra_index_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Pivot data from long to wide format while maintaining observation integrity."""

        # For baseline model, skip pivoting since there's no LLM score
        if score_col is None:
            return group_df.copy(), []

        unique_dims = group_df['dimension_name'].unique()

        index_cols = (
            self.experimental_groups +
            cat_predictors +
            bin_predictors +
            num_predictors +
            [self.target_col, 'deal_text']
        )

        # Include offset column in index if it exists
        if self.offset_col:
            index_cols.append(self.offset_col)

        if extra_index_cols:
            index_cols.extend(extra_index_cols)

        # Ensure we don't have duplicates before pivoting
        if len(unique_dims) == 1 and unique_dims[0] == 'quality':
            wide_df = group_df.copy()
            score_cols = [score_col]
        else:
            # Use aggfunc='first' to ensure we don't accidentally mean-average descriptors
            wide_df = group_df.pivot_table(
                index=index_cols,
                columns='dimension_name',
                values=score_col,
                aggfunc='first'
            ).reset_index()
            score_cols = list(unique_dims)

        return wide_df, score_cols

    def _build_predictor_matrix(
        self,
        wide_df: pd.DataFrame,
        score_cols: List[str],
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str]
    ) -> Optional[pd.DataFrame]:
        """Build the X matrix and check for mathematical validity (rank)."""
        predictor_parts = []

        # Numeric columns: LLM scores + binary predictors + continuous numerical predictors
        numeric_cols = score_cols + bin_predictors + num_predictors
        if numeric_cols:
            X_num = wide_df[numeric_cols].copy()
            # Convert binary predictors to int
            for col in bin_predictors:
                X_num[col] = X_num[col].astype(int)
            predictor_parts.append(X_num)

        # Categorical dummy columns
        if cat_predictors:
            cat_subset = wide_df[cat_predictors].copy()
            cat_subset = cat_subset.astype(str)

            X_cat = pd.get_dummies(cat_subset, drop_first=True, dtype=int)
            predictor_parts.append(X_cat)

        if not predictor_parts:
            return None

        X = pd.concat(predictor_parts, axis=1)
        X = sm.add_constant(X)

        # Rank check: Ensure no perfect multicollinearity
        if np.linalg.matrix_rank(X.values) < X.shape[1]:
            print(
                "⚠️ Warning: Design matrix is rank-deficient (perfect multicollinearity)")
            return None

        return X

    @abstractmethod
    def prepare_group_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        new_predictor: Optional[str],  # None for baseline
        dims_to_exclude: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare data for regression. Implementation varies by subclass.

        Returns:
            Dictionary containing at minimum (y, X, offset), with optional additional elements
            Returns None if data preparation fails
        """
        pass

    @abstractmethod
    def _fit_model(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        offset: Optional[pd.Series] = None,
        **kwargs
    ) -> Union[NegativeBinomialResultsWrapper, CountResultsWrapper]:
        """Fit the GLM model. Implementation varies by subclass.
        Args:
            y: Target variable
            X: Predictor matrix
            offset: Offset term (or None)
            **kwargs: Additional arguments (e.g., clusters for clustered errors)
        """
        pass

    def run_baseline_model(
        self,
        df: pd.DataFrame,
        cat_vars: List[str],
        bin_vars: List[str],
        num_vars: List[str],
        label: str = "baseline"
    ) -> Dict[str, CountResultsWrapper]:
        """
        Fit a single baseline model without LLM predictors.

        This uses the original input data (og_df) which does NOT have the experimental
        structure (no model types, no prompt_ids). Fits one model with only the 
        control variables (categorical, binary, numerical) but NO LLM scores.

        Args:
            df: Original input dataframe (og_df - all data, no LLM evaluations)
            cat_vars: Categorical control variables
            bin_vars: Binary control variables  
            num_vars: Numerical control variables
            label: Label for the baseline model (default: "baseline")

        Returns:
            Dictionary with single baseline model result
        """
        # Prepare data WITHOUT any LLM predictor (new_predictor=None)
        # No grouping - this is a single model on all original data
        data_dict = self.prepare_group_data(
            df, cat_vars, bin_vars, num_vars,
            new_predictor=None,  # Signal this is baseline
            dims_to_exclude=None
        )

        # print(data_dict)

        if data_dict is None:
            print("Baseline model: Data preparation failed.")
            return self.results

        y = data_dict.get('y')
        X = data_dict.get('X')

        if y is None or X is None:
            print("Baseline model: Missing y or X data.")
            return self.results

        if len(y) < (X.shape[1] + 2):
            print("Baseline model: Insufficient observations.")
            return self.results

        offset = data_dict.get('offset')
        clusters = data_dict.get('clusters')

        try:
            res = self._fit_model(
                y=y,
                X=X,
                offset=offset,
                clusters=clusters
            )

            if not res.converged:
                print(f"⚠️ Warning: Baseline model did not converge.")

            self.results[label] = res
            print(
                f"✓ Baseline model fitted: {res.nobs} observations, AIC={res.aic:.2f}")

        except Exception as e:
            print(f"Error fitting baseline model: {e}")

        return self.results

    def run_negative_binomial(
        self,
        df: pd.DataFrame,
        cat_vars: List[str],
        bin_vars: List[str],
        num_vars: List[str],
        predictor_name: str,
        label_map: Optional[Dict[str, str]] = None,
        dims_to_exclude: Optional[List[str]] = None
    ) -> Dict[str, CountResultsWrapper]:
        """Main execution loop with reliability checks for LLM-based models."""
        grouped = df.groupby(self.experimental_groups)

        for group_keys, group_df in grouped:
            data_dict = self.prepare_group_data(
                group_df, cat_vars, bin_vars, num_vars, predictor_name, dims_to_exclude
            )

            if data_dict is None:
                continue

            y = data_dict.get('y')
            X = data_dict.get('X')

            if y is None or X is None:
                print(f"Skipping {group_keys}: Missing y or X data.")
                continue

            if X is None or len(y) < (X.shape[1] + 2):
                print(f"Skipping {group_keys}: Insufficient observations.")
                continue

            offset = data_dict.get('offset')
            clusters = data_dict.get('clusters')

            try:
                res = self._fit_model(
                    y=y,
                    X=X,
                    offset=offset,
                    clusters=clusters
                )

                if not res.converged:
                    print(
                        f"⚠️ Warning: Model for {group_keys} did not converge.")

                model_label = f"{group_keys[0]}_{label_map.get(group_keys[1], group_keys[1])}" if label_map else str(
                    group_keys)
                self.results[model_label] = res

            except Exception as e:
                print(f"Error fitting model for {group_keys}: {e}")

        return self.results


class StandardErrorRegression(BaseRegressionRunner):
    def prepare_group_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        new_predictor: Optional[str],  # None for baseline
        dims_to_exclude: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare data for standard error regression (No clusters)."""

        # 1. Pivot & Build Initial Matrices
        wide_df, score_cols = self._pivot_data(
            group_df, cat_predictors, bin_predictors, num_predictors, new_predictor
        )

        if dims_to_exclude:
            score_cols = [
                col for col in score_cols if col not in dims_to_exclude]

        X = self._build_predictor_matrix(
            wide_df, score_cols, cat_predictors, bin_predictors, num_predictors)

        if X is None:
            return None

        y = wide_df[self.target_col]

        # 2. Build Component List Dynamically
        data_components = [y, X]

        # Add offset ONLY if it is defined and exists in the data
        if self.offset_col and self.offset_col in wide_df.columns:
            data_components.append(wide_df[self.offset_col])

        # 3. Single Pipeline: Concat -> Align -> DropNA
        combined = pd.concat(data_components, axis=1).dropna()

        if combined.empty:
            return None

        # 4. Return Dictionary
        return {
            'y': combined[self.target_col],
            'X': combined[X.columns],
            'offset': combined[self.offset_col] if self.offset_col else None,
            'clusters': None
        }

    def _fit_model(self, y: pd.Series, X: pd.DataFrame, offset: Optional[pd.Series] = None, **kwargs) -> Union[NegativeBinomialResultsWrapper, CountResultsWrapper]:
        """Fit GLM with standard errors."""
        model = sm.NegativeBinomial(y, X, loglike_method='nb2', offset=offset)
        return model.fit(maxiter=2000, method='bfgs', disp=0)


class ClusteredErrorRegression(BaseRegressionRunner):
    def __init__(self, target_col: str, experimental_groups: List[str], cluster_col: str, offset_col: Optional[str] = None):
        super().__init__(target_col, experimental_groups, offset_col)
        self.cluster_col = cluster_col

    def prepare_group_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        new_predictor: Optional[str],  # None for baseline
        dims_to_exclude: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare data for clustered error regression."""

        # 1. Pivot & Build Initial Matrices
        wide_df, score_cols = self._pivot_data(
            group_df, cat_predictors, bin_predictors, num_predictors,
            new_predictor, extra_index_cols=[self.cluster_col]
        )

        if dims_to_exclude:
            score_cols = [
                col for col in score_cols if col not in dims_to_exclude]

        X = self._build_predictor_matrix(
            wide_df, score_cols, cat_predictors, bin_predictors, num_predictors)
        if X is None:
            return None

        y = wide_df[self.target_col]
        clusters = wide_df[self.cluster_col]

        # 2. Build Component List Dynamically
        data_components = [y, X, clusters]

        # Add offset ONLY if it exists
        if self.offset_col and self.offset_col in wide_df.columns:
            data_components.append(wide_df[self.offset_col])

        # 3. Single Pipeline: Concat -> Align -> DropNA
        combined = pd.concat(data_components, axis=1).dropna()

        if combined.empty:
            return None

        # 4. Return Dictionary (Clean Extraction)
        return {
            'y': combined[self.target_col],
            'X': combined[X.columns],
            'clusters': combined[self.cluster_col],
            'offset': combined[self.offset_col] if self.offset_col else None
        }

    def _fit_model(self, y, X, offset=None, **kwargs) -> Union[NegativeBinomialResultsWrapper, CountResultsWrapper]:
        """Fit GLM with clustered standard errors."""
        clusters = kwargs.get('clusters')

        if clusters is None:
            raise ValueError(
                "ClusteredErrorRegression requires 'clusters' argument.")

        try:
            model = sm.NegativeBinomial(
                y, X, loglike_method='nb2', offset=offset)

            # Use 'cluster' covariance
            return model.fit(
                cov_type='cluster',
                cov_kwds={'groups': clusters},
                maxiter=2000,
                disp=0,
                use_t=True
            )
        except Exception as e:
            print(f"MLE Fit failed: {e}")
            raise e


def create_regression_runner(
    target_col: str,
    experimental_groups: List[str],
    cluster_col: Optional[str] = None,
    offset_col: Optional[str] = None
) -> BaseRegressionRunner:
    """Factory function to create the appropriate regression runner.

    Args:
        target_col: Name of the dependent variable (count outcome)
        experimental_groups: List of columns to group by
        cluster_col: Column name for clustering (if using clustered SEs)
        offset_col: Column name containing pre-computed log(offset) values
                   This should be log-transformed BEFORE passing to the runner
    """
    if cluster_col:
        return ClusteredErrorRegression(target_col, experimental_groups, cluster_col, offset_col)
    return StandardErrorRegression(target_col, experimental_groups, offset_col)


# =============================================================================
# USAGE EXAMPLE - BASELINE + LLM MODELS
# =============================================================================
"""
# Scenario: Compare LLM evaluation prompts against a baseline

# Step 1: Prepare your data
# -------------------------
# og_df: Original input data (ALL samples, no LLM evaluations, no experimental structure)
#        Columns: deal_text, num_applications, category, is_premium, month_numeric, partner_id
#        This is just your raw input data before any LLM evaluation
#
# eval_df: Evaluation data (SUBSET with LLM scores, HAS experimental structure)
#          Columns: same as above + model_type, prompt_id, dimension_name, llm_score
#          This has the pivoted structure for different prompts/models

# Add offset to both dataframes
og_df['log_monthly_mean'] = np.log(og_df.groupby('month')['num_applications'].transform('mean'))
eval_df['log_monthly_mean'] = np.log(eval_df.groupby('month')['num_applications'].transform('mean'))

# Step 2: Create runner
# ---------------------
runner = create_regression_runner(
    target_col='num_applications',
    experimental_groups=['model_type', 'prompt_id'],  # For LLM models
    cluster_col='partner_id',
    offset_col='log_monthly_mean'
)

# Step 3: Fit SINGLE baseline model on original data
# --------------------------------------------------
# This fits ONE model using ALL the original data (no experimental splits)
baseline_results = runner.run_baseline_model(
    df=og_df,  # Original data - no model_type, no prompt_id columns
    cat_vars=['category'],
    bin_vars=['is_premium'],
    num_vars=['month_numeric'],
    label='baseline'
)
# Result: runner.results['baseline']

# Step 4: Fit MULTIPLE LLM models on evaluation data
# --------------------------------------------------
# This fits separate models for each (model_type, prompt_id) combination
llm_results = runner.run_negative_binomial(
    df=eval_df,  # Has model_type and prompt_id columns
    cat_vars=['category'],
    bin_vars=['is_premium'],
    num_vars=['month_numeric'],
    predictor_name='llm_score',
    label_map={'prompt_v1': 'holistic', 'prompt_v2': 'detailed'}
)
# Results: runner.results['gpt4_holistic'], runner.results['gpt4_detailed'], 
#          runner.results['claude_holistic'], runner.results['claude_detailed'], etc.

# Step 5: Compare baseline vs LLM models
# --------------------------------------
baseline_aic = runner.results['baseline'].aic
baseline_llf = runner.results['baseline'].llf

print(f"{'Model':<30} {'AIC':<10} {'ΔAIC':<10} {'Pseudo-R²':<10}")
print("="*60)

for model_name, result in sorted(runner.results.items()):
    if model_name == 'baseline':
        pseudo_r2 = np.nan
        delta_aic = 0
    else:
        # McFadden's Pseudo-R²: how much better than baseline
        pseudo_r2 = 1 - (result.llf / baseline_llf)
        delta_aic = result.aic - baseline_aic
    
    print(f"{model_name:<30} {result.aic:<10.2f} {delta_aic:<10.2f} {pseudo_r2:<10.3f}")

# Interpretation:
# - Baseline: Predictive power of controls alone
# - LLM models: Incremental value of LLM evaluation over controls
# - ΔAIC < 0: LLM adds predictive value
# - Pseudo-R² > 0: LLM explains additional variance beyond baseline
"""
