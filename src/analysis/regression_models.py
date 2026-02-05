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
        self.offset_col = offset_col  # Column containing pre-computed log(offset)
        self.results = {}

    def _pivot_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        score_col: str,  # The LLM metric to pivot
        extra_index_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Pivot data from long to wide format while maintaining observation integrity."""
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
            print("⚠️ Warning: Design matrix is rank-deficient (perfect multicollinearity)")
            return None
            
        return X

    # def _estimate_alpha(self, y: pd.Series, X: pd.DataFrame, offset: Optional[pd.Series] = None) -> float:
    #     """Estimate the Negative Binomial dispersion parameter (alpha)."""
    #     try:
    #         # Fit a discrete NB model to estimate dispersion (NB2)
    #         # method='nm' (Nelder-Mead) is robust for dispersion estimation
    #         if offset is not None:
    #             nb_fit = sm.NegativeBinomial(y, X, offset=offset).fit(disp=0, method='nm', maxiter=500)
    #         else:
    #             nb_fit = sm.NegativeBinomial(y, X).fit(disp=0, method='nm', maxiter=500)
    #         return max(1e-9, nb_fit.alpha)  # Ensure alpha is positive
    #     except Exception:
    #         # Fallback to default if estimation fails
    #         return 1.0

    @abstractmethod
    def prepare_group_data(
        self,
        group_df: pd.DataFrame,
        cat_predictors: List[str],
        bin_predictors: List[str],
        num_predictors: List[str],
        new_predictor: str,
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
        """Main execution loop with reliability checks."""
        grouped = df.groupby(self.experimental_groups)
        
        for group_keys, group_df in grouped:
            # prepared_data = self.prepare_group_data(
            #     group_df, cat_vars, bin_vars, num_vars, predictor_name, dims_to_exclude
            # )
            data_dict = self.prepare_group_data(
                group_df, cat_vars, bin_vars, num_vars, predictor_name, dims_to_exclude
            )
            
            if data_dict is None:
                continue
            
            # y, X, offset, clusters = prepared_data

            # 2. Extract essentials for the safety check
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
                    print(f"⚠️ Warning: Model for {group_keys} did not converge.")

                model_label = f"{group_keys[0]}_{label_map.get(group_keys[1], group_keys[1])}" if label_map else str(group_keys)
                self.results[model_label] = res
                
            except Exception as e:
                print(f"Error fitting model for {group_keys}: {e}")
        
        return self.results


class StandardErrorRegression(BaseRegressionRunner):
    # def prepare_group_data(
    #     self, 
    #     group_df: pd.DataFrame, 
    #     cat_predictors: List[str], 
    #     bin_predictors: List[str], 
    #     num_predictors: List[str],
    #     new_predictor: str, 
    #     dims_to_exclude: Optional[List[str]] = None
    # ) -> Optional[Tuple[pd.Series, pd.DataFrame, Optional[pd.Series]]]:
    #     """Prepare data for standard error regression."""
        
    #     wide_df, score_cols = self._pivot_data(
    #         group_df, cat_predictors, bin_predictors, num_predictors, new_predictor
    #     )
        
    #     if dims_to_exclude:
    #         score_cols = [col for col in score_cols if col not in dims_to_exclude]

    #     X = self._build_predictor_matrix(wide_df, score_cols, cat_predictors, bin_predictors, num_predictors)
    #     if X is None:
    #         return None
        
    #     y = wide_df[self.target_col]
        
    #     # Extract offset if specified
    #     offset = wide_df[self.offset_col] if self.offset_col else None
        
    #     # Clean NaNs - include offset in alignment if it exists
    #     if offset is not None:
    #         combined = pd.concat([X, y, offset], axis=1).dropna()
    #         return (
    #             combined[self.target_col], 
    #             combined.drop(columns=[self.target_col, self.offset_col]),
    #             combined[self.offset_col]
    #         )
    #     else:
    #         combined = pd.concat([X, y], axis=1).dropna()
    #         return (
    #             combined[self.target_col], 
    #             combined.drop(columns=[self.target_col]),
    #             None
    #         )

    def prepare_group_data(
        self, 
        group_df: pd.DataFrame, 
        cat_predictors: List[str], 
        bin_predictors: List[str], 
        num_predictors: List[str],
        new_predictor: str, 
        dims_to_exclude: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare data for standard error regression (No clusters)."""
        
        # 1. Pivot & Build Initial Matrices
        wide_df, score_cols = self._pivot_data(
            group_df, cat_predictors, bin_predictors, num_predictors, new_predictor
        )
        
        if dims_to_exclude:
            score_cols = [col for col in score_cols if col not in dims_to_exclude]

        X = self._build_predictor_matrix(wide_df, score_cols, cat_predictors, bin_predictors, num_predictors)
        
        if X is None:
            return None
        
        y = wide_df[self.target_col]
        
        # 2. Build Component List Dynamically
        # Start with the mandatory parts
        data_components = [y, X]
        
        # Add offset ONLY if it is defined and exists in the data
        if self.offset_col and self.offset_col in wide_df.columns:
            data_components.append(wide_df[self.offset_col])

        # 3. Single Pipeline: Concat -> Align -> DropNA
        # This automatically aligns indices and drops rows where ANY component is NaN
        combined = pd.concat(data_components, axis=1).dropna()
        
        if combined.empty:
            return None

        # 4. Return Dictionary
        # Note: We explicitly set 'clusters' to None since this is Standard Regression
        return {
            'y': combined[self.target_col],
            'X': combined[X.columns],  # Safe extraction using original column names
            'offset': combined[self.offset_col] if self.offset_col else None,
            'clusters': None
        }

    def _fit_model(self, y: pd.Series, X: pd.DataFrame, offset: Optional[pd.Series]=None, **kwargs) -> Union[NegativeBinomialResultsWrapper, CountResultsWrapper]:
        """Fit GLM with standard errors."""
        model = sm.NegativeBinomial(y, X, loglike_method='nb2', offset=offset)
        return model.fit(maxiter=2000, method='bfgs', disp=0)


class ClusteredErrorRegression(BaseRegressionRunner):
    def __init__(self, target_col: str, experimental_groups: List[str], cluster_col: str, offset_col: Optional[str] = None):
        super().__init__(target_col, experimental_groups, offset_col)
        self.cluster_col = cluster_col

    # def prepare_group_data(
    #     self, 
    #     group_df: pd.DataFrame, 
    #     cat_predictors: List[str], 
    #     bin_predictors: List[str],
    #     num_predictors: List[str], 
    #     new_predictor: str, 
    #     dims_to_exclude: Optional[List[str]] = None
    # ) -> Optional[Tuple[pd.Series, pd.DataFrame, Optional[pd.Series], pd.Series]]:
    #     """Prepare data for clustered error regression."""
        
    #     wide_df, score_cols = self._pivot_data(
    #         group_df, cat_predictors, bin_predictors, num_predictors,
    #         new_predictor, extra_index_cols=[self.cluster_col]
    #     )
        
    #     if dims_to_exclude:
    #         score_cols = [col for col in score_cols if col not in dims_to_exclude]

    #     X = self._build_predictor_matrix(wide_df, score_cols, cat_predictors, bin_predictors, num_predictors)
    #     if X is None:
    #         return None
        
    #     y = wide_df[self.target_col]
    #     clusters = wide_df[self.cluster_col]
        
    #     # Extract offset if specified
    #     offset = wide_df[self.offset_col] if self.offset_col else None
        
    #     # Logic to handle offset existence
    #     if self.offset_col:
    #         # Case A: We have everything
    #         combined = pd.concat([X, y, offset, clusters], axis=1).dropna()
            
    #         # Return Dict
    #         return {
    #             'y': combined[self.target_col],
    #             'X': combined.drop(columns=[self.target_col, self.cluster_col, self.offset_col]),
    #             'offset': combined[self.offset_col],
    #             'clusters': combined[self.cluster_col] # <--- Passed safely
    #         }
    #     else:
    #         # Case B: No Offset
    #         combined = pd.concat([X, y, clusters], axis=1).dropna()
            
    #         return {
    #             'y': combined[self.target_col],
    #             'X': combined.drop(columns=[self.target_col, self.cluster_col]),
    #             'offset': None,
    #             'clusters': combined[self.cluster_col]
    #         }
        
    def prepare_group_data(
        self, 
        group_df: pd.DataFrame, 
        cat_predictors: List[str], 
        bin_predictors: List[str],
        num_predictors: List[str], 
        new_predictor: str, 
        dims_to_exclude: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        
        # 1. Pivot & Build Initial Matrices
        wide_df, score_cols = self._pivot_data(
            group_df, cat_predictors, bin_predictors, num_predictors,
            new_predictor, extra_index_cols=[self.cluster_col]
        )
        
        if dims_to_exclude:
            score_cols = [col for col in score_cols if col not in dims_to_exclude]

        X = self._build_predictor_matrix(wide_df, score_cols, cat_predictors, bin_predictors, num_predictors)
        if X is None:
            return None
            
        y = wide_df[self.target_col]
        clusters = wide_df[self.cluster_col]
        
        # 2. Build Component List Dynamically
        # Start with the mandatory parts
        data_components = [y, X, clusters]
        
        # Add offset ONLY if it exists
        if self.offset_col and self.offset_col in wide_df.columns:
            data_components.append(wide_df[self.offset_col])

        # 3. Single Pipeline: Concat -> Align -> DropNA
        # This automatically aligns indices and drops rows where ANY component is NaN
        combined = pd.concat(data_components, axis=1).dropna()
        
        if combined.empty:
            return None

        # 4. Return Dictionary (Clean Extraction)
        # We use X.columns to pull the predictor columns back out safely, 
        # avoiding the need to calculate what to 'drop'.
        return {
            'y': combined[self.target_col],
            'X': combined[X.columns], 
            'clusters': combined[self.cluster_col],
            # If offset_col was used, it's in combined; otherwise return None
            'offset': combined[self.offset_col] if self.offset_col else None
        }



    def _fit_model(self, y, X, offset=None, **kwargs) -> Union[NegativeBinomialResultsWrapper, CountResultsWrapper]:
        # Direct MLE estimation of Beta and Alpha
        # We assume X already includes the constant

        clusters = kwargs.get('clusters')

        if clusters is None:
            raise ValueError("ClusteredErrorRegression requires 'clusters' argument.")
        
        try:
            model = sm.NegativeBinomial(y, X, loglike_method='nb2', offset=offset)
            
            # Use 'cluster' covariance
            # use_t=True gives t-stats instead of z-stats (better for finite samples)
            return model.fit(
                cov_type='cluster', 
                cov_kwds={'groups': clusters}, 
                maxiter=2000,
                disp=0,
                use_t=True 
            )
        except Exception as e:
            # Fallback for singular matrix or convergence failure
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
# USAGE EXAMPLE
# =============================================================================
"""
# Option 1: Using offset (coefficient fixed at 1.0)
# -------------------------------------------------
# Appropriate when you want to control for exposure/baseline rate
# but don't want to estimate its effect

# Data prep: Create log-transformed offset
df['log_monthly_mean'] = np.log(df.groupby('month')['num_applications'].transform('mean'))

# Create runner with offset
runner = create_regression_runner(
    target_col='num_applications',
    experimental_groups=['condition', 'model'],
    offset_col='log_monthly_mean'  # Pre-computed log(monthly mean)
)

# Run regression
results = runner.run_negative_binomial(
    df=df,
    cat_vars=['category'],
    bin_vars=['is_premium'],
    num_vars=[],  # No additional numerical predictors
    predictor_name='llm_score'
)

# Interpretation: 
# Coefficients represent the effect on the RATE (per unit of baseline)
# E.g., β=0.5 means a 1-unit increase in X increases the rate by exp(0.5)=1.65x


# Option 2: Including as a predictor (coefficient estimated)
# -----------------------------------------------------------
# Appropriate when you want to estimate whether/how much time matters

# Data prep: Create log-transformed predictor
df['log_monthly_mean'] = np.log(df.groupby('month')['num_applications'].transform('mean'))

# Create runner WITHOUT offset
runner = create_regression_runner(
    target_col='num_applications',
    experimental_groups=['condition', 'model']
)

# Run regression with time as a numerical predictor
results = runner.run_negative_binomial(
    df=df,
    cat_vars=['category'],
    bin_vars=['is_premium'],
    num_vars=['log_monthly_mean'],  # Time effect as predictor
    predictor_name='llm_score'
)

# Interpretation:
# The coefficient on log_monthly_mean tells you how sensitive applications
# are to the baseline time trend (might be <1, =1, or >1)


# Option 3: Both offset AND additional time controls
# ---------------------------------------------------
# Use offset for baseline rate, add time as predictor for deviations

df['log_monthly_mean'] = np.log(df.groupby('month')['num_applications'].transform('mean'))
df['month_numeric'] = (df['month'] - df['month'].min()).dt.days / 30  # Months since start

runner = create_regression_runner(
    target_col='num_applications',
    experimental_groups=['condition', 'model'],
    offset_col='log_monthly_mean'
)

results = runner.run_negative_binomial(
    df=df,
    cat_vars=['category'],
    bin_vars=['is_premium'],
    num_vars=['month_numeric'],  # Linear time trend as additional control
    predictor_name='llm_score'
)
"""







# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from statsmodels.genmod.generalized_linear_model import GLMResults, GLMResultsWrapper
# from typing import List, Dict, Optional, Tuple, Any
# from abc import ABC, abstractmethod

# class BaseRegressionRunner(ABC):
#     """Base class for regression analysis with robust error handling and alpha estimation."""
    
#     def __init__(self, target_col: str, experimental_groups: List[str]):
#         self.target_col = target_col
#         self.experimental_groups = experimental_groups
#         self.results = {}

#     def _pivot_data(
#         self,
#         group_df: pd.DataFrame,
#         cat_predictors: List[str],
#         bin_predictors: List[str],
#         numerical_predictors: List[str],
#         score_col: str, # The LLM metric to pivot
#         extra_index_cols: List[str] = None
#     ) -> Tuple[pd.DataFrame, List[str]]:
#         """Pivot data from long to wide format while maintaining observation integrity."""
#         unique_dims = group_df['dimension_name'].unique()
        
#         index_cols = (
#             self.experimental_groups + 
#             cat_predictors + 
#             bin_predictors + 
#             [self.target_col, 'deal_text']
#         )
#         if extra_index_cols:
#             index_cols.extend(extra_index_cols)

        
#         # Ensure we don't have duplicates before pivoting
#         if len(unique_dims) == 1 and unique_dims[0] == 'quality':
#             wide_df = group_df.copy()
#             score_cols = [score_col]
#         else:
#             # Use aggfunc='first' to ensure we don't accidentally mean-average descriptors
#             wide_df = group_df.pivot_table(
#                 index=index_cols,
#                 columns='dimension_name',
#                 values=score_col,
#                 aggfunc='first'
#             ).reset_index()
#             score_cols = list(unique_dims)
        
#         return wide_df, score_cols

#     def _build_predictor_matrix(
#         self,
#         wide_df: pd.DataFrame,
#         score_cols: List[str],
#         cat_predictors: List[str],
#         bin_predictors: List[str]
#     ) -> Optional[pd.DataFrame]:
#         """Build the X matrix and check for mathematical validity (rank)."""
#         predictor_parts = []
        
#         numeric_cols = score_cols + bin_predictors
#         if numeric_cols:
#             X_num = wide_df[numeric_cols].copy()
#             for col in bin_predictors:
#                 X_num[col] = X_num[col].astype(int)
#             predictor_parts.append(X_num)
        
#         if cat_predictors:
#             X_cat = pd.get_dummies(wide_df[cat_predictors], drop_first=True, dtype=int)
#             predictor_parts.append(X_cat)
        
#         if not predictor_parts:
#             return None
        
#         X = pd.concat(predictor_parts, axis=1)
#         X = sm.add_constant(X)

#         # Rank check: Ensure no perfect multicollinearity
#         if np.linalg.matrix_rank(X.values) < X.shape[1]:
#             return None
            
#         return X

#     def _estimate_alpha(self, y: pd.Series, X: pd.DataFrame) -> float:
#         """Estimate the Negative Binomial dispersion parameter (alpha)."""
#         try:
#             # Fit a discrete NB model to estimate dispersion (NB2)
#             # method='nm' (Nelder-Mead) is robust for dispersion estimation
#             nb_fit = sm.NegativeBinomial(y, X).fit(disp=0, method='nm', maxiter=500)
#             return max(1e-9, nb_fit.alpha) # Ensure alpha is positive
#         except Exception:
#             # Fallback to default if estimation fails
#             return 1.0

#     @abstractmethod
#     def prepare_group_data(
#         self,
#         group_df: pd.DataFrame,
#         cat_predictors: List[str],
#         bin_predictors: List[str],
#         new_predictor: str
#     ) -> Optional[Tuple]:
#         """Prepare data for regression. Implementation varies by subclass.
        
#         Returns:
#             Tuple containing at minimum (y, X), with optional additional elements
#             Returns None if data preparation fails
#         """
#         pass

#     @abstractmethod
#     def _fit_model(self, y: pd.Series, X: pd.DataFrame, *args) -> GLMResultsWrapper:
#         """Fit the GLM model. Implementation varies by subclass.
        
#         Args:
#             y: Target variable
#             X: Predictor matrix
#             *args: Additional arguments (e.g., clusters for clustered errors)
#         """
#         pass

#     def run_negative_binomial(
#         self,
#         df: pd.DataFrame,
#         cat_vars: List[str],
#         bin_vars: List[str],
#         predictor_name: str,
#         label_map: Optional[Dict[str, str]] = None,
#         dims_to_exclude: Optional[List[str]] = None
#     ) -> Dict[str, GLMResults]:
#         """Main execution loop with reliability checks."""
#         grouped = df.groupby(self.experimental_groups)
        
#         for group_keys, group_df in grouped:
#             prepared_data = self.prepare_group_data(
#                 group_df, cat_vars, bin_vars, predictor_name, dims_to_exclude
#             )
            
#             if prepared_data is None:
#                 continue
            
#             y, X = prepared_data[:2]
#             extra_args = prepared_data[2:] if len(prepared_data) > 2 else []

#             if X is None or len(y) < (X.shape[1] + 2):
#                 print(f"Skipping {group_keys}: Insufficient observations or singular matrix.")
#                 continue

#             try:
#                 res = self._fit_model(y, X, *extra_args)
                
#                 if not res.converged:
#                     print(f"⚠️ Warning: Model for {group_keys} did not converge.")

#                 model_label = f"{group_keys[0]}_{label_map.get(group_keys[1], group_keys[1])}" if label_map else str(group_keys)
#                 self.results[model_label] = res
                
#             except Exception as e:
#                 print(f"Error fitting model for {group_keys}: {e}")
        
#         return self.results


# class StandardErrorRegression(BaseRegressionRunner):
#     def prepare_group_data(self, group_df, cat_predictors, bin_predictors, new_predictor, dims_to_exclude: Optional[list[str]] = None):
#         wide_df, score_cols = self._pivot_data(group_df, cat_predictors, bin_predictors, new_predictor)
#         if dims_to_exclude:
#             score_cols = [col for col in score_cols if col not in dims_to_exclude]

#         X = self._build_predictor_matrix(wide_df, score_cols, cat_predictors, bin_predictors)
#         if X is None: return None
        
#         y = wide_df[self.target_col]
#         combined = pd.concat([X, y], axis=1).dropna()
#         return combined[self.target_col], combined.drop(columns=[self.target_col])

#     # def _fit_model(self, y, X):
#     #     alpha = self._estimate_alpha(y, X)
#     #     return sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit()

#     def _fit_model(self, y: pd.Series, X: pd.DataFrame, *args) -> GLMResultsWrapper:
#         """Fit GLM with standard errors."""
#         glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
#         return glm.fit()


# class ClusteredErrorRegression(BaseRegressionRunner):
#     def __init__(self, target_col: str, experimental_groups: List[str], cluster_col: str):
#         super().__init__(target_col, experimental_groups)
#         self.cluster_col = cluster_col

#     def prepare_group_data(self, group_df, cat_predictors, bin_predictors, new_predictor, dims_to_exclude: Optional[List[str]] = None):
#         wide_df, score_cols = self._pivot_data(group_df, cat_predictors, bin_predictors, 
#                                               new_predictor, extra_index_cols=[self.cluster_col])
        
#         if dims_to_exclude:
#             score_cols = [col for col in score_cols if col not in dims_to_exclude]

#         X = self._build_predictor_matrix(wide_df, score_cols, cat_predictors, bin_predictors)

#         if X is None: return None
        
#         y = wide_df[self.target_col]
#         clusters = wide_df[self.cluster_col]
        
#         combined = pd.concat([X, y, clusters], axis=1).dropna()
#         return combined[self.target_col], combined.drop(columns=[self.target_col, self.cluster_col]), combined[self.cluster_col]

#     # def _fit_model(self, y, X, clusters):
#     #     alpha = self._estimate_alpha(y, X)
#     #     glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha))
#     #     # Ensure maxiter is higher for clustered models
#     #     return glm.fit(cov_type='cluster', cov_kwds={'groups': clusters}, maxiter=200)


#     def _fit_model(self, y: pd.Series, X: pd.DataFrame, *args) -> GLMResultsWrapper:
#         """Fit GLM with clustered standard errors."""
#         clusters = args[0]  # Extract clusters from args
#         glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
#         return glm.fit(cov_type='cluster', cov_kwds={'groups': clusters})







# def create_regression_runner(target_col, experimental_groups, cluster_col=None):
#     if cluster_col:
#         return ClusteredErrorRegression(target_col, experimental_groups, cluster_col)
#     return StandardErrorRegression(target_col, experimental_groups)



# # import pandas as pd
# # import statsmodels.api as sm
# # from statsmodels.genmod.generalized_linear_model import GLMResults
# # from typing import List, Dict, Optional, Tuple
# # from abc import ABC, abstractmethod


# # class BaseRegressionRunner(ABC):
# #     """Base class for regression analysis."""
    
# #     def __init__(self, target_col: str, experimental_groups: List[str]):
# #         self.target_col = target_col
# #         self.experimental_groups = experimental_groups
# #         self.results = {}

# #     def _pivot_data(
# #         self,
# #         group_df: pd.DataFrame,
# #         cat_predictors: List[str],
# #         bin_predictors: List[str],
# #         new_predictor: str,
# #         extra_index_cols: List[str] = None
# #     ) -> Tuple[pd.DataFrame, List[str]]:
# #         """Pivot data from long to wide format if needed.
        
# #         Returns:
# #             wide_df: Pivoted dataframe
# #             score_cols: List of score column names
# #         """
# #         unique_dims = group_df['dimension_name'].unique()
        
# #         # Build index columns for pivot
# #         index_cols = (
# #             self.experimental_groups + 
# #             cat_predictors + 
# #             bin_predictors + 
# #             [self.target_col, 'input_id']
# #         )
# #         if extra_index_cols:
# #             index_cols.extend(extra_index_cols)
        
# #         if len(unique_dims) == 1 and unique_dims[0] == 'quality':
# #             # Holistic: Data is already 'wide' enough
# #             wide_df = group_df.copy()
# #             score_cols = [new_predictor]
# #         else:
# #             # Formative: Pivot dimensions into separate columns
# #             wide_df = group_df.pivot_table(
# #                 index=index_cols,
# #                 columns='dimension_name',
# #                 values=new_predictor
# #             ).reset_index()
# #             score_cols = list(unique_dims)
        
# #         return wide_df, score_cols

# #     def _build_predictor_matrix(
# #         self,
# #         wide_df: pd.DataFrame,
# #         score_cols: List[str],
# #         cat_predictors: List[str],
# #         bin_predictors: List[str]
# #     ) -> pd.DataFrame:
# #         """Build the X matrix from predictors."""
# #         predictor_parts = []
        
# #         # Numeric/Score/Binary columns
# #         numeric_cols = score_cols + bin_predictors
# #         if numeric_cols:
# #             X_num = wide_df[numeric_cols].copy()
# #             for col in bin_predictors:
# #                 X_num[col] = X_num[col].astype(int)
# #             predictor_parts.append(X_num)
        
# #         # Categorical Dummy columns
# #         if cat_predictors:
# #             X_cat = pd.get_dummies(wide_df[cat_predictors], drop_first=True, dtype=int)
# #             predictor_parts.append(X_cat)
        
# #         if not predictor_parts:
# #             return None
        
# #         X = pd.concat(predictor_parts, axis=1)
# #         return X

# #     @abstractmethod
# #     def prepare_group_data(
# #         self,
# #         group_df: pd.DataFrame,
# #         cat_predictors: List[str],
# #         bin_predictors: List[str],
# #         new_predictor: str
# #     ):
# #         """Prepare data for regression. Implementation varies by subclass."""
# #         pass

# #     @abstractmethod
# #     def _fit_model(self, y: pd.Series, X: pd.DataFrame):
# #         """Fit the GLM model. Implementation varies by subclass."""
# #         pass

# #     def run_negative_binomial(
# #         self,
# #         df: pd.DataFrame,
# #         cat_vars: List[str],
# #         bin_vars: List[str],
# #         predictor_name: str,
# #         label_map: Optional[Dict[str, str]] = None
# #     ) -> Dict[str, GLMResults]:
# #         """Groups data and fits a Negative Binomial GLM to each group."""
        
# #         grouped = df.groupby(self.experimental_groups)
        
# #         for group_keys, group_df in grouped:
# #             # Prepare data (method varies by subclass)
# #             prepared_data = self.prepare_group_data(
# #                 group_df, cat_vars, bin_vars, predictor_name
# #             )
            
# #             if prepared_data is None:
# #                 print(f"Skipping {group_keys}: Data preparation failed.")
# #                 continue
            
# #             y, X = prepared_data[:2]  # First two are always y and X
            
# #             # Check if we have sufficient data
# #             if X is None or len(y) < (X.shape[1] + 2):
# #                 print(f"Skipping {group_keys}: Insufficient data.")
# #                 continue

# #             try:
# #                 # Fit model (method varies by subclass)
# #                 res = self._fit_model(y, X, *prepared_data[2:])  # Pass any additional data
                
# #                 # Create a readable name for the results dictionary
# #                 if label_map and len(group_keys) > 1 and group_keys[1] in label_map:
# #                     model_label = f"{group_keys[0]}_{label_map[group_keys[1]]}"
# #                 else:
# #                     model_label = str(group_keys)
                
# #                 self.results[model_label] = res
                
# #             except Exception as e:
# #                 print(f"Error fitting model for {group_keys}: {e}")
        
# #         return self.results


# # class StandardErrorRegression(BaseRegressionRunner):
# #     """Regression with standard errors."""
    
# #     def prepare_group_data(
# #         self,
# #         group_df: pd.DataFrame,
# #         cat_predictors: List[str],
# #         bin_predictors: List[str],
# #         new_predictor: str
# #     ) -> Optional[Tuple[pd.Series, pd.DataFrame]]:
# #         """Prepare data for standard error regression."""
        
# #         # Pivot data
# #         wide_df, score_cols = self._pivot_data(
# #             group_df, cat_predictors, bin_predictors, new_predictor
# #         )
        
# #         # Build predictor matrix
# #         X = self._build_predictor_matrix(wide_df, score_cols, cat_predictors, bin_predictors)
# #         if X is None:
# #             return None
        
# #         y = wide_df[self.target_col]
        
# #         # Clean NaNs
# #         combined = pd.concat([X, y], axis=1)
# #         valid_idx = combined.dropna().index
        
# #         X_clean = X.loc[valid_idx]
# #         y_clean = y.loc[valid_idx]
        
# #         # Add constant
# #         X_clean = sm.add_constant(X_clean)
        
# #         return y_clean, X_clean

# #     def _fit_model(self, y: pd.Series, X: pd.DataFrame) -> GLMResults:
# #         """Fit GLM with standard errors."""
# #         glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
# #         return glm.fit()


# # class ClusteredErrorRegression(BaseRegressionRunner):
# #     """Regression with clustered standard errors."""
    
# #     def __init__(self, target_col: str, experimental_groups: List[str], cluster_col: str):
# #         super().__init__(target_col, experimental_groups)
# #         self.cluster_col = cluster_col

# #     def prepare_group_data(
# #         self,
# #         group_df: pd.DataFrame,
# #         cat_predictors: List[str],
# #         bin_predictors: List[str],
# #         new_predictor: str
# #     ) -> Optional[Tuple[pd.Series, pd.DataFrame, pd.Series]]:
# #         """Prepare data for clustered error regression."""
        
# #         # Pivot data (include cluster column in index)
# #         wide_df, score_cols = self._pivot_data(
# #             group_df, cat_predictors, bin_predictors, new_predictor,
# #             extra_index_cols=[self.cluster_col]
# #         )
        
# #         # Build predictor matrix
# #         X = self._build_predictor_matrix(wide_df, score_cols, cat_predictors, bin_predictors)
# #         if X is None:
# #             return None
        
# #         y = wide_df[self.target_col]
# #         clusters = wide_df[self.cluster_col]
        
# #         # Clean NaNs - align all variables
# #         combined = pd.concat([X, y, clusters], axis=1)
# #         valid_idx = combined.dropna().index
        
# #         X_clean = X.loc[valid_idx]
# #         y_clean = y.loc[valid_idx]
# #         clusters_clean = clusters.loc[valid_idx]
        
# #         # Add constant
# #         X_clean = sm.add_constant(X_clean)
        
# #         return y_clean, X_clean, clusters_clean

# #     def _fit_model(self, y: pd.Series, X: pd.DataFrame, clusters: pd.Series) -> GLMResults:
# #         """Fit GLM with clustered standard errors."""
# #         glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
# #         return glm.fit(cov_type='cluster', cov_kwds={'groups': clusters})


# # # Convenience factory function
# # def create_regression_runner(
# #     target_col: str,
# #     experimental_groups: List[str],
# #     cluster_col: Optional[str] = None
# # ) -> BaseRegressionRunner:
# #     """Factory function to create the appropriate regression runner."""
# #     if cluster_col:
# #         return ClusteredErrorRegression(target_col, experimental_groups, cluster_col)
# #     else:
# #         return StandardErrorRegression(target_col, experimental_groups)

