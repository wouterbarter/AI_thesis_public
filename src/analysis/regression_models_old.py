import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import hashlib
from pathlib import Path
from src import paths

# --- 1. MAIN "RUN" FUNCTION (Called by run_analysis.py) ---

def run_grouped_regression(df: pd.DataFrame, config: dict, output_paths: dict):
    """
    "RUN" Step: Loops through all experimental groups and fits a model,
    saving one result file per group.
    """
    model_config = config['model_vars']
    experimental_groups = model_config['experimental_groups']
    output_dir = paths.PROJECT_ROOT / output_paths['raw_results_dir']
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    grouped_data = df.groupby(experimental_groups)
    
    print(f"Total groups to analyze: {len(grouped_data)}. Checking for saved results...")
    
    for group_name, group_df in grouped_data:
        
        # Create a unique, safe filename for this group's result
        result_filename = _get_result_filename(group_name, model_config['model_type'])
        result_path = output_dir / result_filename
        
        # This is the "caching" step:
        if result_path.exists():
            print(f"  Skipping {group_name}: Result file already exists.")
            continue
            
        print(f"  Analyzing: {group_name}")
        
        try:
            # 1. PREP: Get model-specific inputs (y, X, etc.)
            model_inputs = _prepare_inputs(group_df.copy(), model_config)
            
            # 2. FIT: Run the model
            model = _fit_model(model_inputs, model_config)
            
            # 3. SAVE: Extract and save the raw result
            _save_result(model, group_name, model_config, result_path)
            
        except Exception as e:
            print(f"    FAILED for group {group_name}: {e}")

# --- 2. MAIN "COMPILE" FUNCTION (Called by run_analysis.py) ---

def compile_results(output_paths: dict) -> pd.DataFrame:
    """
    "COMPILE" Step: Reads all individual JSON result files from the
    raw_results_dir and compiles them into a single DataFrame.
    """
    results_dir = paths.PROJECT_ROOT / output_paths['raw_results_dir']
    
    results_list = []
    for f in results_dir.glob("*.json"):
        with open(f, 'r') as file:
            results_list.append(json.load(file))
            
    if not results_list:
        raise FileNotFoundError(f"No result files found in {results_dir}")
        
    return pd.DataFrame(results_list)

# --- 3. HELPER: "PREPARE" ROUTER (The Strategy Pattern) ---

def _prepare_inputs(df: pd.DataFrame, config: dict) -> dict:
    """
    "Router" function that calls the correct data preparation
    function based on the model_type in the config.
    """
    model_type = config['model_type']
    
    if model_type == 'OLS':
        return _prep_ols_inputs(df, config)
    elif model_type == 'CRE_NB':
        return _prep_cre_nb_inputs(df, config)
    # elif model_type == 'FEP':
    #     return _prep_fep_inputs(df, config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def _prep_ols_inputs(df: pd.DataFrame, config: dict) -> dict:
    """Prepares data for OLS: y, X, add_constant, dropna."""
    # This is your OLS logic from before
    X_dummies = pd.get_dummies(df[config['categorical_predictors']], drop_first=True, dtype=int)
    all_predictors = config['binary_predictors'] + [config['var_of_interest']]
    X_combined = pd.concat([df[all_predictors], X_dummies], axis=1)
    
    X = sm.add_constant(X_combined)
    y = df[config['target']]
    
    final_data = pd.concat([y, X], axis=1).dropna()
    y_final = final_data[config['target']]
    X_final = final_data.drop(columns=[config['target']])
    
    if len(y_final) < (X_final.shape[1] + 2):
        raise ValueError("Not enough observations after dropna.")
        
    return {'y': y_final, 'X': X_final}

def _prep_cre_nb_inputs(df: pd.DataFrame, config: dict) -> dict:
    """Prepares data for Correlated Random Effects (CRE) NB model."""
    # This is the logic from our CRE_NB script
    X_dummies = pd.get_dummies(df[config['categorical_predictors']], drop_first=True, dtype=int)
    
    # Create cluster averages
    avg_names = []
    for col in config['varying_predictors']:
        avg_col_name = f'partner_avg_{col}'
        df[avg_col_name] = df.groupby(config['cluster_col'])[col].transform('mean')
        avg_names.append(avg_col_name)
    
    all_predictors = (
        config['binary_predictors'] + 
        [config['var_of_interest']] + 
        list(X_dummies.columns) + 
        avg_names
    )
    
    X = sm.add_constant(df[all_predictors])
    y = df[config['target']]
    cluster = df[config['cluster_col']]
    
    final_data = pd.concat([y, X, cluster], axis=1).dropna()
    
    y_final = final_data[config['target']]
    X_final = final_data.drop(columns=[config['target'], config['cluster_col']])
    cluster_final = final_data[config['cluster_col']]
    
    if len(y_final) < (X_final.shape[1] + 2):
        raise ValueError("Not enough observations after dropna.")
        
    return {'y': y_final, 'X': X_final, 'cluster': cluster_final}


# --- 4. HELPER: "FIT" ROUTER (The Strategy Pattern) ---

def _fit_model(inputs: dict, config: dict):
    """
    "Router" function that calls the correct statsmodels
    fit function based on the model_type.
    """
    model_type = config['model_type']
    
    if model_type == 'OLS':
        return sm.OLS(inputs['y'], inputs['X']).fit()
        
    elif model_type == 'CRE_NB':
        glm_model = sm.GLM(
            inputs['y'], 
            inputs['X'],
            family=sm.families.NegativeBinomial()
        )
        return glm_model.fit(
            cov_type='cluster',
            cov_kwds={'groups': inputs['cluster']}
        )
    
    # elif model_type == 'FEP':
    #     ...
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# --- 5. HELPER: "SAVE" FUNCTIONS ---

def _save_result(model, group_name: tuple, config: dict, filepath: Path):
    """
    Extracts the key results from a fitted model and saves
    them to a single JSON file.
    """
    var_of_interest = config['var_of_interest']
    
    # This is the data we want to save
    result_data = {
        'model_type': config['model_type'],
        'n_obs': model.nobs,
        'rsquared_adj': getattr(model, 'rsquared_adj', None),
        'pseudo_rsquared': getattr(model, 'pseudo_rsquared', None)
    }
    
    # Add experimental group names
    for i, name in enumerate(config['experimental_groups']):
        result_data[name] = group_name[i]
        
    # Add stats for our variable of interest
    try:
        result_data['coef'] = model.params[var_of_interest]
        result_data['p_value'] = model.pvalues[var_of_interest]
        result_data['std_err'] = model.bse[var_of_interest]
        conf = model.conf_int().loc[var_of_interest]
        result_data['ci_low'] = conf.iloc[0]
        result_data['ci_high'] = conf.iloc[1]
    except KeyError:
        result_data['error'] = f"'{var_of_interest}' was not in model."
    
    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=4)

def _get_result_filename(group_name: tuple, model_type: str) -> str:
    """Creates a unique, safe filename for a group."""
    # Use a hash to create a unique ID for the group
    group_str = "".join(str(g) for g in group_name) + model_type
    group_hash = hashlib.md5(group_str.encode()).hexdigest()
    return f"result_{group_hash}.json"