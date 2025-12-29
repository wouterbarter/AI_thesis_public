# run_analysis.py
import yaml
from src import data_processing as dp
from src.analysis import regression_models

def main():
    # 1. Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load and Process Data
    print("Loading data...")
    raw_df = dp.load_data(config['data_paths']['raw_db'])
    results_df = dp.load_results(...) # Or load from config path
    
    print("Cleaning and merging...")
    clean_df = dp.clean_data(raw_df)
    merged_df = dp.merge_data(clean_df, results_df)
    
    # Optional: Save the clean, merged data
    merged_df.to_parquet(config['data_paths']['final_merged'])

    # 3. Run Analysis
    print("Running grouped regressions...")
    reg_models = regression_models.run_grouped_regression(
        merged_df,
        cluster_col=config['cluster_variable'],
        target_col=config['target_variable'],
        # ... other params from config
    )

    # 4. Extract Results
    print("Extracting results...")
    comparison_df = regression_models.extract_results(reg_models)
    
    # 5. Save Results
    comparison_df.to_csv('./reports/regression_summary.csv')
    print("Analysis complete. Results saved.")

if __name__ == "__main__":
    main()