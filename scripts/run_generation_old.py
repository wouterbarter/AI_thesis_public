import hashlib
import yaml
import gc
import torch
import pandas as pd
from pathlib import Path
from src.prompt_manager import PromptManager
from src.modeler import Modeler
from src.pipeline import run_experiment
from src import paths
from src.data_manager import DataManager
from src.results import ResultsContainer

# # Optimization for Qwen3.5
# import os

# # Force Triton to skip autotuning so the Turing GPU doesn't deadlock
# os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"

# # Manually inject the CUDA path so the C++ kernels can find their DLLs
# cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
# if cuda_path not in os.environ["PATH"]:
#     os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
# print("TRITON AUTOTUNE STATUS:", os.environ.get("TRITON_DISABLE_AUTOTUNE", "NOT SET"))

# try:
#     import fla
#     print("KERNELS LOADED SUCCESSFULLY.")
# except Exception as e:
#     print("KERNEL LOAD FAILED:", e)


def generate_output_filename_stem(dataset_name, model_name, prompt_template_id, assistant_prefix):
    """Generates a standardized, safe filename for an experiment run."""
    safe_model_name = model_name.replace('/', '-')
    if assistant_prefix == "":
        safe_prefix = "none"
    elif "Rating" in assistant_prefix:
        safe_prefix = "rating"
    else:
        safe_prefix = f"{hashlib.sha1(assistant_prefix.encode()).hexdigest()[:6]}"

    return (
        f"dataset-{dataset_name}_"
        f"model-{safe_model_name}_"
        f"prompt-{prompt_template_id}_"
        f"prefix-{safe_prefix}"
    )


def main(config: dict):
    """
    Main orchestration function for the data generation pipeline.
    """
    sandbox_mode = config['sandbox_mode'] == 'True'
    if sandbox_mode:
        print('Running in Sandbox mode')

    # --- 1. Load Configs & Data ---
    active_analysis_name = config['active_analysis']
    try:
        analysis_config = config['analyses'][active_analysis_name]
        gen_globals = config['generation_globals']
    except KeyError:
        print(
            f"Error: Config blocks for '{active_analysis_name}' or 'generation_globals' not found.")
        return

    print(f"--- 🚀 Starting Generation for: {analysis_config['name']} ---")

    # Load paths and params from config
    # Local vars
    analysis_name = config['active_analysis'].upper()
    if gen_globals['use_cleaned_data']:
        print("Using cleaned data")
        input_data_filename = analysis_config.get(["processed_data_filename"])

        if input_data_filename is None:
            print("Attempted loading cleaned data but no filename specified. Exitting.")
            return

        data_path = paths.PROCESSED_DATA_DIR / f'{input_data_filename}.parquet'
    else:
        print("Using uncleaned data")
        data_path = paths.RAW_DATA_DIR / \
            f'{analysis_config["raw_data_filename"]}.parquet'

    raw_id_col = analysis_config['keys']['raw_id_col']

    if not sandbox_mode:
        output_dir = paths.RESULTS_DIR / analysis_name
        prompt_suites_dir = paths.PROMPT_SUITE_DIR / analysis_name
    else:
        output_dir = paths.RESULTS_DIR / 'sandbox' / analysis_name
        prompt_suites_dir = paths.PROMPT_SUITE_DIR / 'sandbox' / analysis_name

    print(f"Output dir: {output_dir}")

    # Global vars
    models = gen_globals['models_to_run']
    batch_size = gen_globals['batch_size']
    top_k = gen_globals['top_k']
    shards_per_save = gen_globals['shards_per_save']

    # Load and filter prompts
    # Analysis-specific vars
    gen_analysis = analysis_config['generation']
    tags_to_skip = set(gen_analysis['tags_to_skip'])
    required_tags = set(gen_analysis['required_tags'])
    ids_to_skip = set(set(gen_analysis['ids_to_skip']))
    ids_to_include = set(gen_analysis['ids_to_include'])


    pm = PromptManager(folder=prompt_suites_dir)
    prompt_suites = pm.load_all(
        tags_to_skip=tags_to_skip,
        required_tags=required_tags,
        ids_to_skip=ids_to_skip,
        ids_to_include=ids_to_include)

    if len(prompt_suites) == 0:
        print('Prompt suites folder empty, quitting')
        return

    # TODO: 'tokens_5' should also be a config variable
    # prompt_templates = pm.get_filtered_prompts(required_tags=['tokens_5'])

    # Load and (optionally) limit data
    df = pd.read_parquet(data_path)

    limit = gen_globals.get('debug_row_limit')
    if limit:
        # TODO: Does not work properly when df has changed after previously generating partial results
        df = df[:limit]

    # Sort by input size to optimize padding
    input_variables = analysis_config['variable_names']

    # Sort by str len to optimize generation for padding
    ## .sum(axis=1) on string columns concatenates them together into a single Series
    ## Then we can safely use .str.len()
    df['_char_len'] = df[input_variables].astype(str).sum(axis=1).str.len()
    df = df.sort_values('_char_len').reset_index(drop=True)

    # The 'or {}' ensures that if data_filtering is None, it becomes an empty dict
    data_filters = gen_analysis.get('data_filtering') or {}
    cats_to_subset = data_filters.get('cats_to_subset', [])
    if len(cats_to_subset) >= 1:
        df = df[df['consolidated_categories'].isin(cats_to_subset)]
        print(f"Processing {len(df)} rows after filtering categories")
    else:
        print(f"Processing {len(df)} rows")

    torch.cuda.empty_cache()

    # --- 2. Run the Experiment Loop ---
    for model_name in models:
        print(f"--- Loading Model: {model_name} ---")
        modeler = Modeler(model_name)

        for prompt_suite in prompt_suites.values():
            # suite_tags = prompt_suite.tags

            # if "BARS" in suite_tags:
            #     if "holistic" in suite_tags:
            #         assistant_prefix = "Based strictly on the rubric, the Quality score is: "

            #     else:
            #         assistant_prefix = "Based strictly on the rubric, the {dim_name} score is: "
            # elif "naive" in suite_tags:  # Only naive when it is not BARS
            #     assistant_prefix = "The Quality score is: "
            # else:  # Require the assistant_prefix, if not defined, raise error
            #     raise ValueError(
            #         f"Invalid Experiment Configuration: PromptSuite '{prompt_suite.id}' "
            #         f"must be tagged with 'BARS' or 'naive'. Current tags: {suite_tags}"
            #     )

            assistant_prefix = 'Rating: '

            print(f"Running: {prompt_suite.id} | Prefix: '{assistant_prefix}'")

            file_stem = generate_output_filename_stem(
                active_analysis_name,
                model_name,
                prompt_suite.id,
                assistant_prefix
            )
            exp_output_dir = output_dir / file_stem

            prompt_suite.precompute_constraints(modeler.tokenizer)

            if model_name == 'Qwen/Qwen3.5-4B':
                chat_template_kwargs = {"enable_thinking": False}
            else:
                chat_template_kwargs = {}

            run_experiment(
                df=df,
                modeler=modeler,
                suite=prompt_suite,
                output_dir=exp_output_dir,
                file_stem=file_stem,
                model_name=model_name,
                batch_size=batch_size,
                id_col=raw_id_col,
                top_k=top_k,
                assistant_prefix=assistant_prefix,
                shards_per_save=shards_per_save,
                max_new_tokens=1,
                chat_template_kwargs=chat_template_kwargs)

        print(f"--- 🧹 Releasing VRAM from model: {model_name} ---")
        del modeler.model
        del modeler.tokenizer
        del modeler
        gc.collect()
        torch.cuda.empty_cache()

    print("--- ✅ Data Generation Complete ---")


if __name__ == "__main__":

    CONFIG_PATH = 'src/configs/config.yaml'

    print(f"Loading configuration from {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        full_config = yaml.safe_load(f)

    main(config=full_config)
