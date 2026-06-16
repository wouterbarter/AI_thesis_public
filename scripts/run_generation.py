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


def resolve_experiment_config(output_dir: Path, analysis_config: dict, gen_globals: dict) -> tuple[dict, dict]:
    """
    Synchronizes the experiment folder with a stable ground truth, 
    while allowing orchestration variables to remain dynamic.
    """
    # 1. Define what should NEVER be locked into the folder's history
    # These are "Transient" variables that stay in your main config.yaml
    transient_gen_keys = ['models_to_run',
                          'debug_row_limit', 'batch_size', 'items_per_shard']
    transient_analysis_keys = ['ids_to_skip',
                               'ids_to_include', 'required_tags', 'tags_to_skip']

    # 2. Capture current values from the main config to re-inject later
    current_gen_vars = {k: gen_globals.get(k) for k in transient_gen_keys}

    # Prompts are nested under generation -> prompts
    current_prompt_vars = {k: analysis_config['generation']['prompts'].get(k)
                           for k in transient_analysis_keys}

    # 3. Create a "Stable" version for saving/comparison
    stable_analysis = analysis_config.copy()
    stable_gen = gen_globals.copy()

    # Remove the transient keys from the stable copy
    for k in transient_gen_keys:
        stable_gen.pop(k, None)
    for k in transient_analysis_keys:
        stable_analysis['generation']['prompts'].pop(k, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.yaml"

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                saved = yaml.safe_load(f)

            # Use the SAVED versions for the 'stable' parts of the config
            analysis_config = saved["analysis_config"]
            gen_globals = saved["gen_globals"]
            print(f"✅ Loaded stable parameters from {output_dir.name}")

        except Exception as e:
            print(
                f"❌ Error reading config at {config_path}: {e}. Falling back to main config.")
    else:
        # First run: Lock the current stable state as the ground truth
        print(f"💾 Establishing ground truth for {output_dir.name}")
        with open(config_path, "w") as f:
            yaml.dump({
                "analysis_config": stable_analysis,
                "gen_globals": stable_gen
            }, f)

        analysis_config = stable_analysis
        gen_globals = stable_gen

    # 4. RE-INJECT the dynamic values back into the live dictionaries
    # This ensures your loops in run_generation.py still work
    for k, v in current_gen_vars.items():
        gen_globals[k] = v

    for k, v in current_prompt_vars.items():
        analysis_config['generation']['prompts'][k] = v

    return analysis_config, gen_globals


def main(config: dict):
    """
    Main orchestration function for the data generation pipeline.
    """
    sandbox_mode = config['sandbox_mode']
    if sandbox_mode:
        print('Running in Sandbox mode')

    # --- Load Configs & Data ---
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

    input_data_filename = analysis_config['input_filenames'].get(
        "processed_data_filename")
    print(f"Loading data from: {input_data_filename}")

    if not sandbox_mode:
        output_dir = paths.RESULTS_DIR / analysis_name
        prompt_suites_dir = paths.PROMPT_SUITE_DIR / analysis_name
    else:
        output_dir = paths.RESULTS_DIR / 'sandbox' / analysis_name
        prompt_suites_dir = paths.PROMPT_SUITE_DIR / 'sandbox' / analysis_name

    print(f"Output dir: {output_dir}")

    # In main() of run_generation.py, right after determining the output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the new results are generated from the same input data
    analysis_config, gen_globals = resolve_experiment_config(
        output_dir, analysis_config, gen_globals)

    # Global vars
    models = gen_globals['models_to_run']
    items_per_shard = gen_globals['items_per_shard']
    # shards_per_save = gen_globals['shards_per_save']

    # Analysis-specific vars
    gen_analysis = analysis_config['generation']

    prompt_params = gen_analysis['prompts']
    tags_to_skip = set(prompt_params['tags_to_skip'])
    required_tags = set(prompt_params['required_tags'])
    ids_to_skip = set(set(prompt_params['ids_to_skip']))
    ids_to_include = set(prompt_params['ids_to_include'])

    # Load and filter prompts
    pm = PromptManager(folder=prompt_suites_dir)
    prompt_suites = pm.load_all(
        tags_to_skip=tags_to_skip,
        required_tags=required_tags,
        ids_to_skip=ids_to_skip,
        ids_to_include=ids_to_include)

    if len(prompt_suites) == 0:
        print('Prompt suites folder empty, quitting')
        return

    # Data prep: Load and (optionally) limit data, sort by input length, optionally filter
    data_path = paths.PROCESSED_DATA_DIR / f"{input_data_filename}.parquet"
    df = pd.read_parquet(data_path)

    # Check if the data contain the {variables} required by PromptSuite
    print("🔍 Validating datasets against prompt templates...")
    for suite in prompt_suites.values():
        suite.validate_dataset(df)
    print("✅ All prompts successfully validated. Commencing generation.")

    limit = gen_globals.get('debug_row_limit')
    if limit:
        # TODO: Does not work properly when df has changed after previously generating partial results
        df = df[:limit]

    # The 'or {}' ensures that if data_filtering is None, it becomes an empty dict
    data_params = gen_analysis['data']
    data_filters = data_params.get('filtering') or {}
    if analysis_name == 'barter_deals':
        cats_to_subset = data_filters.get('cats_to_subset', [])
        if len(cats_to_subset) >= 1:
            df = df[df['consolidated_categories'].isin(cats_to_subset)]
            print(f"Processing {len(df)} rows after filtering categories")
        else:
            print(f"Processing {len(df)} rows")

    torch.cuda.empty_cache()

    # --- Run the Experiment Loop ---
    for model_name in models:
        print(f"--- Loading Model: {model_name} ---")
        modeler = Modeler(model_name)
        # Model specific kwargs
        if model_name == 'Qwen/Qwen3.5-4B':
            chat_template_kwargs = {"enable_thinking": False}
        else:
            chat_template_kwargs = {}

        for prompt_suite in prompt_suites.values():
            assistant_prefix = 'Rating: '

            print(f"Running: {prompt_suite.id} | Prefix: '{assistant_prefix}'")

            file_stem = generate_output_filename_stem(
                active_analysis_name,
                model_name,
                prompt_suite.id,
                assistant_prefix
            )
            exp_output_dir = output_dir / file_stem

            # analysis_config, gen_globals = resolve_experiment_config(
            #     exp_output_dir, analysis_config, gen_globals)

            prompt_suite.precompute_constraints(modeler.tokenizer)

            run_experiment(
                df=df,
                modeler=modeler,
                suite=prompt_suite,
                output_dir=exp_output_dir,
                file_stem=file_stem,
                model_name=model_name,
                batch_size=gen_globals['batch_size'],
                id_col=analysis_config['keys']['raw_id_col'],
                top_k=gen_globals['top_k'],
                assistant_prefix=assistant_prefix,
                # shards_per_save=shards_per_save,
                items_per_shard=items_per_shard,
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
