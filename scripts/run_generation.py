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


def main(config: dict):
    """
    Main orchestration function for the data generation pipeline.
    """



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
    output_dir = paths.RESULTS_DIR / analysis_name
    raw_data_path = paths.RAW_DATA_DIR / f'{analysis_name}.parquet'
    prompt_suites_dir = paths.PROMPT_SUITE_DIR / analysis_name

    # output_dir = Path(analysis_config['paths']['results_logs_dir'])
    # raw_data_path = Path(analysis_config['paths']['raw_data'])
    # prompt_templates_dir = Path(analysis_config['paths']['prompt_templates_dir'])

    raw_id_col = analysis_config['keys']['raw_id_col']
    # eval_id_col = analysis_config['keys']['evaluations_id_col'] # Auto-filled, ensure PromptTemplate and df/dict Variable match #TODO Safety check

    # Global vars
    models = gen_globals['models_to_run']
    # assistant_prefices = gen_globals['assistant_prefices']
    batch_size = gen_globals['batch_size']
    top_k = gen_globals['top_k']

    # Load and filter prompts
    tags_to_skip = set(['baseline']) # TODO WARNING HARDCODED,
    required_tags = set(['holistic'])
    pm = PromptManager(folder=prompt_suites_dir)
    prompt_suites = pm.load_all(tags_to_skip=tags_to_skip, required_tags=required_tags)

    # TODO: 'tokens_5' should also be a config variable
    # prompt_templates = pm.get_filtered_prompts(required_tags=['tokens_5'])

    # Load and (optionally) limit data
    df = pd.read_parquet(raw_data_path)
    limit = gen_globals.get('debug_row_limit')

    if limit:
        # TODO: Does not work properly when df has changed after previously generating partial results
        df = df[:limit]

    # if limit:
    #     set_new_ids = set(df[:limit][raw_id_col])

    #     dm = DataManager(output_dir)
    #     dm.load_all()

    #     # Error handling for when results dir is empty
    #     try:
    #         set_existing_ids = set(dm.master_df[eval_id_col])
    #     except:
    #         set_existing_ids = set()

    #     set_ids_union = set.union(set_existing_ids, set_new_ids)

    #     df = df[df[raw_id_col].isin(set_ids_union)]
    #     if hard_limit:
    #         df = df[df[raw_id_col].isin(set_existing_ids)][:limit]

    #     print(f"--- ⚠️  WARNING: Running in debug mode with {len(df)} rows ---")

    torch.cuda.empty_cache()

    # --- 2. Run the Experiment Loop ---
    for model_name in models:
        print(f"--- Loading Model: {model_name} ---")
        modeler = Modeler(model_name)

        for prompt_suite in prompt_suites.values():
            suite_tags = prompt_suite.tags

            # if skip_tag:  # TODO WARNING HARDCODED
            #     if tag_to_skip in prompt_suite.tags:
            #         print(f"Skipping {prompt_suite.id} with tags {suite_tags}")
            #         continue

            if "BARS" in suite_tags:
                if "holistic" in suite_tags:
                    assistant_prefix = "Based strictly on the rubric, the Quality score is: "

                else:
                    assistant_prefix = "Based strictly on the rubric, the {dim_name} score is: "
            elif "naive" in suite_tags:  # Only naive when it is not BARS
                assistant_prefix = "The Quality score is: "
            else:  # Require the assistant_prefix, if not defined, raise error
                raise ValueError(
                    f"Invalid Experiment Configuration: PromptSuite '{prompt_suite.id}' "
                    f"must be tagged with 'BARS' or 'naive'. Current tags: {suite_tags}"
                )

            print(f"Running: {prompt_suite.id} | Prefix: '{assistant_prefix}'")

            file_stem = generate_output_filename_stem(
                active_analysis_name,
                model_name,
                prompt_suite.id,
                assistant_prefix
            )
            exp_output_dir = output_dir / file_stem

            prompt_suite.precompute_constraints(modeler.tokenizer)

            shards_per_save = 10
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
                max_new_tokens=1)

        print(f"--- 🧹 Releasing VRAM from model: {model_name} ---")
        del modeler.model
        del modeler.tokenizer
        del modeler
        gc.collect()
        torch.cuda.empty_cache()

    print("--- ✅ Data Generation Complete ---")


# --- This is the standard Python entry point guard ---
if __name__ == "__main__":

    CONFIG_PATH = 'src/configs/config.yaml'  # Or pass as a CLI argument

    print(f"Loading configuration from {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        full_config = yaml.safe_load(f)

    main(config=full_config)
