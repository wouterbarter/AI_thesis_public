import pandas as pd
import numpy as np


def collect_feedbackqa_example_outputs(
    input_id: str,
    validator,
    cv_results: pd.DataFrame,
    final_df: pd.DataFrame,
    original_df: pd.DataFrame,
    models=("Gemma 4", "Qwen 3.5"),
    conditions=("Formative", "Holistic Naive", "Holistic Informed"),
    id_col: str = "input_id",
):
    """
    Collects OOF predictions and raw LLM measurements for one FeedbackQA artefact.

    Returns
    -------
    artefact_row : pd.Series
        Original artefact metadata and labels.
    oof_predictions : pd.DataFrame
        One row per model-condition with OOF prediction, target, and error.
    llm_scores_long : pd.DataFrame
        Raw criterion/holistic scores for the selected models.
    llm_scores_wide : pd.DataFrame
        Same scores in wide format: one row per model, one column per dimension.
    """

    # ------------------------------------------------------------------
    # 1. Original artefact row
    # ------------------------------------------------------------------
    artefact_matches = original_df.loc[original_df[id_col] == input_id]

    if artefact_matches.empty:
        raise ValueError(f"No artefact found for {id_col}={input_id}")

    if len(artefact_matches) > 1:
        print(
            f"Warning: {len(artefact_matches)} rows found for {id_col}={input_id}; using first.")

    artefact_row = artefact_matches.iloc[0]

    # ------------------------------------------------------------------
    # 2. OOF predictions for selected model-condition pairs
    # ------------------------------------------------------------------
    prediction_rows = []

    for model in models:
        for condition in conditions:
            label = f"{model}_{condition}"

            if label not in validator.runner.run_data:
                print(f"Skipping missing label: {label}")
                continue

            # Build artefact-level OOF dataset for this label
            oof_df = validator.build_oof_prediction_dataset(
                label=label,
                original_df=original_df,
                id_col=id_col,
                pred_col="OOF_prediction",
            )

            row = oof_df.loc[oof_df[id_col] == input_id]

            if row.empty:
                print(
                    f"No OOF prediction found for {label}, {id_col}={input_id}")
                continue

            row = row.iloc[0]

            oof_pred = row["OOF_prediction"]
            oof_target = row["OOF_target"]

            prediction_rows.append({
                "model": model,
                "condition": condition,
                "label": label,
                "OOF_prediction": oof_pred,
                "OOF_target": oof_target,
                "absolute_error": abs(oof_pred - oof_target),
                "OOF_n_observations": row.get("OOF_n_observations", np.nan),
                "CV_RMSE": cv_results.loc[label, "RMSE"] if label in cv_results.index else np.nan,
                "CV_Spearman_rho": cv_results.loc[label, "Spearman_rho"] if label in cv_results.index else np.nan,
            })

    oof_predictions = pd.DataFrame(prediction_rows)

    # ------------------------------------------------------------------
    # 3. Raw LLM scores for the selected models
    # ------------------------------------------------------------------
    llm_scores_long = (
        final_df.loc[
            (final_df[id_col] == input_id)
            & (final_df["model_name_clean"].isin(models)),
            ["model_name_clean", "dimension_name_clean", "mean_rating"]
        ]
        .rename(columns={
            "model_name_clean": "model",
            "dimension_name_clean": "dimension",
            "mean_rating": "raw_score",
        })
        .sort_values(["model", "dimension"])
        .reset_index(drop=True)
    )

    llm_scores_wide = (
        llm_scores_long
        .pivot_table(
            index="model",
            columns="dimension",
            values="raw_score",
            aggfunc="first"
        )
        .reset_index()
    )

    return artefact_row, oof_predictions, llm_scores_long, llm_scores_wide
