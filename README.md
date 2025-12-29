# LLM Evaluator Thesis

This repository implements, documents, and experiments with **LLM-as-a-Judge (LLMAJ)** methods for subjective text evaluation in marketing copy.

## 🧭 Structure Overview

| Folder | Description |
|--------|-------------|
| `src/` | Core reusable code modules (data, evaluation, metrics, analysis, utils). |
| `experiments/` | Reproducible experiment folders (construct validity, reliability, etc.). |
| `notebooks/` | Interactive exploration and visualization notebooks. |
| `docs/` | Markdown documentation, integrated with thesis writing. |
| `data/` | Local datasets, intermediate files, cache. |

## 🚀 Workflow

1. Implement new methods in `src/`.
2. Test interactively in `notebooks/`.
3. When stable, wrap into an `experiments/` folder with a `config.yaml` and `run_experiment.py`.
4. Store outputs under `results/` and summarize in `notes.md`.
5. Document high-level learnings in `docs/`.

## 🧩 Quickstart

```bash
# (Optional) create environment
python -m venv venv && source venv/bin/activate

# install deps
pip install -r requirements.txt

# run first experiment
python experiments/01_construct_validity/run_experiment.py