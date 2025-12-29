# In: src/paths.py
from pathlib import Path

# This file lives in 'src/'.
# Its parent is 'src/'.
# Its parent's parent is the project root.
# .resolve() makes it a clean, absolute path.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Define all your *other* key paths relative to this anchor
CONFIG_PATH = PROJECT_ROOT / "src/configs/config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPT_TEMPLATE_DIR = PROJECT_ROOT / "prompts/PromptTemplates"
PROMPT_SUITE_DIR = PROJECT_ROOT / "prompts/PromptSuites"


