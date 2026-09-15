"""Configuration and constants for the silicon sampling pipeline."""

from pathlib import Path
from typing import Literal

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_REFERENCE = DATA_DIR / "reference"  # Codebook-derived metadata (committed)
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = PROJECT_ROOT / "paper"
DECK_DIR = PROJECT_ROOT / "deck"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# WVS settings
WVS_INDIA_N = 1692  # Approximate; verify at download
COUNTRY_CODE = "IND"
WVS_WAVE = 7

# The India survey was fielded 2022-2023 (completed July 2023) and only entered
# the WVS-7 cross-national file at v6.0. Releases at v5.0 and earlier cover 64
# countries and contain NO India rows -- see DATA_ACQUISITION.md.
WVS_MIN_VERSION = "v6_0"

# Filenames the WVS-7 cross-national zip unpacks to, best first. Both the
# standard and the inverted-scale archives unpack to the same name, so the
# filename alone cannot tell them apart.
WVS_CSV_CANDIDATES = (
    "WVS_Cross-National_Wave_7_csv_v6_0.csv",
    "WVS_Cross-National_Wave_7_csv_v5_0.csv",  # no India; rejected downstream
)


def resolve_wvs_csv(data_raw: Path = DATA_RAW) -> Path:
    """Locate the WVS-7 cross-national CSV in data/raw/.

    Tries the known release filenames, then falls back to any CSV that looks
    like a WVS cross-national extract, so a renamed download still works.

    Raises:
        FileNotFoundError: with download instructions if nothing matches.
    """
    for name in WVS_CSV_CANDIDATES:
        candidate = data_raw / name
        if candidate.exists():
            return candidate

    loose = sorted(p for p in data_raw.glob("*.csv") if "Cross-National" in p.name)
    if loose:
        return loose[0]

    raise FileNotFoundError(
        f"No WVS-7 cross-national CSV found in {data_raw}/\n"
        f"Expected: {WVS_CSV_CANDIDATES[0]}\n"
        "Download it (free registration) from\n"
        "  https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp\n"
        "  -> Statistical Data Files -> 'WVS Cross-National Wave 7 csv v6 0.zip'\n"
        "then unzip into data/raw/. See DATA_ACQUISITION.md for the full walkthrough."
    )

# Models
MODELS = {
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "provider": "huggingface",
        "quantize": "4bit",  # QLoRA
    },
    "gemini-3.5-flash-lite": {
        # gemini-2.5-flash was retired for new API keys in early 2026;
        # verified working against a real key -- see src/inference/gemini.py
        "provider": "google",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "provider": "huggingface",
        "quantize": "4bit",
    },
    "gemma-3-4b": {
        "hf_id": "google/gemma-3-4b-it",
        "provider": "huggingface",
        "quantize": "4bit",
    },
}

# Prompt conditions
PROMPT_CONDITIONS = {
    "P0": "No demographics (control)",
    "P1": "Minimal (age, sex, region)",
    "P2": "Full structured (14 attributes)",
    "P3": "Full naturalistic backstory",
}

# CV settings
N_FOLDS = 5
RANDOM_SEED = 42

# Item selection settings
MIN_RESPONSE_SCALE_SIZE = 4  # Min ordinal scale points
MAX_MISSINGNESS_PCT = 10  # Drop items with >10% missing
MIN_MODAL_ENTROPY = 0.15  # Avoid near-unanimous items

# Subgroup variables for stratification and slicing
SUBGROUP_VARS = {
    "urban_rural": ["Urban", "Rural"],
    "income_tercile": ["Low", "Mid", "High"],
    "education": ["<=Primary", "Secondary", "Tertiary"],
    "region": None,  # Will be populated from data
    "language": None,  # Will be populated from data
    "age_band": None,  # Will be binned from Q262
    "sex": ["Male", "Female"],
    "religion": None,  # Will be populated from data
}

# Bootstrap settings
N_BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CI_LEVEL = 0.95

# Inference caching
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Fine-tuning settings (QLoRA)
QLORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "max_seq_length": 512,
    "optim": "paged_adamw_8bit",
}
