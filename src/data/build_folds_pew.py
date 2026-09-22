"""Build stratified cross-validation folds for the Pew India respondents.

src.data.build_folds.build_folds() computes its stratification target
internally from hardcoded WVS column names (H_URBRURAL, Q288, Q260), so it
can't be called as-is against Pew columns. This reimplements the same
StratifiedKFold loop with a Pew-specific stratification target
(Urban x income-band, the closest clean, low-missingness analogue to the
WVS urban/rural x income-tercile target) instead of duplicating it via a
monkeypatch.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import DATA_PROCESSED, N_FOLDS, RANDOM_SEED

logger = logging.getLogger(__name__)


def create_stratification_target(df: pd.DataFrame) -> pd.Series:
    """Urban/rural x income-band, same spirit as the WVS target -- both are
    present in Pew as clean, low-missingness, near-universally-answered
    columns, unlike most demographics here which still await real labels."""
    strat_cols = []
    if "Urban" in df.columns:
        strat_cols.append(df["Urban"].fillna(-1).astype(int).astype(str))
    if "QINCINDrec" in df.columns:
        strat_cols.append(df["QINCINDrec"].fillna(-1).astype(int).astype(str))
    if not strat_cols:
        strat_cols.append(df["QGEN"].fillna(-1).astype(int).astype(str))
    return pd.concat(strat_cols, axis=1).astype(str).agg("_".join, axis=1)


def main(input_path: Path = None, output_path: Path = None):
    input_path = input_path or DATA_PROCESSED / "pew_india_2021.parquet"
    output_path = output_path or DATA_PROCESSED / "pew_folds.json"

    if not input_path.exists():
        logger.error(f"Input data not found: {input_path}\nRun: python -m src.data.load_pew")
        return False

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} respondents")

    strat_target = create_stratification_target(df)

    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))
    respondent_ids = df["respondent_id"].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, strat_target)):
        fold = {
            "fold": fold_idx,
            "train": respondent_ids[train_idx].tolist(),
            "test": respondent_ids[test_idx].tolist(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        folds.append(fold)
        logger.info(f"Fold {fold_idx}: train={fold['n_train']}, test={fold['n_test']}")

    all_test_ids = set()
    for fold in folds:
        test_set = set(fold["test"])
        assert not (all_test_ids & test_set), "Respondent appears in multiple test sets!"
        all_test_ids.update(test_set)

    fold_config = {
        "n_respondents": len(df), "n_folds": N_FOLDS, "random_seed": RANDOM_SEED, "folds": folds,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fold_config, f, indent=2)
    logger.info(f"Saved fold configuration to {output_path}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(0 if main() else 1)
