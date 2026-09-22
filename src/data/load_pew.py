"""Load and preprocess the Pew Research "Religion in India" (2021) survey."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pandas.api.types as ptypes

from src.config import DATA_PROCESSED, PEW_INDIA_N, resolve_pew_csv

logger = logging.getLogger(__name__)

# Pew's don't-know/refused/not-applicable/blank sentinel codes, confirmed by
# inspecting the real released CSV: every substantive item column tops out
# with a cluster at 96-99 (96=other/none-of-these, 97=refused, 98=don't
# know, 99=missing), plus a small number of columns using a literal blank
# string for "not asked" (skip logic). This mirrors WVS's -1..-5 scheme --
# same treatment, different sentinel values.
MISSING_CODES = {96, 97, 98, 99, 996, 997, 998, 999}

# Columns that are IDs/weights, not survey items -- never touch these during
# missing-value recoding, and never select them as a training-target item.
# respondent_id (added by add_respondent_id(), a copy of QRID) is listed
# here too: it used to slip through the recode loop since it doesn't exist
# yet when this set is defined, silently NaN-ing out any respondent whose
# ID happened to equal a sentinel code (96-99, 996-999). No such collision
# actually occurs in this release's ID range (confirmed by direct check),
# but that was luck, not a guarantee -- respondent_id must never be treated
# as a survey answer regardless.
NON_ITEM_COLS = {"COUNTRY", "QRID", "weight", "QMLangRec", "respondent_id"}


def load_pew_raw(csv_path: Path) -> pd.DataFrame:
    """Load the raw Pew CSV file."""
    logger.info(f"Loading Pew CSV from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df)} total respondents, {len(df.columns)} columns")
    return df


def recode_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Recode Pew's 96-99 (and blank-string) missing/DK/refused codes to NaN.

    Only touches columns not in NON_ITEM_COLS -- weight and QRID are never
    survey answers and must survive untouched.
    """
    logger.info("Recoding missing values...")

    item_cols = [c for c in df.columns if c not in NON_ITEM_COLS]

    n_recoded = 0
    n_coerced_cols = 0
    for col in item_cols:
        # Blank-string cells (skip-logic "not asked") show up as NaN already
        # for a column pandas inferred as numeric, or as a literal " "/""
        # string for a non-numeric column -- coerce any non-numeric column
        # to numeric first so both cases end up as one consistent NaN, then
        # apply the sentinel recode on top. `dtype == object` alone is NOT
        # enough to detect this: about half of this CSV's columns (150/308,
        # confirmed by direct inspection) come back as pandas' newer
        # Arrow-backed "string" dtype rather than legacy "object" -- that
        # comparison silently returns False for them, so they skipped
        # to_numeric entirely and kept raw strings (including un-recoded
        # sentinel codes) straight through to the saved parquet, a real bug
        # caught only by later re-testing against an item that happened to
        # live in one of those 150 columns.
        if not ptypes.is_numeric_dtype(df[col]):
            series = pd.to_numeric(df[col], errors="coerce")
            n_coerced_cols += 1
        else:
            series = df[col]
        mask = series.isin(MISSING_CODES)
        if mask.any():
            n_recoded += int(mask.sum())
        df[col] = series.where(~mask, pd.NA)

    logger.info(f"Coerced {n_coerced_cols} non-numeric-dtype columns to numeric. Recoded {n_recoded} missing value codes across {len(item_cols)} columns")
    return df


def add_respondent_id(df: pd.DataFrame) -> pd.DataFrame:
    """QRID is Pew's own respondent ID -- reuse it as respondent_id so it
    lines up with the same column name the WVS pipeline and every
    downstream src.report.* function already expects."""
    df = df.copy()
    df["respondent_id"] = df["QRID"].astype(int)
    return df


def verify_demographics(df: pd.DataFrame) -> dict:
    """Log and return basic demographic marginals, as a sanity check against
    the published Pew methodology numbers (N=29,999; religion breakdown
    22,975 Hindu / 3,336 Muslim / 1,782 Sikh / 1,011 Christian / 719
    Buddhist / 109 Jain / 67 other -- see Pew's June 2021 release)."""
    results = {}

    if "QGEN" in df.columns:
        results["sex"] = df["QGEN"].value_counts(normalize=True).to_dict()
        logger.info(f"Sex distribution (QGEN): {results['sex']}")

    if "Urban" in df.columns:
        results["urban_rural"] = df["Urban"].value_counts(normalize=True).to_dict()
        logger.info(f"Urban/rural distribution (Urban): {results['urban_rural']}")

    if "QRELSING" in df.columns:
        results["religion"] = df["QRELSING"].value_counts().to_dict()
        logger.info(f"Religion counts (QRELSING): {results['religion']}")

    return results


def main():
    csv_path = resolve_pew_csv()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_pew_raw(csv_path)

    if abs(len(df) - PEW_INDIA_N) > 50:
        logger.warning(
            f"Row count {len(df)} differs from the expected {PEW_INDIA_N}. "
            "Confirm you downloaded the 'India Religion Public Data ... (All Vars).csv' "
            "release, not a subset or a different wave."
        )

    df = add_respondent_id(df)
    df = recode_missing_values(df)
    verify_demographics(df)

    output_path = DATA_PROCESSED / "pew_india_2021.parquet"
    df.to_parquet(output_path)
    logger.info(f"Saved {len(df)} respondents to {output_path}")

    # Verify no missing codes survived recoding.
    item_cols = [c for c in df.columns if c not in NON_ITEM_COLS and c != "respondent_id"]
    for col in item_cols:
        series = df[col]
        if series.dtype != object and series.isin(MISSING_CODES).any():
            logger.error(f"Missing codes survived in {col}!")
            return False

    logger.info("Data load and clean complete")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.parse_args()
    success = main()
    exit(0 if success else 1)
