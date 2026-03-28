"""
data_validation.py
------------------
Runs data quality checks on the combined wine quality dataset before training.
Logs validation results as MLflow tags on the active run.

Checks performed:
  - Missing value detection
  - Feature range validation (physicochemical bounds)
  - Class balance check
  - Duplicate row detection
"""

import logging
import sys

import mlflow
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Expected physicochemical feature bounds (from UCI documentation) ───────────
FEATURE_BOUNDS = {
    "fixed acidity": (3.0, 16.0),
    "volatile acidity": (0.05, 2.0),
    "citric acid": (0.0, 1.7),
    "residual sugar": (0.5, 70.0),
    "chlorides": (0.01, 0.7),
    "free sulfur dioxide": (1.0, 290.0),
    "total sulfur dioxide": (6.0, 450.0),
    "density": (0.985, 1.005),
    "pH": (2.7, 4.1),
    "sulphates": (0.2, 2.1),
    "alcohol": (7.0, 15.5),
}


def load_combined_dataset(red_path: str, white_path: str) -> pd.DataFrame:
    """Load and combine red and white wine datasets with a wine_type feature."""
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")
    red["wine_type"] = 0       # 0 = red
    white["wine_type"] = 1     # 1 = white
    combined = pd.concat([red, white], ignore_index=True)
    logger.info("Loaded %d red + %d white = %d total rows", len(red), len(white), len(combined))
    return combined


def check_missing_values(df: pd.DataFrame) -> dict:
    """Return per-column missing value counts."""
    missing = df.isnull().sum()
    result = {col: int(count) for col, count in missing.items() if count > 0}
    if result:
        logger.warning("Missing values detected: %s", result)
    else:
        logger.info("No missing values found.")
    return result


def check_feature_ranges(df: pd.DataFrame) -> dict:
    """Check that feature values fall within expected physicochemical bounds."""
    violations = {}
    for feature, (low, high) in FEATURE_BOUNDS.items():
        if feature not in df.columns:
            continue
        out_of_range = df[(df[feature] < low) | (df[feature] > high)]
        count = len(out_of_range)
        if count > 0:
            violations[feature] = count
            logger.warning("%d rows out of range for '%s' [%.2f, %.2f]", count, feature, low, high)
    if not violations:
        logger.info("All feature ranges are within expected bounds.")
    return violations


def check_class_balance(df: pd.DataFrame, target_col: str = "quality") -> dict:
    """Log class distribution and flag severe imbalance (< 5% for any class)."""
    counts = df[target_col].value_counts().sort_index()
    total = len(df)
    balance = {str(k): round(v / total * 100, 2) for k, v in counts.items()}
    minority_pct = min(balance.values())
    imbalanced = minority_pct < 5.0
    if imbalanced:
        logger.warning("Class imbalance detected — minority class is %.1f%% of data.", minority_pct)
    else:
        logger.info("Class balance OK — minority class is %.1f%%.", minority_pct)
    return {"class_distribution_pct": balance, "imbalanced": imbalanced}


def check_duplicates(df: pd.DataFrame) -> int:
    """Return count of duplicate rows."""
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        logger.warning("%d duplicate rows detected.", n_dupes)
    else:
        logger.info("No duplicate rows found.")
    return int(n_dupes)


def run_validation(red_path: str, white_path: str) -> bool:
    """
    Run all validation checks and log results as MLflow tags.
    Returns True if dataset passes, False if critical issues found.
    """
    df = load_combined_dataset(red_path, white_path)

    missing = check_missing_values(df)
    range_violations = check_feature_ranges(df)
    balance_info = check_class_balance(df)
    n_dupes = check_duplicates(df)

    # ── Log to MLflow if a run is active ──────────────────────────────────────
    try:
        mlflow.set_tag("validation.total_rows", len(df))
        mlflow.set_tag("validation.missing_columns", len(missing))
        mlflow.set_tag("validation.range_violations", len(range_violations))
        mlflow.set_tag("validation.duplicate_rows", n_dupes)
        mlflow.set_tag("validation.class_imbalanced", balance_info["imbalanced"])
        mlflow.set_tag("validation.passed", len(missing) == 0 and len(range_violations) == 0)
        logger.info("Validation results logged to MLflow.")
    except Exception:
        logger.info("No active MLflow run — skipping tag logging.")

    # ── Critical failure conditions ────────────────────────────────────────────
    if missing:
        logger.error("VALIDATION FAILED: Missing values in columns: %s", list(missing.keys()))
        return False

    logger.info("Dataset validation PASSED.")
    return True


if __name__ == "__main__":
    red = sys.argv[1] if len(sys.argv) > 1 else "data/winequality-red.csv"
    white = sys.argv[2] if len(sys.argv) > 2 else "data/winequality-white.csv"
    passed = run_validation(red, white)
    sys.exit(0 if passed else 1)
