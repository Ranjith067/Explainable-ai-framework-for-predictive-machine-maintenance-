"""
feature_selection.py
======================
Phase 1 — Explainable AI Framework for Machine Failure Prediction
and Intelligent Fault Diagnosis

This module runs BETWEEN data_preprocessing.py and model_training.py:

    data_preprocessing.py -> processed/X_train.csv, X_test.csv, ...
    feature_selection.py  -> (drops low-value columns from those same files)
    model_training.py     -> reads processed/X_train.csv, X_test.csv as usual

It does NOT retrain a model and does NOT change data_preprocessing.py or
model_training.py -- it overwrites processed/X_train.csv and X_test.csv
in place, keeping the exact same file names/locations, so every downstream
script keeps working unmodified regardless of which features get dropped.

Two combined techniques are used:
    1. Correlation-based redundancy filtering: if two features are very
       highly correlated with each other (e.g. air temperature and process
       temperature often move together), one of the pair is dropped, since
       it adds little new information beyond the other.
    2. Model-based importance filtering: a quick XGBoost model is fit on
       the (already redundancy-filtered) training data, and any feature
       contributing almost nothing to predictions (importance below a
       threshold) is dropped.

A backup of the original, unfiltered processed/ files is written first
(processed/backup_before_feature_selection/), so this step is always
reversible.

Usage (command line):
    python feature_selection.py

Usage (import):
    from feature_selection import main
    main()
"""

import os
import shutil
import pandas as pd
import numpy as np

from xgboost import XGBClassifier


PROCESSED_DIR = "processed"
BACKUP_DIR = os.path.join(PROCESSED_DIR, "backup_before_feature_selection")
OUTPUTS_DIR = "outputs"

CORRELATION_THRESHOLD = 0.90   # features correlated above this are considered redundant
IMPORTANCE_THRESHOLD = 0.01    # features contributing less than this (of total) are dropped
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. LOAD & BACKUP
# ----------------------------------------------------------------------
def load_and_backup(processed_dir: str = PROCESSED_DIR):
    """Loads the processed train/test files and makes a one-time backup
    of the originals before any columns are dropped, so this step can
    always be undone by restoring from the backup folder."""
    x_train_path = os.path.join(processed_dir, "X_train.csv")
    x_test_path = os.path.join(processed_dir, "X_test.csv")

    for path in [x_train_path, x_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run data_preprocessing.py first."
            )

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns")

    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        shutil.copy(x_train_path, os.path.join(BACKUP_DIR, "X_train.csv"))
        shutil.copy(x_test_path, os.path.join(BACKUP_DIR, "X_test.csv"))
        print(f"[BACKUP] Original X_train/X_test backed up to '{BACKUP_DIR}/' "
              f"(only created once; safe to run this script again).")
    else:
        print(f"[BACKUP] Backup already exists at '{BACKUP_DIR}/' -- not overwritten.")

    print(f"[LOAD] X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train


# ----------------------------------------------------------------------
# 2. CORRELATION-BASED REDUNDANCY FILTER
# ----------------------------------------------------------------------
def drop_correlated_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             threshold: float = CORRELATION_THRESHOLD):
    """
    For every pair of numeric features correlated above `threshold`, drops
    the second one in the pair. Correlation is computed on X_train ONLY
    (not X_test), consistent with fitting every data-derived decision on
    training data alone.
    """
    numeric_cols = X_train.select_dtypes(include="number").columns
    corr_matrix = X_train[numeric_cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        print(f"\n[CORRELATION] Dropping {len(to_drop)} redundant feature(s) "
              f"(correlation > {threshold}): {to_drop}")
        for col in to_drop:
            correlated_with = upper.index[upper[col] > threshold].tolist()
            print(f"    '{col}' is highly correlated with {correlated_with}")
    else:
        print(f"\n[CORRELATION] No feature pairs exceeded the {threshold} threshold "
              f"-- nothing dropped at this step.")

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    return X_train, X_test, to_drop


# ----------------------------------------------------------------------
# 3. MODEL-BASED IMPORTANCE FILTER
# ----------------------------------------------------------------------
def drop_low_importance_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                 y_train: pd.Series,
                                 threshold: float = IMPORTANCE_THRESHOLD):
    """
    Fits a quick XGBoost model on X_train (with column names sanitized the
    same way model_training.py does, since XGBoost rejects '[', ']', '<' in
    feature names) purely to rank feature importance, then drops any
    feature contributing less than `threshold` of total importance.

    This is a lightweight, disposable model used only to rank features --
    it is NOT the model saved/evaluated by model_training.py, which will
    fit its own model afterward on whatever features survive this step.
    """
    import re

    def sanitize(cols):
        return [re.sub(r"\[", "(", re.sub(r"\]", ")", c)).replace("<", "lt") for c in cols]

    X_train_sanitized = X_train.copy()
    X_train_sanitized.columns = sanitize(X_train.columns)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    ranker = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
    )
    ranker.fit(X_train_sanitized, y_train)

    importance = pd.Series(ranker.feature_importances_, index=X_train.columns)
    importance = importance / importance.sum()  # normalize to proportions

    to_drop = importance[importance < threshold].index.tolist()

    print(f"\n[IMPORTANCE] Feature importance (proportion of total):")
    print(importance.sort_values(ascending=False).round(4))

    if to_drop:
        print(f"\n[IMPORTANCE] Dropping {len(to_drop)} low-importance feature(s) "
              f"(< {threshold}): {to_drop}")
    else:
        print(f"\n[IMPORTANCE] All remaining features are above the {threshold} "
              f"importance threshold -- nothing dropped at this step.")

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    return X_train, X_test, to_drop, importance


# ----------------------------------------------------------------------
# 4. SAVE
# ----------------------------------------------------------------------
def save_selected_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           dropped_correlation: list, dropped_importance: list,
                           importance: pd.Series,
                           processed_dir: str = PROCESSED_DIR,
                           outputs_dir: str = OUTPUTS_DIR) -> None:
    """
    Overwrites processed/X_train.csv and X_test.csv with the reduced
    feature set, and writes a log of what was selected/dropped and why,
    for the report.
    """
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    print(f"\n[SAVE] processed/X_train.csv and X_test.csv overwritten "
          f"with {X_train.shape[1]} selected feature(s): {list(X_train.columns)}")

    os.makedirs(outputs_dir, exist_ok=True)
    log_path = os.path.join(outputs_dir, "feature_selection_log.csv")

    log_rows = []
    for col in importance.index:
        if col in dropped_correlation:
            reason = "dropped: highly correlated with another feature"
        elif col in dropped_importance:
            reason = "dropped: low XGBoost importance"
        elif col in X_train.columns:
            reason = "kept"
        else:
            reason = "dropped"
        log_rows.append({
            "feature": col,
            "importance_proportion": round(importance[col], 4),
            "status": reason,
        })

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"[SAVE] Feature selection log written to '{log_path}'")


# ----------------------------------------------------------------------
# ORCHESTRATOR
# ----------------------------------------------------------------------
def main():
    X_train, X_test, y_train = load_and_backup()

    X_train, X_test, dropped_corr = drop_correlated_features(X_train, X_test)
    X_train, X_test, dropped_imp, importance = drop_low_importance_features(
        X_train, X_test, y_train
    )

    save_selected_features(X_train, X_test, dropped_corr, dropped_imp, importance)

    print(f"\n[DONE] Feature selection complete. "
          f"{len(dropped_corr) + len(dropped_imp)} feature(s) dropped total.")


if __name__ == "__main__":
    main()