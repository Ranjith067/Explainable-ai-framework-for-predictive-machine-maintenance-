"""
shap_explainability.py
======================
Phase 1 — Explainable AI Framework for Machine Failure Prediction
and Intelligent Fault Diagnosis

This module handles ONLY the explainability stage:
    1. Load the trained XGBoost model (models/xgboost_model.pkl)
    2. Load the processed test set (processed/X_test.csv)
    3. Compute SHAP values using a TreeExplainer
    4. Report which features matter most overall (global importance)
    5. Report which features drove EACH individual prediction (local explanations)
    6. Save plots and CSVs that the fault-diagnosis / dashboard stages will
       consume later

This is intentionally the "why did the model predict this" layer, sitting on
top of model_training.py. It does not retrain the model, does not diagnose a
specific fault type, and does not build any dashboard/UI — those are later
stages of this project.

Usage (command line):
    python shap_explainability.py

Usage (import):
    from shap_explainability import main
    main()
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe: only ever saving figures, never showing them
import matplotlib.pyplot as plt

import shap


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROCESSED_DIR = "processed"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

TOP_N_FACTORS = 3  # how many contributing factors to report per prediction


# ----------------------------------------------------------------------
# Helper: keep column names consistent with model_training.py
# ----------------------------------------------------------------------
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns so none contain '[', ']', or '<' — mirrors the same
    sanitization used in model_training.py before the model was trained
    (e.g. 'Air temperature [K]' -> 'Air temperature (K)').

    This MUST match model_training.py's sanitization exactly, since the
    saved model was fit on those sanitized column names. Duplicated here
    (rather than importing from model_training.py) to keep this module
    runnable on its own, with no dependency on that file's internals.
    """
    rename_map = {
        col: re.sub(r"\[", "(", re.sub(r"\]", ")", col)).replace("<", "lt")
        for col in df.columns
    }
    return df.rename(columns=rename_map)


# ----------------------------------------------------------------------
# 1. LOAD MODEL
# ----------------------------------------------------------------------
def load_model(models_dir: str = MODELS_DIR):
    """Loads the trained XGBoost model saved by model_training.py."""
    path = os.path.join(models_dir, "xgboost_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run model_training.py first to train and save the model."
        )
    model = joblib.load(path)
    print(f"[LOAD] Model loaded from '{path}'")
    return model


# ----------------------------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------------------------
def load_data(processed_dir: str = PROCESSED_DIR):
    """
    Loads the test features (and their true labels, for reference) produced
    by data_preprocessing.py, then sanitizes column names to match what the
    model was trained on.
    """
    x_test_path = os.path.join(processed_dir, "X_test.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")

    for path in [x_test_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run data_preprocessing.py first."
            )

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    X_test = sanitize_feature_names(X_test)

    print(f"[LOAD] X_test: {X_test.shape} | y_test: {y_test.shape}")
    return X_test, y_test


# ----------------------------------------------------------------------
# 3. COMPUTE SHAP VALUES
# ----------------------------------------------------------------------
def compute_shap_values(model, X: pd.DataFrame):
    """
    Builds a SHAP TreeExplainer for the trained XGBoost model and computes
    SHAP values for every row in X.

    TreeExplainer is used (rather than the generic/model-agnostic KernelExplainer)
    because it is exact and fast for tree-based models like XGBoost — no
    background sampling or approximation needed.

    Handles both SHAP API return shapes for robustness across versions:
      - Newer/typical XGBoost binary-classifier case: shap_values is a single
        2D array of shape (n_samples, n_features), one SHAP value per feature
        per row, for the positive ("failure") class log-odds output.
      - Some SHAP versions return a list of per-class arrays for classifiers;
        in that case index [1] (the positive/failure class) is used.
    """
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X)

    if isinstance(raw_shap_values, list):
        # older-style API: one array per class -> take the positive class
        shap_values = raw_shap_values[1]
    else:
        shap_values = raw_shap_values

    print(f"[SHAP] Computed SHAP values for {shap_values.shape[0]} rows, "
          f"{shap_values.shape[1]} features")

    return explainer, shap_values


# ----------------------------------------------------------------------
# 4. GLOBAL FEATURE IMPORTANCE (mean |SHAP value| across all test rows)
# ----------------------------------------------------------------------
def save_global_importance(shap_values: np.ndarray, X: pd.DataFrame,
                           outputs_dir: str = OUTPUTS_DIR) -> pd.DataFrame:
    """
    Ranks features by mean absolute SHAP value across the whole test set —
    i.e. how much each feature matters ON AVERAGE, regardless of direction.

    This is a different (and generally more trustworthy) view than XGBoost's
    built-in `feature_importances_` saved in model_training.py: SHAP importance
    reflects each feature's actual contribution to individual predictions,
    not just how often/how usefully a feature was split on during training.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap_value": mean_abs_shap,
    }).sort_values(by="mean_abs_shap_value", ascending=False).reset_index(drop=True)

    path = os.path.join(outputs_dir, "shap_feature_importance.csv")
    importance_df.to_csv(path, index=False)
    print(f"\n[SAVE] SHAP global feature importance written to '{path}'")

    return importance_df


# ----------------------------------------------------------------------
# 5. SUMMARY PLOTS (global visual explanation)
# ----------------------------------------------------------------------
def save_summary_plots(shap_values: np.ndarray, X: pd.DataFrame,
                       outputs_dir: str = OUTPUTS_DIR) -> None:
    """
    Saves two standard SHAP summary visualizations:
      - Beeswarm plot: shows each feature's SHAP value spread across all test
        rows (magnitude AND direction — whether high/low feature values push
        the prediction toward "failure" or "no failure").
      - Bar plot: a simpler, report-friendly ranking of mean |SHAP value|
        per feature, easier to read at a glance than the beeswarm plot.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    # Beeswarm (dot) summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    beeswarm_path = os.path.join(outputs_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVE] SHAP beeswarm summary plot written to '{beeswarm_path}'")

    # Bar summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    bar_path = os.path.join(outputs_dir, "shap_summary_bar.png")
    plt.savefig(bar_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVE] SHAP bar summary plot written to '{bar_path}'")


# ----------------------------------------------------------------------
# 6. PER-INSTANCE (LOCAL) EXPLANATIONS
# ----------------------------------------------------------------------
def save_instance_explanations(shap_values: np.ndarray, X: pd.DataFrame,
                               y_test: pd.Series, model,
                               top_n: int = TOP_N_FACTORS,
                               outputs_dir: str = OUTPUTS_DIR) -> pd.DataFrame:
    """
    For every row in the test set, records the top-N features that
    contributed most to THAT specific prediction (by absolute SHAP value),
    along with each one's signed SHAP value (positive = pushed toward
    'failure', negative = pushed toward 'no failure').

    This is the bridge to the next stage of the project: intelligent fault
    diagnosis will later use these per-row top contributing factors to infer
    which fault type is likely, instead of only knowing failure/no-failure.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    feature_names = X.columns.to_numpy()
    predicted_failure = model.predict(X)
    failure_probability = model.predict_proba(X)[:, 1]

    records = []
    for i in range(shap_values.shape[0]):
        row_shap = shap_values[i]
        # indices of the top-N contributing features by |SHAP value|, descending
        top_idx = np.argsort(-np.abs(row_shap))[:top_n]

        record = {
            "actual_failure": y_test.iloc[i],
            "predicted_failure": predicted_failure[i],
            "failure_probability": failure_probability[i],
        }
        for rank, idx in enumerate(top_idx, start=1):
            record[f"top{rank}_feature"] = feature_names[idx]
            record[f"top{rank}_shap_value"] = row_shap[idx]

        records.append(record)

    explanations_df = pd.DataFrame(records)

    path = os.path.join(outputs_dir, "shap_explanations.csv")
    explanations_df.to_csv(path, index=False)
    print(f"[SAVE] Per-instance SHAP explanations written to '{path}'")

    return explanations_df


# ----------------------------------------------------------------------
# ORCHESTRATOR
# ----------------------------------------------------------------------
def main():
    """Runs the full SHAP explainability pipeline end-to-end."""
    model = load_model()
    X_test, y_test = load_data()

    explainer, shap_values = compute_shap_values(model, X_test)

    save_global_importance(shap_values, X_test)
    save_summary_plots(shap_values, X_test)
    save_instance_explanations(shap_values, X_test, y_test, model)

    print("\n[DONE] SHAP explainability pipeline complete.")


if __name__ == "__main__":
    main()