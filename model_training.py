"""
model_training.py
======================
Phase 1 — Explainable AI Framework for Machine Failure Prediction
and Intelligent Fault Diagnosis
This module handles ONLY the model-training and evaluation stage:
    1. Load the processed train/test CSVs produced by data_preprocessing.py
    2. Train an XGBoost classifier to predict 'Machine failure'
    3. Handle class imbalance via a data-derived scale_pos_weight
    4. Evaluate the model with standard classification metrics
    5. Save feature importance, test predictions, and the trained model
No SHAP, no fault-type diagnosis, no dashboard, no API/UI code lives here.
This module assumes data_preprocessing.py has already been run and that
processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv exist.
Usage (command line):
    python model_training.py
Usage (import):
    from model_training import main
    main()
"""
import os
import re
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROCESSED_DIR = "processed"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
TARGET_COLUMN = "Machine failure"
RANDOM_STATE = 42
# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns so none contain '[', ']', or '<'.
    The AI4I 2020 dataset has column names like 'Air temperature [K]' and
    'Torque [Nm]'. Modern XGBoost's sklearn API validates feature names and
    raises `ValueError: feature_names must be string, and may not contain
    [, ] or <` if any of these characters are present. This is unrelated to
    data_preprocessing.py (which correctly leaves the original column names
    untouched) — it is purely an XGBoost input requirement, so the rename
    happens here, right after loading, and only affects this module.
    '[' and ']' are replaced with '(' and ')' so the columns stay readable
    (e.g. 'Air temperature [K]' -> 'Air temperature (K)').
    """
    rename_map = {
        col: re.sub(r"\[", "(", re.sub(r"\]", ")", col)).replace("<", "lt")
        for col in df.columns
    }
    return df.rename(columns=rename_map)
def load_data(processed_dir: str = PROCESSED_DIR):
    """
    Loads the four CSV files produced by data_preprocessing.py.
    y_train.csv / y_test.csv are read and squeezed into 1-D Series, since
    train_test_split originally saved them as single-column CSVs (they load
    back as a DataFrame with one column otherwise, which most sklearn/XGBoost
    functions do not accept directly as a target).
    X_train / X_test column names are sanitized immediately after loading
    (see `sanitize_feature_names`) so XGBoost does not reject them later.
    """
    x_train_path = os.path.join(processed_dir, "X_train.csv")
    x_test_path = os.path.join(processed_dir, "X_test.csv")
    y_train_path = os.path.join(processed_dir, "y_train.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")
    for path in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run data_preprocessing.py first to "
                f"generate the processed train/test files."
            )
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")
    X_train = sanitize_feature_names(X_train)
    X_test = sanitize_feature_names(X_test)
    print(f"[LOAD] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[LOAD] y_train: {y_train.shape} | y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test
# ----------------------------------------------------------------------
# 2. TRAIN MODEL
# ----------------------------------------------------------------------
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Trains an XGBoost classifier to predict 'Machine failure'.
    Class imbalance is handled with `scale_pos_weight`, computed directly
    from the training data as (negative class count / positive class count)
    rather than a hardcoded value. This is the standard XGBoost approach for
    binary imbalance: it up-weights errors on the minority (failure) class
    during training without altering the actual data via resampling.
    scale_pos_weight is calculated from y_train ONLY (not y_test or the full
    dataset), consistent with fitting everything solely on training data.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"\n[TRAIN] Training class counts -> negative: {neg_count}, positive: {pos_count}")
    print(f"[TRAIN] Computed scale_pos_weight = {scale_pos_weight:.2f}")
    # Notes on XGBoost API compatibility (checked against current XGBoost 2.x /
    # scikit-learn 1.x releases):
    #   - `use_label_encoder` is removed in modern XGBoost and must NOT be passed.
    #   - `eval_metric` should be set in the constructor (not passed to .fit()),
    #     since passing it to .fit() is deprecated/unsupported in current releases.
    #   - No early stopping / eval_set is used here, since that would require
    #     carving out a validation split — out of scope for this module.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[TRAIN] XGBoost model training complete.")
    return model
# ----------------------------------------------------------------------
# 3. EVALUATE MODEL
# ----------------------------------------------------------------------
def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the trained model on the held-out test set and prints all
    standard classification metrics. Predictions and probabilities are
    computed once here and returned, so the same values can be reused for
    saving predictions later without recomputing them.
    No thresholds, sampling, or metrics are tuned to inflate performance —
    this reports the model's actual behavior on unseen data as-is.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (failure)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\n" + "=" * 60)
    print("MODEL EVALUATION — Machine Failure Prediction (XGBoost)")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print("               Predicted 0   Predicted 1")
    print(f"Actual 0       {cm[0][0]:<13}{cm[0][1]}")
    print(f"Actual 1       {cm[1][0]:<13}{cm[1][1]}")
    print("\nClassification Report:")
    print(report)
    print("=" * 60)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }
    return y_pred, y_proba, metrics
# ----------------------------------------------------------------------
# 4. SAVE FEATURE IMPORTANCE
# ----------------------------------------------------------------------
def save_feature_importance(model: XGBClassifier, X_train: pd.DataFrame,
                            outputs_dir: str = OUTPUTS_DIR) -> pd.DataFrame:
    """
    Extracts XGBoost's built-in feature importance (gain-based, the default
    `feature_importances_` attribute in the sklearn API) and saves it sorted
    from most to least important. This is a quick, model-native importance
    view — separate from and simpler than the SHAP-based explanations
    planned for a later stage of this project.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    path = os.path.join(outputs_dir, "feature_importance.csv")
    importance_df.to_csv(path, index=False)
    print(f"\n[SAVE] Feature importance written to '{path}'")
    return importance_df
# ----------------------------------------------------------------------
# 5. SAVE PREDICTIONS
# ----------------------------------------------------------------------
def save_predictions(X_test: pd.DataFrame, y_test: pd.Series, y_pred, y_proba,
                     outputs_dir: str = OUTPUTS_DIR) -> pd.DataFrame:
    """
    Saves test-set predictions alongside actual labels and predicted failure
    probability, indexed to match X_test rows, so results can be traced back
    to the corresponding test samples for later fault-diagnosis work.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    predictions_df = X_test.copy().reset_index(drop=True)
    predictions_df["actual_failure"] = y_test.reset_index(drop=True)
    predictions_df["predicted_failure"] = y_pred
    predictions_df["failure_probability"] = y_proba
    path = os.path.join(outputs_dir, "predictions.csv")
    predictions_df.to_csv(path, index=False)
    print(f"[SAVE] Test predictions written to '{path}'")
    return predictions_df
# ----------------------------------------------------------------------
# 6. SAVE MODEL
# ----------------------------------------------------------------------
def save_model(model: XGBClassifier, models_dir: str = MODELS_DIR) -> str:
    """Saves the trained XGBoost model to disk using joblib."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, "xgboost_model.pkl")
    joblib.dump(model, path)
    print(f"[SAVE] Trained model written to '{path}'")
    return path
# ----------------------------------------------------------------------
# ORCHESTRATOR
# ----------------------------------------------------------------------
def main():
    """Runs the full training + evaluation pipeline end-to-end."""
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    y_pred, y_proba, metrics = evaluate_model(model, X_test, y_test)
    save_feature_importance(model, X_train)
    save_predictions(X_test, y_test, y_pred, y_proba)
    save_model(model)
    print("\n[DONE] Model training and evaluation pipeline complete.")
if __name__ == "__main__":
    main()