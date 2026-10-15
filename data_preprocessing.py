"""
data_preprocessing.py
======================
Phase 1 — Explainable AI Framework for Machine Failure Prediction
and Intelligent Fault Diagnosis

This module handles ONLY the data-preparation stage of the pipeline:
    1. Load the AI4I 2020 Predictive Maintenance Dataset
    2. Inspect its structure (shape, dtypes, missing values, duplicates)
    3. Identify target variable(s)
    4. Remove duplicate rows and identifier columns
    5. Check class imbalance
    6. Split the data into training and testing sets
    7. Impute missing values (fit on train only, applied to test)
    8. One-hot encode categorical variables (fit on train only, applied to test)

No model training or dashboard code lives here — this module's only
job is to turn the raw CSV into clean, ready-to-model arrays that a
later "model_training.py" module can consume.

Usage (command line):
    python data_preprocessing.py --input ai4i2020.csv --output_dir processed

Usage (import):
    from data_preprocessing import run_pipeline
    data = run_pipeline("ai4i2020.csv", output_dir="processed")
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ----------------------------------------------------------------------
# Column reference for the AI4I 2020 Predictive Maintenance Dataset
# ----------------------------------------------------------------------
# UDI                        -> row index / unique identifier (not predictive)
# Product ID                 -> product serial number (not predictive)
# Type                       -> product quality variant: L / M / H (categorical, NOMINAL)
# Air temperature [K]        -> numeric sensor reading
# Process temperature [K]    -> numeric sensor reading
# Rotational speed [rpm]     -> numeric sensor reading
# Torque [Nm]                -> numeric sensor reading
# Tool wear [min]            -> numeric sensor reading
# Machine failure             -> PRIMARY target (binary: 0 = no failure, 1 = failure)
# TWF, HDF, PWF, OSF, RNF     -> SECONDARY targets: which failure mode fired
#                                (Tool Wear, Heat Dissipation, Power, Overstrain,
#                                Random Failure). Used later for fault-type diagnosis.

ID_COLUMNS = ["UDI", "Product ID"]
CATEGORICAL_COLUMNS = ["Type"]
PRIMARY_TARGET = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame with a clear error if the file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download the AI4I 2020 Predictive "
            f"Maintenance Dataset (ai4i2020.csv) and pass its path with --input."
        )
    df = pd.read_csv(path)
    print(f"[LOAD] Loaded '{path}' -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ----------------------------------------------------------------------
# 2. INSPECT
# ----------------------------------------------------------------------
def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a structural summary of the dataset: shape, dtypes, missing values,
    duplicate rows, and basic statistics. This step never modifies the data —
    it only reports what is there, so later steps are justified by evidence
    rather than assumptions about the dataset.
    """
    print("\n[INSPECT] Shape:", df.shape)

    print("\n[INSPECT] Column dtypes:")
    print(df.dtypes)

    print("\n[INSPECT] Missing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    dup_count = df.duplicated().sum()
    print(f"\n[INSPECT] Duplicate rows: {dup_count}")

    print("\n[INSPECT] Summary statistics (numeric columns):")
    print(df.describe().T)

    if "Type" in df.columns:
        print("\n[INSPECT] Categorical column 'Type' value counts:")
        print(df["Type"].value_counts())


# ----------------------------------------------------------------------
# 3. IDENTIFY TARGETS  (kept as a pure lookup so target names live in one place)
# ----------------------------------------------------------------------
def identify_targets(df: pd.DataFrame) -> dict:
    """
    Confirm which columns exist to serve as prediction targets.

    Returns a dict:
        {
            "primary": "Machine failure",
            "failure_modes": ["TWF", "HDF", "PWF", "OSF", "RNF"]  # present ones only
        }
    """
    if PRIMARY_TARGET not in df.columns:
        raise KeyError(f"Expected primary target column '{PRIMARY_TARGET}' not found.")

    present_modes = [c for c in FAILURE_MODE_COLUMNS if c in df.columns]

    print(f"\n[TARGETS] Primary target: '{PRIMARY_TARGET}' (binary failure / no-failure)")
    print(f"[TARGETS] Failure-mode columns available for fault diagnosis: {present_modes}")

    return {"primary": PRIMARY_TARGET, "failure_modes": present_modes}


# ----------------------------------------------------------------------
# 4. REMOVE DUPLICATES & IDENTIFIER COLUMNS
# ----------------------------------------------------------------------
def remove_duplicates_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drops exact duplicate rows.
    - Drops identifier columns (UDI, Product ID) since they carry no predictive
      signal and would otherwise leak a unique row index into the model.

    NOTE: This step deliberately does NOT touch missing values. Duplicate/ID
    removal is safe to do on the full dataset before splitting because it does
    not compute any statistic from the data (no mean/median/mode is learned
    here), so it cannot leak information from test rows into training.
    Missing-value imputation, which DOES learn a statistic, is handled later
    in `impute_missing()` — after the train/test split, fitted on train only.
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    print(f"\n[CLEAN] Dropped {before - len(df)} duplicate rows "
          f"({before} -> {len(df)})")

    existing_id_cols = [c for c in ID_COLUMNS if c in df.columns]
    if existing_id_cols:
        df = df.drop(columns=existing_id_cols)
        print(f"[CLEAN] Dropped identifier columns: {existing_id_cols}")

    return df


# ----------------------------------------------------------------------
# 5. CHECK CLASS IMBALANCE
# ----------------------------------------------------------------------
def check_class_imbalance(df: pd.DataFrame, target_col: str = PRIMARY_TARGET) -> pd.Series:
    """
    Prints class counts and percentages for the target column and returns
    the counts as a Series. The AI4I 2020 dataset is known to be highly
    imbalanced (failures are rare events), which is why the train/test
    split uses stratification and why the model-training stage should use
    class weighting or resampling.

    This is purely diagnostic (no statistic is fitted/reused elsewhere), so
    it is safe to run on the full dataset before splitting.
    """
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100

    print(f"\n[IMBALANCE] Class distribution for '{target_col}':")
    for cls in counts.index:
        print(f"  Class {cls}: {counts[cls]} samples ({percentages[cls]:.2f}%)")

    ratio = counts.max() / counts.min()
    print(f"[IMBALANCE] Majority:minority ratio ≈ {ratio:.1f}:1")
    if ratio > 5:
        print("[IMBALANCE] Dataset is significantly imbalanced — "
              "use stratified splitting and address this at model-training time "
              "(e.g. class_weight='balanced', SMOTE, or threshold tuning).")

    return counts


# ----------------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    target_col: str = PRIMARY_TARGET,
    drop_cols: list = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Splits features/target into stratified train and test sets.

    Stratification is used (not a plain random split) because the target
    is imbalanced — without it, the test set could end up with too few
    (or zero) failure examples to evaluate the model meaningfully.

    `drop_cols` lets the caller exclude the failure-mode columns (TWF, HDF,
    PWF, OSF, RNF) from X, since those are outcome-adjacent labels, not
    input features.

    This split happens BEFORE imputation and encoding are fitted, so that
    those later steps can be fitted on X_train only (see `impute_missing`
    and `encode_categorical`) and prevent data leakage from the test set.
    """
    drop_cols = drop_cols or []
    feature_cols = [c for c in df.columns if c != target_col and c not in drop_cols]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"\n[SPLIT] Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"[SPLIT] Features used ({len(feature_cols)}): {feature_cols}")
    print(f"[SPLIT] Train class balance:\n{y_train.value_counts(normalize=True) * 100}")
    print(f"[SPLIT] Test class balance:\n{y_test.value_counts(normalize=True) * 100}")

    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# 7. IMPUTE MISSING VALUES (leak-free: fit on train only)
# ----------------------------------------------------------------------
def impute_missing(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fills missing values using sklearn's SimpleImputer:
        - numeric columns  -> median
        - categorical columns ('Type') -> most frequent value (mode)

    CRITICAL: the imputer is `fit()` on X_train ONLY, then `transform()`-ed
    on both X_train and X_test. If the median/mode were computed on the full
    dataset (train + test combined) before splitting, information about the
    test set's distribution would leak into training — the model would be
    evaluated on data it was indirectly informed by. Fitting on train only
    keeps the test set a genuinely unseen sample.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include="number").columns
    categorical_cols = X_train.select_dtypes(exclude="number").columns

    if len(numeric_cols) > 0 and X_train[numeric_cols].isnull().any().any():
        num_imputer = SimpleImputer(strategy="median")
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
        print(f"\n[IMPUTE] Numeric columns imputed with TRAIN median: {list(numeric_cols)}")

    if len(categorical_cols) > 0 and X_train[categorical_cols].isnull().any().any():
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        print(f"[IMPUTE] Categorical columns imputed with TRAIN mode: {list(categorical_cols)}")

    remaining_train_na = X_train.isnull().sum().sum()
    remaining_test_na = X_test.isnull().sum().sum()
    if remaining_train_na == 0 and remaining_test_na == 0:
        print("[IMPUTE] No missing values remain in X_train or X_test.")

    return X_train, X_test


# ----------------------------------------------------------------------
# 8. ENCODE CATEGORICAL VARIABLES (leak-free: fit on train only)
# ----------------------------------------------------------------------
def encode_categorical(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    One-hot encodes the 'Type' column (product quality variant: L / M / H).

    'Type' is treated as NOMINAL, not ordinal: L/M/H are distinct product
    variants rather than a measurable, evenly-spaced quantity, so mapping
    them to 0/1/2 would falsely imply "H is numerically greater than L" to
    the model. One-hot encoding avoids inventing an order that isn't there.

    The encoder is `fit()` on X_train ONLY (learns which categories exist),
    then used to `transform()` both X_train and X_test. This guarantees the
    train and test sets end up with identical dummy columns, and any category
    that might appear only in test (unlikely here, but possible in general)
    is safely handled via `handle_unknown="ignore"` instead of crashing.
    """
    if "Type" not in X_train.columns:
        return X_train, X_test

    X_train = X_train.copy()
    X_test = X_test.copy()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_encoded = encoder.fit_transform(X_train[["Type"]])
    test_encoded = encoder.transform(X_test[["Type"]])
    encoded_cols = encoder.get_feature_names_out(["Type"])

    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=["Type"]), train_encoded_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=["Type"]), test_encoded_df], axis=1)

    print(f"\n[ENCODE] One-hot encoded 'Type' (fit on train only) -> {list(encoded_cols)}")

    return X_train, X_test


# ----------------------------------------------------------------------
# 9. SAVE PROCESSED OUTPUTS
# ----------------------------------------------------------------------
def save_outputs(X_train, X_test, y_train, y_test, output_dir: str) -> None:
    """Writes the four resulting splits to CSV so the model-training module
    can load them independently, without repeating the preprocessing steps."""
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"\n[SAVE] Processed train/test files written to '{output_dir}/'")


# ----------------------------------------------------------------------
# ORCHESTRATOR
# ----------------------------------------------------------------------
def run_pipeline(input_path: str, output_dir: str = "processed") -> dict:
    """
    Runs the full preprocessing pipeline end-to-end and returns the split data.

    Order matters here for leakage prevention:
        load -> inspect -> identify targets -> remove duplicates/IDs
        -> check imbalance -> SPLIT -> impute (fit on train) -> encode (fit on train)
        -> save

    Imputation and encoding are deliberately placed AFTER the split so that
    both are fitted on X_train only and merely applied to X_test.
    """
    df = load_data(input_path)
    inspect_data(df)
    targets = identify_targets(df)

    df = remove_duplicates_and_ids(df)

    check_class_imbalance(df, target_col=targets["primary"])

    # Failure-mode columns are excluded from X: they are only useful later,
    # for diagnosing WHICH fault occurred once a failure has been predicted.
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=targets["primary"],
        drop_cols=targets["failure_modes"],
    )

    X_train, X_test = impute_missing(X_train, X_test)
    X_train, X_test = encode_categorical(X_train, X_test)

    save_outputs(X_train, X_test, y_train, y_test, output_dir)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "targets": targets,
    }


# ----------------------------------------------------------------------
# CLI ENTRY POINT
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the AI4I 2020 Predictive Maintenance Dataset "
                    "for the Explainable AI failure-prediction framework."
    )
    parser.add_argument("--input", required=True, help="Path to ai4i2020.csv")
    parser.add_argument("--output_dir", default="processed",
                        help="Directory to write the processed train/test CSVs")
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir)


if __name__ == "__main__":
    main()
