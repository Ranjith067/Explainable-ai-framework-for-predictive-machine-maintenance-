"""
fault_diagnosis.py
======================
Phase 1 — Explainable AI Framework for Machine Failure Prediction
and Intelligent Fault Diagnosis

This module handles the FAULT-TYPE diagnosis stage: given that the model has
predicted a failure, WHICH kind of failure is it likely to be?

It combines two complementary sources of evidence:
    1. Data-driven: the top contributing features from SHAP
       (outputs/shap_explanations.csv, produced by shap_explainability.py)
    2. Domain knowledge: the AI4I 2020 dataset's own published failure-mode
       formulas (Matzka, 2020), reimplemented here as diagnostic rules

Neither source alone is enough: SHAP tells you which sensor readings mattered
most for a prediction, but not which named fault mode that corresponds to.
The domain rules tell you exactly which fault type would fire under the
dataset's original generation logic, but say nothing about model confidence
or which feature the MODEL itself relied on. Combining both gives an
interpretable diagnosis with supporting evidence from both angles.

Fault-mode formulas used (from the AI4I 2020 dataset documentation):
    TWF (Tool Wear Failure):      tool wear time is between 200-240 min
    HDF (Heat Dissipation Failure): (process temp - air temp) < 8.6 K
                                    AND rotational speed < 1380 rpm
    PWF (Power Failure):           power = torque (Nm) * rotational speed (rad/s)
                                    fails if power < 3500 W or > 9000 W
    OSF (Overstrain Failure):      tool wear (min) * torque (Nm) exceeds
                                    11,000 (Type L) / 12,000 (Type M) / 13,000 (Type H)
    RNF (Random Failure):          0.1% random chance, independent of all
                                    features -- by definition NOT diagnosable
                                    from sensor readings

This module does NOT retrain anything and does NOT build the dashboard —
those are separate stages of this project.

Usage (command line):
    python fault_diagnosis.py

Usage (import):
    from fault_diagnosis import main
    main()
"""

import os
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROCESSED_DIR = "processed"
OUTPUTS_DIR = "outputs"

# Original (un-sanitized) AI4I 2020 column names, as written by data_preprocessing.py
COL_AIR_TEMP = "Air temperature [K]"
COL_PROCESS_TEMP = "Process temperature [K]"
COL_ROT_SPEED = "Rotational speed [rpm]"
COL_TORQUE = "Torque [Nm]"
COL_TOOL_WEAR = "Tool wear [min]"

OSF_THRESHOLDS = {"Type_L": 11000, "Type_M": 12000, "Type_H": 13000}


# ----------------------------------------------------------------------
# 1. LOAD SHAP EXPLANATIONS
# ----------------------------------------------------------------------
def load_shap_explanations(outputs_dir: str = OUTPUTS_DIR) -> pd.DataFrame:
    """
    Loads the per-instance SHAP explanations produced by shap_explainability.py
    (actual/predicted failure, failure probability, and the top contributing
    features for each test-set row).
    """
    path = os.path.join(outputs_dir, "shap_explanations.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run shap_explainability.py first."
        )
    df = pd.read_csv(path)
    print(f"[LOAD] SHAP explanations: {df.shape}")
    return df


# ----------------------------------------------------------------------
# 2. LOAD TEST FEATURES (original, un-sanitized column names)
# ----------------------------------------------------------------------
def load_test_features(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """
    Loads processed/X_test.csv directly (NOT through model_training.py's
    sanitized names), because the diagnostic rules below need the original
    physical units and column names to evaluate the AI4I 2020 formulas
    exactly as documented (e.g. 'Torque [Nm]', 'Tool wear [min]').

    Row order here matches shap_explanations.csv row-for-row, since both
    trace back to the same processed/X_test.csv written once by
    data_preprocessing.py and never reordered afterward.
    """
    path = os.path.join(processed_dir, "X_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run data_preprocessing.py first."
        )
    df = pd.read_csv(path)
    print(f"[LOAD] Test features: {df.shape}")
    return df


# ----------------------------------------------------------------------
# 3. APPLY DIAGNOSTIC RULES
# ----------------------------------------------------------------------
def apply_diagnostic_rules(X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the AI4I 2020 published failure-mode formulas against each
    test row's raw sensor readings, independent of what the model predicted.
    Returns one boolean column per diagnosable fault mode (TWF/HDF/PWF/OSF).

    RNF is deliberately excluded here: it is defined as a fixed random
    chance with NO dependency on any feature, so there is no formula to
    evaluate — it can only ever be inferred by elimination (see
    `diagnose_fault_type` below).
    """
    df = X_test.copy()

    # Tool Wear Failure: tool wear between 200-240 minutes
    twf = df[COL_TOOL_WEAR].between(200, 240)

    # Heat Dissipation Failure: small temp difference + low rotational speed
    temp_diff = df[COL_PROCESS_TEMP] - df[COL_AIR_TEMP]
    hdf = (temp_diff < 8.6) & (df[COL_ROT_SPEED] < 1380)

    # Power Failure: power (W) = torque (Nm) * rotational speed (rad/s)
    rot_speed_rad_s = df[COL_ROT_SPEED] * (2 * np.pi / 60)
    power = df[COL_TORQUE] * rot_speed_rad_s
    pwf = (power < 3500) | (power > 9000)

    # Overstrain Failure: tool wear * torque exceeds a type-dependent threshold
    osf_threshold = pd.Series(np.nan, index=df.index)
    for type_col, threshold in OSF_THRESHOLDS.items():
        if type_col in df.columns:
            osf_threshold = osf_threshold.where(df[type_col] != 1, threshold)
    # rows whose Type one-hot column wasn't found default to the most common
    # threshold (Type L, 11,000) rather than leaving the rule unevaluable
    osf_threshold = osf_threshold.fillna(OSF_THRESHOLDS["Type_L"])
    osf = (df[COL_TOOL_WEAR] * df[COL_TORQUE]) > osf_threshold

    rules_df = pd.DataFrame({
        "rule_TWF": twf,
        "rule_HDF": hdf,
        "rule_PWF": pwf,
        "rule_OSF": osf,
    })

    print(f"\n[RULES] Rule trigger counts across test set:")
    print(rules_df.sum())

    return rules_df


# ----------------------------------------------------------------------
# 4. DIAGNOSE FAULT TYPE
# ----------------------------------------------------------------------
def diagnose_fault_type(rules_df: pd.DataFrame, predicted_failure: pd.Series) -> pd.DataFrame:
    """
    Converts the per-row rule flags into a readable diagnosis:
      - Lists every rule that triggered (a row can trigger more than one).
      - Picks a 'primary' suspected fault type using TWF > HDF > PWF > OSF
        priority (arbitrary but fixed ordering, documented here so it's
        reproducible — the dataset's own generation logic does not define
        a priority when multiple modes co-occur, since any single trigger
        is sufficient to fail the process).
      - Only assigns a diagnosis to rows the MODEL predicted as failures;
        for predicted no-failure rows, diagnosis is left blank, since
        diagnosing a fault type for something the model didn't flag as a
        failure wouldn't be meaningful.
      - If the model predicted a failure but none of TWF/HDF/PWF/OSF
        triggered, the diagnosis is 'Unclassified / Possible Random Failure
        (RNF)' — RNF has no feature-based signature, so this is the
        honest, correct conclusion by elimination rather than a guess.
    """
    priority = ["TWF", "HDF", "PWF", "OSF"]

    triggered_list = []
    primary_list = []

    for i in range(len(rules_df)):
        row = rules_df.iloc[i]
        triggered = [name for name in priority if row[f"rule_{name}"]]

        if predicted_failure.iloc[i] == 0:
            triggered_list.append("")
            primary_list.append("")
            continue

        if triggered:
            triggered_list.append(", ".join(triggered))
            primary_list.append(triggered[0])
        else:
            triggered_list.append("")
            primary_list.append("Unclassified / Possible Random Failure (RNF)")

    return pd.DataFrame({
        "triggered_rules": triggered_list,
        "diagnosed_fault_type": primary_list,
    })


# ----------------------------------------------------------------------
# 5. COMBINE & SAVE
# ----------------------------------------------------------------------
def combine_and_save(shap_df: pd.DataFrame, X_test: pd.DataFrame,
                     outputs_dir: str = OUTPUTS_DIR) -> pd.DataFrame:
    """
    Merges the SHAP-based explanations with the rule-based diagnosis into
    a single report, row-aligned by position, and writes it to
    outputs/fault_diagnosis.csv.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    rules_df = apply_diagnostic_rules(X_test)
    diagnosis_df = diagnose_fault_type(rules_df, shap_df["predicted_failure"])

    combined = pd.concat(
        [shap_df.reset_index(drop=True),
         rules_df.reset_index(drop=True),
         diagnosis_df.reset_index(drop=True)],
        axis=1,
    )

    path = os.path.join(outputs_dir, "fault_diagnosis.csv")
    combined.to_csv(path, index=False)
    print(f"\n[SAVE] Fault diagnosis report written to '{path}'")

    return combined


# ----------------------------------------------------------------------
# 6. SUMMARY / SANITY CHECK
# ----------------------------------------------------------------------
def print_diagnosis_summary(combined: pd.DataFrame) -> None:
    """
    Prints two things maintenance personnel / examiners would want to see:
      - How many predicted failures fall into each diagnosed fault type.
      - How often the rule-based verdict (any rule triggered) agrees with
        the MODEL's own prediction — a quick sanity check on whether the
        model's failure predictions line up with the dataset's own known
        failure logic, not a measure of model accuracy against true labels.
    """
    failed_rows = combined[combined["predicted_failure"] == 1]

    print("\n" + "=" * 60)
    print("FAULT DIAGNOSIS SUMMARY")
    print("=" * 60)

    if len(failed_rows) == 0:
        print("No predicted failures in this test set.")
    else:
        print("\nDiagnosed fault type counts (among predicted failures):")
        print(failed_rows["diagnosed_fault_type"].value_counts())

    rule_flag = (
        combined[["rule_TWF", "rule_HDF", "rule_PWF", "rule_OSF"]].any(axis=1)
    ).astype(int)
    agreement = (rule_flag == combined["predicted_failure"]).mean() * 100
    print(f"\nRule-based vs. model-predicted agreement rate: {agreement:.2f}%")
    print("(Rows where the model's prediction matches whether ANY known "
          "deterministic rule fired — RNF, being random, cannot agree by design.)")
    print("=" * 60)


# ----------------------------------------------------------------------
# ORCHESTRATOR
# ----------------------------------------------------------------------
def main():
    """Runs the full fault diagnosis pipeline end-to-end."""
    shap_df = load_shap_explanations()
    X_test = load_test_features()

    combined = combine_and_save(shap_df, X_test)
    print_diagnosis_summary(combined)

    print("\n[DONE] Fault diagnosis pipeline complete.")


if __name__ == "__main__":
    main()