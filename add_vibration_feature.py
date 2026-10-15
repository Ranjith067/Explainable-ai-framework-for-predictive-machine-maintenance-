"""
add_vibration_feature.py
======================
Phase 1 — Explainable AI Framework for Machine Failure Prediction
and Intelligent Fault Diagnosis

IMPORTANT / HONESTY NOTE FOR YOUR REPORT:
The AI4I 2020 dataset does NOT include a vibration sensor. Since no real
vibration sensor data was available for this project, this script generates
a SIMULATED vibration feature and adds it to the dataset, so the rest of the
pipeline (preprocessing -> training -> SHAP -> fault diagnosis -> dashboard)
can demonstrate working with a vibration-aware feature set.

This must be disclosed as a simulated/synthetic feature in your report --
it is not real sensor data. This is a legitimate and common approach when a
public dataset lacks a sensor you want to demonstrate, AS LONG AS it is
clearly labeled, which is why every column this script adds is prefixed
with 'sim_' and why this docstring exists.

How the simulated vibration is generated (physics-informed, not random):
Vibration RMS in rotating machinery is well documented in predictive-
maintenance literature to increase with:
  - rotational speed (mechanical imbalance / rotation-induced vibration)
  - tool wear (a worn tool cuts irregularly, increasing mechanical vibration)
  - torque / load (higher mechanical stress increases vibration)
plus ordinary sensor measurement noise.

IMPORTANT: vibration is deliberately built ONLY from these physical sensor
readings, never from the 'Machine failure' label itself. Building a feature
directly from the target label is called TARGET LEAKAGE -- it would let a
model score near-perfectly not because it learned anything real, but
because the feature secretly encodes the answer. Since tool wear/torque/
speed are themselves correlated with real failure conditions (the dataset's
own TWF/HDF/PWF/OSF formulas use these same readings), this vibration
feature still carries genuine predictive signal -- just honestly, through
the same physical readings a real sensor would respond to, not by peeking
at the label.

The formula below combines these three factors with weights chosen to
produce a plausible mm/s RMS range (roughly 0.5-8 mm/s, typical for
industrial rotating equipment), NOT fitted to any real vibration dataset --
there isn't one available here. Treat the resulting values as illustrative,
not measured.

Usage (command line):
    python add_vibration_feature.py --input ai4i2020.csv --output ai4i2020_with_vibration.csv

Usage (import):
    from add_vibration_feature import run_pipeline
    run_pipeline("ai4i2020.csv", "ai4i2020_with_vibration.csv")
"""

import os
import argparse
import numpy as np
import pandas as pd


COL_ROT_SPEED = "Rotational speed [rpm]"
COL_TORQUE = "Torque [Nm]"
COL_TOOL_WEAR = "Tool wear [min]"
COL_FAILURE = "Machine failure"

VIBRATION_COLUMN = "sim_Vibration [mm/s]"

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. LOAD RAW DATA
# ----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")
    df = pd.read_csv(path)
    print(f"[LOAD] Loaded '{path}' -> {df.shape[0]} rows, {df.shape[1]} columns")

    required = [COL_ROT_SPEED, COL_TORQUE, COL_TOOL_WEAR]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected column(s) {missing} not found -- this script expects "
            f"the ORIGINAL AI4I2020 CSV (before data_preprocessing.py has run), "
            f"since it needs the raw, un-encoded sensor columns."
        )
    return df


# ----------------------------------------------------------------------
# 2. SIMULATE VIBRATION
# ----------------------------------------------------------------------
def simulate_vibration(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Adds a simulated vibration RMS column, built from normalized rotational
    speed, tool wear, and torque, plus a boost on rows where a failure is
    already recorded, plus Gaussian sensor noise. See module docstring for
    the reasoning behind each term.

    Normalization uses this dataset's own min/max for each contributing
    column, so the formula adapts to whatever range of values the dataset
    actually contains rather than assuming fixed literal sensor ranges.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    def normalize(series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min())

    speed_term = normalize(df[COL_ROT_SPEED]) * 1.5      # up to +1.5 mm/s
    wear_term = (df[COL_TOOL_WEAR] / df[COL_TOOL_WEAR].max()) * 2.0  # up to +2.0 mm/s
    torque_term = normalize(df[COL_TORQUE]) * 1.0         # up to +1.0 mm/s

    baseline = 1.0  # idle/healthy-machine baseline vibration, mm/s
    noise = rng.normal(loc=0.0, scale=0.3, size=len(df))   # sensor noise

    vibration = baseline + speed_term + wear_term + torque_term + noise

    # NOTE: earlier versions of this script added an extra boost here based
    # directly on the 'Machine failure' label. That was a mistake -- using
    # the target label to construct a feature is TARGET LEAKAGE: the model
    # would then be able to "predict" failure almost perfectly not because
    # it learned real sensor patterns, but because the feature itself was
    # built from the answer. Vibration here is deliberately left as a
    # function of physical readings ONLY (speed, tool wear, torque, noise),
    # so it stays a legitimate, honest predictor rather than a leaked one.
    if COL_FAILURE not in df.columns:
        print("[SIMULATE] Note: no 'Machine failure' column found in this file "
              "(expected if this is being run on a test-only split).")

    vibration = np.clip(vibration, 0.1, None)  # vibration can't be negative

    df[VIBRATION_COLUMN] = vibration.round(3)

    print(f"\n[SIMULATE] Added '{VIBRATION_COLUMN}' "
          f"(mean={df[VIBRATION_COLUMN].mean():.2f}, "
          f"min={df[VIBRATION_COLUMN].min():.2f}, "
          f"max={df[VIBRATION_COLUMN].max():.2f})")

    if COL_FAILURE in df.columns:
        print("[SIMULATE] Mean vibration by failure status:")
        print(df.groupby(COL_FAILURE)[VIBRATION_COLUMN].mean())

    return df


# ----------------------------------------------------------------------
# 3. SAVE
# ----------------------------------------------------------------------
def save_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    print(f"\n[SAVE] Augmented dataset written to '{output_path}'")
    print(f"[SAVE] IMPORTANT: '{VIBRATION_COLUMN}' is SIMULATED data, not a "
          f"real sensor reading -- disclose this in your report.")


# ----------------------------------------------------------------------
# ORCHESTRATOR
# ----------------------------------------------------------------------
def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    df = simulate_vibration(df)
    save_data(df, output_path)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add a simulated vibration feature to the AI4I2020 dataset."
    )
    parser.add_argument("--input", required=True, help="Path to the original ai4i2020.csv")
    parser.add_argument("--output", default="ai4i2020_with_vibration.csv",
                        help="Path to write the augmented dataset")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()