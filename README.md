# Explainable AI Framework for Machine Failure Prediction and Intelligent Fault Diagnosis

**Phase 1 — Final Year Project**

An AI-based predictive maintenance system that predicts industrial machine failures, explains *why* each prediction was made using Explainable AI (SHAP), and diagnoses the likely fault type using physics-based rules — all visualized through a custom interactive monitoring dashboard.

**Domain:** Industrial Predictive Maintenance / Smart Manufacturing (Industry 4.0) — combining Machine Learning, Explainable AI (XAI), and Fault Diagnosis Engineering.

---

## What it does

Traditional maintenance is reactive (fix after failure) or scheduled (fix on a timer, whether needed or not). This project predicts failures *before* they happen, and — unlike a typical "black box" ML model — explains which sensor readings drove each prediction and what type of failure is likely, so a maintenance engineer can trust and act on the result.

The pipeline:

```
Raw sensor data
      │
      ▼
add_vibration_feature.py     → adds a simulated vibration sensor feature
      │
      ▼
data_preprocessing.py        → cleans, encodes, splits data (leak-free)
      │
      ▼
feature_selection.py         → drops redundant / low-value features
      │
      ▼
model_training.py            → trains XGBoost, evaluates performance
      │
      ▼
shap_explainability.py       → explains WHY each prediction was made
      │
      ▼
fault_diagnosis.py           → diagnoses WHICH fault type is likely
      │
      ▼
dashboard.py                 → generates an interactive HTML monitoring dashboard
```

---

## Dataset

**[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)** (UCI Machine Learning Repository)

- 10,000 machines, 6 operational features: air temperature, process temperature, rotational speed, torque, tool wear, product quality type (L/M/H)
- Target: `Machine failure` (binary), plus 5 named failure-mode flags (TWF, HDF, PWF, OSF, RNF)
- Severely imbalanced: ~96.6% no-failure vs ~3.4% failure

**Added feature:** `sim_Vibration [mm/s]` — a **simulated** vibration sensor reading, since the original dataset has no vibration sensor. Generated from rotational speed, tool wear, and torque (documented predictive-maintenance relationships), **not from the failure label** — this was explicitly checked for and corrected after an initial version accidentally leaked the target label into the feature (see [Notes](#notes--lessons-learned) below).

> ⚠️ **Disclosure:** the vibration feature is synthetic, not real sensor data. This is disclosed here and in the code's docstrings.

---

## Tech stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Data handling | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Matplotlib, Plotly |
| Model persistence | joblib |
| Dashboard | Self-built HTML/CSS/JS (no server required) |

---

## Results

Trained and evaluated on a held-out test set of 2,000 machines (80/20 stratified split):

| Metric | Score |
|---|---|
| Accuracy | 0.979 |
| Precision | 0.655 |
| Recall | 0.809 |
| F1-score | 0.724 |
| ROC-AUC | 0.970 |

**Why precision < recall, and why that's intentional:** the model is tuned (via `scale_pos_weight`) to prioritize catching real failures over avoiding false alarms — the right tradeoff for predictive maintenance, where missing a failure is far costlier than an unnecessary inspection.

**Feature importance:** rotational speed, torque, and tool wear account for ~76% of the model's decisions — consistent with the physical formulas that actually define failures in this dataset. The added vibration feature contributed meaningfully (~5%) despite being synthetic.

**Fault diagnosis:** among predicted failures, the rule-based diagnosis (derived from the dataset's own published failure formulas) agreed with the model's predictions **91.95%** of the time — evidence the model learned genuine mechanical failure patterns, not noise.

---

## Project structure

```
├── ai4i2020.csv                    # original dataset (not included — download separately)
├── add_vibration_feature.py        # adds simulated vibration sensor data
├── data_preprocessing.py           # cleaning, encoding, train/test split
├── feature_selection.py            # correlation + importance-based feature filtering
├── model_training.py               # XGBoost training + evaluation
├── shap_explainability.py          # SHAP-based explainability
├── fault_diagnosis.py              # rule-based fault-type diagnosis
├── dashboard.py                    # generates dashboard.html
├── processed/                      # train/test splits (generated)
├── outputs/                        # predictions, SHAP values, diagnosis reports (generated)
├── models/                         # saved trained model (generated)
└── dashboard.html                  # interactive monitoring dashboard (generated)
```

---

## How to run

```bash
pip install pandas scikit-learn xgboost shap matplotlib joblib

python add_vibration_feature.py --input ai4i2020.csv --output ai4i2020_with_vibration.csv
python data_preprocessing.py --input ai4i2020_with_vibration.csv --output_dir processed
python feature_selection.py
python model_training.py
python shap_explainability.py
python fault_diagnosis.py
python dashboard.py
```

Then open `dashboard.html` directly in a browser — no server required.

Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and place it as `ai4i2020.csv` in the project root before running.

---

## Notes & lessons learned

During development, an early version of the vibration feature accidentally used the `Machine failure` label to construct itself (a "failure boost" term), causing **target leakage** — the model scored a suspicious 100% recall / 1.0 ROC-AUC because it was effectively reading the answer key. This was caught by noticing the implausibly perfect metrics, traced to the exact line of code, and fixed by rebuilding the feature from legitimate physical sensor readings only. The corrected model's more modest, believable metrics (above) are the honest result.

---

## Limitations

- Vibration sensor data is simulated, not measured — a real deployment would need an actual vibration sensor.
- Fault-type diagnosis uses the dataset's own documented generation rules (Matzka, 2020) rather than a separately trained fault-classification model.
- The rule-agreement rate validates consistency with known physics, not ground-truth fault-type labels (which aren't available per-row in this dataset).


