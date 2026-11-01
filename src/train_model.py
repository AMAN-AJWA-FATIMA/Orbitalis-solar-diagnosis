"""Train and evaluate the Isolation Forest.

Methodology
-----------
* Observations are panel readings (log-scaled voltage and current) taken from
  ``dataset/processed/cleaned_data.csv``.
* Documented-condition labels follow SRC-1 Eq.(1)/(2) (see ``labeling.py``).
  They are rule-derived, not independent operator ground truth.
* Fitting uses only observations labelled normal (no documented condition) from
  the *training sessions*.
* The split is chronological and satellite-aware: for every satellite the most
  recent session is held out for testing, so no test session is seen in training.
* The contamination parameter (decision threshold) is chosen by
  leave-one-session-out validation *inside the training sessions*; the test
  sessions are never used for tuning.
* Metrics are computed on the held-out sessions only.

Run:  ``python -m src.train_model``
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import config, preprocessing
from .knowledge_base import load_knowledge_base
from .labeling import label_documented_conditions
from .ml_detector import FEATURES, MLDetector, build_features

RANDOM_STATE = 42
N_ESTIMATORS = 200
# Candidate decision thresholds (share of normal training data scored as anomalous).
CONTAMINATION_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def load_long_table() -> pd.DataFrame:
    """Processed CSV -> labelled long table (preprocessing is run if needed)."""
    if not config.CLEANED_CSV.exists():
        preprocessing.run()
    wide = pd.read_csv(config.CLEANED_CSV, parse_dates=[config.SESSION_DATE_COL, config.DATETIME_COL])
    long = preprocessing.wide_to_long(wide)
    kb = load_knowledge_base()
    return label_documented_conditions(long, kb.detection_parameters)


def split_sessions(long: pd.DataFrame) -> tuple[list[tuple], list[tuple]]:
    """Per satellite, the latest session (by date) is the test session."""
    sessions = (
        long[[config.SATELLITE_COL, config.SESSION_COL, config.SESSION_DATE_COL]]
        .drop_duplicates().sort_values([config.SATELLITE_COL, config.SESSION_DATE_COL])
    )
    train, test = [], []
    for sat, grp in sessions.groupby(config.SATELLITE_COL):
        rows = [(sat, r[config.SESSION_COL]) for _, r in grp.iterrows()]
        if len(rows) == 1:
            train += rows  # cannot hold out the only session
        else:
            train += rows[:-1]
            test.append(rows[-1])
    return train, test


def _select(long: pd.DataFrame, sessions: list[tuple]) -> pd.DataFrame:
    mask = pd.Series(False, index=long.index)
    for sat, sess in sessions:
        mask |= (long[config.SATELLITE_COL] == sat) & (long[config.SESSION_COL] == sess)
    return long[mask]


def evaluate(scored: pd.DataFrame) -> dict:
    """Metrics on scored, labelled observations (anomaly = positive class)."""
    scored = scored[scored["ML_Status"] != config.LABEL_NOT_SCORED]
    y_true = scored["Documented_Label"].to_numpy()
    y_pred = (scored["ML_Prediction"] == -1).astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result = {
        "n_observations": int(len(scored)),
        "n_documented_anomalies": int(y_true.sum()),
        "n_predicted_anomalies": int(y_pred.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"labels": ["Normal", "Anomaly"], "matrix": cm.tolist()},
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["Normal", "Anomaly"],
            output_dict=True, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        # threshold-free: lower decision score = more anomalous
        result["roc_auc"] = float(roc_auc_score(y_true, -scored["ML_Score"]))
        result["average_precision"] = float(average_precision_score(y_true, -scored["ML_Score"]))
    for kind, col in (("Type 1", "Type1_Condition"), ("Type 2", "Type2_Condition")):
        sub = scored[scored[col]]
        result[f"recall_{kind.replace(' ', '_').lower()}"] = (
            float((sub["ML_Prediction"] == -1).mean()) if len(sub) else None
        )
        result[f"n_{kind.replace(' ', '_').lower()}"] = int(len(sub))
    return result


def _fit(features: pd.DataFrame, contamination: float) -> tuple[StandardScaler, IsolationForest]:
    scaler = StandardScaler().fit(features)
    model = IsolationForest(
        n_estimators=N_ESTIMATORS, contamination=contamination, random_state=RANDOM_STATE
    ).fit(scaler.transform(features))
    return scaler, model


def select_contamination(train_df: pd.DataFrame) -> tuple[float, list[dict]]:
    """Leave-one-session-out validation inside the training sessions.

    For each training session: fit on the normal observations of the other
    training sessions, then compute F1 on the held-out session (all of its
    observations, anomalies included). Returns the contamination with the best
    mean F1 and the full validation table.
    """
    feats = build_features(train_df)
    valid = feats.notna().all(axis=1)
    sessions = train_df[[config.SATELLITE_COL, config.SESSION_COL]].drop_duplicates().to_numpy().tolist()
    table: list[dict] = []
    for c in CONTAMINATION_GRID:
        scores = []
        for sat, sess in sessions:
            hold = (train_df[config.SATELLITE_COL] == sat) & (train_df[config.SESSION_COL] == sess)
            fit_mask = ~hold & valid & (train_df["Documented_Label"] == 0)
            test_mask = hold & valid
            if fit_mask.sum() < 50 or test_mask.sum() == 0:
                continue
            scaler, model = _fit(feats.loc[fit_mask], c)
            pred = model.predict(scaler.transform(feats.loc[test_mask])) == -1
            scores.append(f1_score(train_df.loc[test_mask, "Documented_Label"], pred.astype(int), zero_division=0))
        if scores:
            table.append({"contamination": c, "mean_f1": float(np.mean(scores)),
                          "fold_f1": [float(x) for x in scores], "folds": len(scores)})
    if not table:
        raise RuntimeError("Validation failed: too few sessions for leave-one-session-out.")
    best = max(table, key=lambda r: r["mean_f1"])
    return best["contamination"], table


def train() -> dict:
    long = load_long_table()
    train_sessions, test_sessions = split_sessions(long)
    train_df = _select(long, train_sessions)
    test_df = _select(long, test_sessions)

    contamination, validation_table = select_contamination(train_df)

    feats = build_features(train_df)
    usable = feats.notna().all(axis=1) & (train_df["Documented_Label"] == 0)
    x_train = feats.loc[usable]
    if len(x_train) < 50:
        raise RuntimeError("Not enough normal training observations to fit the model.")
    scaler, model = _fit(x_train, contamination)

    preprocessing_info = {"features": FEATURES, "scaler": scaler}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(preprocessing_info, config.PREPROCESSING_PATH)

    detector = MLDetector(model, preprocessing_info)
    scored_test = detector.detect(test_df)
    metrics = evaluate(scored_test)
    scored_all = detector.detect(long)

    def dist(df: pd.DataFrame) -> list[dict]:
        g = (df.groupby([config.SATELLITE_COL, config.SESSION_COL, "Panel_Label", "ML_Status"])
             .size().reset_index(name="count"))
        return g.to_dict("records")

    metrics["evaluation_sessions"] = [{"satellite": s, "session": x} for s, x in test_sessions]
    metrics["anomaly_distribution_test"] = dist(scored_test)
    metrics["anomaly_distribution_all"] = dist(scored_all)
    metrics["limitations"] = [
        "Labels are derived from the published definitions (SRC-1 Eq.1 and Eq.2); they are not independent operator-verified ground truth.",
        "Only five orbit-long sessions from two satellites are available; the held-out set is small and results are not a measure of operational performance.",
        "Documented anomalies are concentrated on one panel per satellite (RAAVANA +Y: Type 1, UGUISU +Y: Type 2), so panel-level generalisation is untested.",
        "The decision threshold (contamination) was selected by leave-one-session-out validation on training sessions only; with so few sessions this choice is noisy.",
        "Isolation Forest flags a near-zero voltage only when the current also differs from the eclipse idle value (about 4.7 mA); a failed panel reading 0 mV with idle current is scored normal by the model and is caught only by the knowledge-based Type 1 rule.",
        "The ML model scores each observation independently (log voltage and log current); sequence logic such as SRC-1 Eq.(2) is handled by the knowledge-based rules, not by the model.",
    ]

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "sklearn.ensemble.IsolationForest",
        "n_estimators": N_ESTIMATORS, "contamination": contamination, "random_state": RANDOM_STATE,
        "contamination_validation": validation_table,
        "feature_selection": "Log-scaled voltage and current. Temperature and 3-sample rolling features were tried in leave-one-session-out validation and reduced F1; temperature is also excluded by SRC-1 Section 2.4.",
        "features": FEATURES, "target": "documented-condition label (SRC-1 Eq.1/Eq.2), anomaly = positive class",
        "training_sessions": [{"satellite": s, "session": x} for s, x in train_sessions],
        "test_sessions": [{"satellite": s, "session": x} for s, x in test_sessions],
        "n_training_observations": int(len(x_train)),
        "fit_on": "observations without a documented condition (normal only)",
        "prediction_convention": "Isolation Forest -1 -> Anomaly, 1 -> Normal",
        "imputation": "None. Observations lacking voltage or current are not scored.",
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    keep = [config.SATELLITE_COL, config.SESSION_COL, "Elapsed_s", "Panel_Label", "Voltage_mV",
            "Current_mA", "Temperature_C", "Documented_Type", "ML_Status", "ML_Score"]
    scored_test[keep].to_csv(config.TEST_RESULTS_CSV, index=False)
    return {"metrics": metrics, "metadata": metadata}


def main() -> None:
    out = train()
    m = out["metrics"]
    print("Model training completed.")
    print("Training observations (normal only):", out["metadata"]["n_training_observations"])
    print("Selected contamination (LOSO validation):", out["metadata"]["contamination"])
    for row in out["metadata"]["contamination_validation"]:
        print("   c=%.2f  mean F1=%.3f  folds=%s" % (row["contamination"], row["mean_f1"], [round(x, 2) for x in row["fold_f1"]]))
    print("Test sessions:", m["evaluation_sessions"])
    print(f"Test observations: {m['n_observations']}  documented anomalies: {m['n_documented_anomalies']}  predicted: {m['n_predicted_anomalies']}")
    print(f"Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("Confusion matrix [Normal, Anomaly]:", m["confusion_matrix"]["matrix"])
    print("ROC-AUC:", m.get("roc_auc"), " Avg precision:", m.get("average_precision"))
    print("Recall Type 1:", m["recall_type_1"], " Recall Type 2:", m["recall_type_2"])
    print("Saved:", config.MODEL_PATH, config.PREPROCESSING_PATH, config.METRICS_PATH)


if __name__ == "__main__":
    main()
