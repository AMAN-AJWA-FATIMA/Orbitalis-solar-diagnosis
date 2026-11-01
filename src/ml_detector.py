"""Isolation Forest inference for solar-panel telemetry.

The model scores one panel observation at a time from its log-scaled voltage
and current. Temporal structure is handled by the knowledge-based rules, which
inspect neighbouring samples (SRC-1 Eq.2). Isolation Forest uses ``-1`` for anomalies
and ``1`` for normal observations; this module converts these to the
application labels ``Anomaly`` / ``Normal``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config

FEATURES = ["Log_Voltage", "Log_Current"]
_KEYS = [config.SATELLITE_COL, config.SESSION_COL, "Panel"]


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model files do not exist."""


class InvalidTelemetryError(ValueError):
    """Raised when telemetry passed to the detector lacks required columns."""


def build_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix aligned with ``long_df`` (same index).

    Features are ``log(1 + V[mV])`` and ``log(1 + I[mA])``. The log scale is used
    because a failed panel (V of a few mV) and an eclipsed panel (V of about
    1300 mV) are close on a linear scale but far apart on a log scale. Negative
    sensor values are clipped to 0 before the transform. Temperature is *not* an
    ML feature: leave-one-session-out validation showed it lowers detection
    quality, in line with SRC-1 Section 2.4 (low correlation with voltage/current).
    """
    required = {"Voltage_mV", "Current_mA"}
    missing = required - set(long_df.columns)
    if missing:
        raise InvalidTelemetryError(f"Telemetry is missing columns: {sorted(missing)}")
    voltage = pd.to_numeric(long_df["Voltage_mV"], errors="coerce")
    current = pd.to_numeric(long_df["Current_mA"], errors="coerce")
    feats = pd.DataFrame(index=long_df.index)
    feats["Log_Voltage"] = np.log1p(voltage.clip(lower=0))
    feats["Log_Current"] = np.log1p(current.clip(lower=0))
    return feats[FEATURES]


class MLDetector:
    """Wraps the fitted scaler + Isolation Forest."""

    def __init__(self, model: Any, preprocessing: dict[str, Any], metadata: dict[str, Any] | None = None):
        self.model = model
        self.preprocessing = preprocessing
        self.metadata = metadata or {}

    @classmethod
    def load(
        cls,
        model_path: Path = config.MODEL_PATH,
        preprocessing_path: Path = config.PREPROCESSING_PATH,
        metadata_path: Path = config.METADATA_PATH,
    ) -> "MLDetector":
        model_path, preprocessing_path = Path(model_path), Path(preprocessing_path)
        if not model_path.exists() or not preprocessing_path.exists():
            raise ModelNotFoundError(
                "Trained model not found. Run `python -m src.preprocessing` and "
                "`python -m src.train_model` first."
            )
        model = joblib.load(model_path)
        prep = joblib.load(preprocessing_path)
        meta = {}
        if Path(metadata_path).exists():
            meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(model, prep, meta)

    @property
    def expected_anomaly_rate(self) -> float | None:
        """Share of observations the model flags on normal data (its contamination)."""
        value = getattr(self.model, "contamination", None)
        return float(value) if isinstance(value, (int, float)) else None

    def detect(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``long_df`` with ``ML_Prediction`` (-1/1/0),
        ``ML_Status`` (Anomaly/Normal/Not scored) and ``ML_Score``.

        ``ML_Score`` is the Isolation Forest decision function: negative values are
        anomalous, larger values are more normal. It is an anomaly score, not a
        probability.
        """
        feats = build_features(long_df)
        scorable = feats.notna().all(axis=1)
        filled = feats
        out = long_df.copy()
        out["ML_Prediction"] = 0
        out["ML_Score"] = np.nan
        if scorable.any():
            x = self.preprocessing["scaler"].transform(filled.loc[scorable, FEATURES])
            out.loc[scorable, "ML_Prediction"] = self.model.predict(x)
            out.loc[scorable, "ML_Score"] = self.model.decision_function(x)
        out["ML_Status"] = np.select(
            [out["ML_Prediction"] == -1, out["ML_Prediction"] == 1],
            [config.LABEL_ANOMALY, config.LABEL_NORMAL],
            default=config.LABEL_NOT_SCORED,
        )
        return out


def detect_anomalies(long_df: pd.DataFrame, detector: MLDetector | None = None) -> pd.DataFrame:
    """Convenience wrapper that loads the saved model when none is supplied."""
    return (detector or MLDetector.load()).detect(long_df)
