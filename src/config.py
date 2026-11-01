"""Central configuration: paths, column naming and constants for ORBITALIS.

Every other module imports paths and naming conventions from here so that
there is a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- directories / files -----------------------------------------------------
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
CLEANED_CSV = PROCESSED_DIR / "cleaned_data.csv"
TEST_RESULTS_CSV = PROCESSED_DIR / "test_results.csv"
QUALITY_REPORT_JSON = PROCESSED_DIR / "data_quality_report.json"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "metadata.json"

KB_DIR = PROJECT_ROOT / "knowledge_base"
COMPONENTS_JSON = KB_DIR / "components.json"
FAULTS_JSON = KB_DIR / "faults.json"
RULES_JSON = KB_DIR / "rules.json"

DB_PATH = PROJECT_ROOT / "database" / "orbitalis.db"

# --- dataset schema (verified against RAAVANA.xlsx / UGUISU.xlsx) -------------
TIME_COL = "Time Stamp"
SATELLITE_COL = "Satellite_ID"
SESSION_COL = "Source_Sheet"
SESSION_DATE_COL = "Session_Date"
DATETIME_COL = "Datetime"

# BIRDS solar panels: code -> human readable orientation
PANELS: dict[str, str] = {
    "py": "+Y",
    "px": "+X",
    "mz": "-Z",
    "mx": "-X",
    "pz": "+Z",
}

# Columns that belong to the battery subsystem. They are explicitly OUT OF SCOPE
# and are never used as features or displayed.
BATTERY_COLUMN_PREFIXES = ("Vbat", "Ibatt", "Tbatt")


def voltage_col(panel: str) -> str:
    return f"V{panel} (mV)"


def current_col(panel: str) -> str:
    return f"I{panel} (mA)"


def temperature_col(panel: str) -> str:
    return f"T{panel} (°C)"


SENSOR_COLUMNS: list[str] = (
    [temperature_col(p) for p in PANELS]
    + [voltage_col(p) for p in PANELS]
    + [current_col(p) for p in PANELS]
)

# Sampling interval documented for RAAVANA / UGUISU / NEPALISAT (5 s).
DEFAULT_SAMPLING_INTERVAL_S = 5.0

# Application-level labels (Isolation Forest returns -1 / 1)
LABEL_ANOMALY = "Anomaly"
LABEL_NORMAL = "Normal"
LABEL_NOT_SCORED = "Not scored"
