"""Dataset loading, validation, cleaning and reshaping for ORBITALIS.

Design rules (see project specification):
* the raw workbooks are never modified;
* only solar-panel columns (voltage / current / temperature) are selected,
  battery columns are dropped;
* anomalous values are NOT removed as outliers;
* missing / invalid numbers stay ``NaN`` (no pooled median imputation, which
  would hide failed panels) - imputation, if any, happens inside the ML
  module using statistics learned from normal training data only;
* rows without a usable timestamp cannot be placed in the time series and are
  quarantined and reported instead of silently kept.

Run as a script:  ``python -m src.preprocessing``
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

LOGGER = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when an input table cannot be used for diagnosis."""


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def parse_session_date(name: Any) -> pd.Timestamp:
    """Worksheet names such as '13 Feb 2021' or '23 October 2020' -> Timestamp."""
    try:
        return pd.to_datetime(str(name), dayfirst=True, errors="coerce")
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return pd.NaT


# --------------------------------------------------------------------------- #
# Raw workbook loading
# --------------------------------------------------------------------------- #
def load_raw_workbooks(raw_dir: Path = config.RAW_DIR) -> tuple[pd.DataFrame, list[dict]]:
    """Read every worksheet of every .xlsx in ``raw_dir``.

    Returns the combined wide table (solar columns only) and an inventory that
    documents what was found in each worksheet (nothing is assumed).
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"No .xlsx files found in '{raw_dir}'. Download the BIRDS EPS dataset "
            "(https://github.com/BIRDSOpenSource/EPS_dataset) and place the "
            "satellite workbooks (e.g. RAAVANA.xlsx, UGUISU.xlsx) in that folder."
        )

    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []
    for path in files:
        workbook = pd.ExcelFile(path)
        for sheet in workbook.sheet_names:
            raw = workbook.parse(sheet)
            raw.columns = [str(c).strip() for c in raw.columns]
            present = [c for c in config.SENSOR_COLUMNS if c in raw.columns]
            missing = [c for c in config.SENSOR_COLUMNS if c not in raw.columns]
            battery = [c for c in raw.columns if c.startswith(config.BATTERY_COLUMN_PREFIXES)]
            inventory.append(
                {
                    "file": path.name,
                    "sheet": sheet,
                    "rows": int(len(raw)),
                    "columns_total": int(raw.shape[1]),
                    "solar_columns_found": len(present),
                    "solar_columns_missing": missing,
                    "battery_columns_excluded": battery,
                    "has_time_stamp": config.TIME_COL in raw.columns,
                }
            )
            if config.TIME_COL not in raw.columns or not present:
                LOGGER.warning("Skipping %s / %s: no time stamp or solar columns", path.name, sheet)
                continue
            selected = raw[[config.TIME_COL] + present].copy()
            selected[config.SATELLITE_COL] = path.stem.upper()
            selected[config.SESSION_COL] = str(sheet)
            frames.append(selected)

    if not frames:
        raise DataValidationError("No worksheet contained solar-panel telemetry.")
    return pd.concat(frames, ignore_index=True), inventory


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_wide(
    df: pd.DataFrame,
    expected_interval_s: float = config.DEFAULT_SAMPLING_INTERVAL_S,
) -> tuple[pd.DataFrame, dict]:
    """Clean a wide telemetry table. Returns ``(clean_df, quality_report)``."""
    report: dict[str, Any] = {"rows_input": int(len(df))}
    data = df.copy()

    sensor_cols = [c for c in config.SENSOR_COLUMNS if c in data.columns]
    before_na = data[sensor_cols].isna().sum()
    data[sensor_cols] = data[sensor_cols].apply(pd.to_numeric, errors="coerce")
    after_na = data[sensor_cols].isna().sum()
    report["invalid_numeric_values_coerced_to_nan"] = {
        c: int(after_na[c] - before_na[c]) for c in sensor_cols if after_na[c] > before_na[c]
    }

    # --- time handling -------------------------------------------------------
    data[config.TIME_COL] = pd.to_numeric(data[config.TIME_COL], errors="coerce")
    no_time = data[config.TIME_COL].isna()
    report["rows_quarantined_missing_timestamp"] = int(no_time.sum())
    quarantined = data[no_time]
    if len(quarantined):
        report["quarantined_by_session"] = (
            quarantined.groupby([config.SATELLITE_COL, config.SESSION_COL]).size().astype(int)
            .rename(lambda x: x).reset_index(name="rows").to_dict("records")
        )
    data = data[~no_time].copy()

    # --- duplicates ----------------------------------------------------------
    keys = [config.SATELLITE_COL, config.SESSION_COL, config.TIME_COL]
    dup = data.duplicated(subset=keys, keep="first")
    report["duplicate_observations_removed"] = int(dup.sum())
    data = data[~dup].copy()

    # --- chronological order & datetime ---------------------------------------
    data[config.SESSION_DATE_COL] = data[config.SESSION_COL].map(parse_session_date)
    data[config.DATETIME_COL] = data[config.SESSION_DATE_COL] + pd.to_timedelta(
        data[config.TIME_COL], unit="s"
    )
    data = data.sort_values(
        [config.SATELLITE_COL, config.SESSION_DATE_COL, config.SESSION_COL, config.TIME_COL]
    ).reset_index(drop=True)

    # --- sampling interval check ------------------------------------------------
    gaps: list[dict] = []
    for (sat, sess), grp in data.groupby([config.SATELLITE_COL, config.SESSION_COL], sort=False):
        dt = grp[config.TIME_COL].diff().dropna()
        off = dt[~np.isclose(dt, expected_interval_s)]
        gaps.append(
            {
                "satellite": sat,
                "session": sess,
                "samples": int(len(grp)),
                "median_interval_s": float(dt.median()) if len(dt) else None,
                "irregular_intervals": int(len(off)),
            }
        )
    report["sampling_check"] = gaps
    report["expected_interval_s"] = expected_interval_s
    report["missing_values_per_sensor_column"] = {
        c: int(data[c].isna().sum()) for c in sensor_cols if data[c].isna().any()
    }
    report["rows_output"] = int(len(data))
    return data, report


# --------------------------------------------------------------------------- #
# Validation of user supplied tables (CSV upload)
# --------------------------------------------------------------------------- #
def validate_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate an uploaded / external wide table.

    Raises :class:`DataValidationError` with a precise message when the table is
    unusable; otherwise returns the normalised table plus a list of warnings.
    """
    if df is None or df.empty:
        raise DataValidationError("The file contains no rows.")
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    warnings: list[str] = []

    if config.TIME_COL not in data.columns:
        raise DataValidationError(
            f"Missing required column '{config.TIME_COL}' (elapsed seconds, or datetimes)."
        )

    battery = [c for c in data.columns if c.startswith(config.BATTERY_COLUMN_PREFIXES)]
    if battery:
        warnings.append(f"Battery columns ignored (out of scope): {', '.join(battery)}")

    usable_panels = [
        p for p in config.PANELS
        if config.voltage_col(p) in data.columns and config.current_col(p) in data.columns
    ]
    if not usable_panels:
        raise DataValidationError(
            "No solar panel has both a voltage and a current column. Expected columns such "
            f"as '{config.voltage_col('py')}' and '{config.current_col('py')}'."
        )
    skipped = [p for p in config.PANELS if p not in usable_panels]
    if skipped:
        warnings.append(
            "Panels without voltage+current columns are skipped: "
            + ", ".join(config.PANELS[p] for p in skipped)
        )
    for p in usable_panels:
        if config.temperature_col(p) not in data.columns:
            data[config.temperature_col(p)] = np.nan
            warnings.append(f"Temperature column for panel {config.PANELS[p]} missing; left empty.")

    # time column: numeric elapsed seconds or parseable datetimes
    numeric_time = pd.to_numeric(data[config.TIME_COL], errors="coerce")
    if numeric_time.notna().mean() < 0.5:
        parsed = pd.to_datetime(data[config.TIME_COL], errors="coerce")
        if parsed.notna().mean() < 0.5:
            raise DataValidationError(
                f"Column '{config.TIME_COL}' is neither numeric seconds nor parseable datetimes."
            )
        if config.SESSION_COL not in data.columns:
            data[config.SESSION_COL] = "upload"
        start = parsed.groupby(data[config.SESSION_COL]).transform("min")
        data[config.TIME_COL] = (parsed - start).dt.total_seconds()
        warnings.append("Datetimes converted to elapsed seconds from the start of each session.")
    else:
        data[config.TIME_COL] = numeric_time

    if config.SATELLITE_COL not in data.columns:
        data[config.SATELLITE_COL] = "UPLOAD"
        warnings.append("No 'Satellite_ID' column: rows labelled 'UPLOAD'.")
    if config.SESSION_COL not in data.columns:
        data[config.SESSION_COL] = "upload"

    keep = (
        [config.TIME_COL, config.SATELLITE_COL, config.SESSION_COL]
        + [c for p in usable_panels
           for c in (config.temperature_col(p), config.voltage_col(p), config.current_col(p))]
    )
    return data[keep], warnings


# --------------------------------------------------------------------------- #
# Wide -> long (one row per panel observation)
# --------------------------------------------------------------------------- #
LONG_COLUMNS = [
    config.SATELLITE_COL, config.SESSION_COL, config.SESSION_DATE_COL, config.DATETIME_COL,
    "Elapsed_s", "Panel", "Panel_Label", "Voltage_mV", "Current_mA", "Temperature_C",
]


def wide_to_long(
    df: pd.DataFrame,
    expected_interval_s: float = config.DEFAULT_SAMPLING_INTERVAL_S,
) -> pd.DataFrame:
    """Reshape to one row per (satellite, session, panel, time) and add the
    sequence-context flags needed for consecutive-sample conditions."""
    frames: list[pd.DataFrame] = []
    base_cols = [c for c in (config.SATELLITE_COL, config.SESSION_COL,
                             config.SESSION_DATE_COL, config.DATETIME_COL) if c in df.columns]
    for panel, label in config.PANELS.items():
        v, i, t = config.voltage_col(panel), config.current_col(panel), config.temperature_col(panel)
        if v not in df.columns or i not in df.columns:
            continue
        part = df[base_cols].copy()
        part["Elapsed_s"] = df[config.TIME_COL].to_numpy()
        part["Panel"] = panel
        part["Panel_Label"] = label
        part["Voltage_mV"] = df[v].to_numpy()
        part["Current_mA"] = df[i].to_numpy()
        part["Temperature_C"] = df[t].to_numpy() if t in df.columns else np.nan
        frames.append(part)
    if not frames:
        raise DataValidationError("No panel with voltage and current columns was found.")

    long = pd.concat(frames, ignore_index=True)
    for col in (config.SESSION_DATE_COL, config.DATETIME_COL):
        if col not in long.columns:
            long[col] = pd.NaT
    keys = [config.SATELLITE_COL, config.SESSION_COL, "Panel"]
    long = long.sort_values(keys + ["Elapsed_s"]).reset_index(drop=True)

    grp = long.groupby(keys, sort=False)["Elapsed_s"]
    long["Prev_contiguous"] = np.isclose(grp.diff(), expected_interval_s)
    long["Next_contiguous"] = np.isclose(-grp.diff(-1), expected_interval_s)
    return long[LONG_COLUMNS + ["Prev_contiguous", "Next_contiguous"]]


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
def run(raw_dir: Path = config.RAW_DIR, output: Path = config.CLEANED_CSV) -> pd.DataFrame:
    """Raw workbooks -> cleaned wide CSV + data-quality report."""
    wide, inventory = load_raw_workbooks(raw_dir)
    clean, report = clean_wide(wide)
    report["inventory"] = inventory
    report["features_used"] = [c for c in config.SENSOR_COLUMNS if c in clean.columns]
    report["note"] = (
        "Battery columns (Vbat, Ibatt, Tbatt) are excluded by design. Missing values are "
        "preserved (no imputation). Datetime = session date + elapsed seconds; the clock "
        "time of day at which each high-sampling session started is not part of the dataset."
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    config.QUALITY_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.QUALITY_REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return clean


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    clean = run()
    print("Preprocessing completed.")
    print("Rows:", len(clean), "| Columns:", list(clean.columns))
    print("Saved:", config.CLEANED_CSV)
    print("Quality report:", config.QUALITY_REPORT_JSON)


if __name__ == "__main__":
    main()
