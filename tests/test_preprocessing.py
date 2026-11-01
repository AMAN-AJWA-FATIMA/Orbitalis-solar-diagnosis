import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing as pp
from tests.conftest import make_wide


def test_valid_upload_passes():
    wide, warns = pp.validate_wide(make_wide([4000] * 10))
    assert config.voltage_col("py") in wide.columns


def test_missing_time_column_rejected():
    with pytest.raises(pp.DataValidationError, match="Time Stamp"):
        pp.validate_wide(make_wide([4000] * 5).drop(columns=config.TIME_COL))


def test_missing_panel_columns_rejected():
    df = make_wide([4000] * 5)[[config.TIME_COL, "Vpy (mV)"]]
    with pytest.raises(pp.DataValidationError, match="voltage and a current"):
        pp.validate_wide(df)


def test_empty_rejected():
    with pytest.raises(pp.DataValidationError):
        pp.validate_wide(pd.DataFrame())


def test_battery_columns_ignored_with_warning():
    df = make_wide([4000] * 5); df["Vbat (V)"] = 4.1
    wide, warns = pp.validate_wide(df)
    assert "Vbat (V)" not in wide.columns and any("Battery" in w for w in warns)


def test_invalid_numbers_become_nan_not_dropped():
    df = make_wide([4000, "bad", 4000, 4000]); df[config.voltage_col("py")] = df[config.voltage_col("py")]
    clean, report = pp.clean_wide(df)
    assert len(clean) == 4 and clean[config.voltage_col("py")].isna().sum() == 1
    assert "Vpy (mV)" in report["invalid_numeric_values_coerced_to_nan"]


def test_rows_without_timestamp_quarantined_and_duplicates_removed():
    df = make_wide([4000] * 4)
    df.loc[2, config.TIME_COL] = np.nan
    df = pd.concat([df, df.iloc[[0]]])
    clean, report = pp.clean_wide(df)
    assert report["rows_quarantined_missing_timestamp"] == 1
    assert report["duplicate_observations_removed"] == 1 and len(clean) == 3


def test_sorted_and_datetime_created():
    df = make_wide([4000] * 4).iloc[::-1]
    clean, _ = pp.clean_wide(df)
    assert clean[config.TIME_COL].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(clean[config.DATETIME_COL])


def test_anomalous_values_not_removed():
    clean, _ = pp.clean_wide(make_wide([0.0, 1.5, 4000, 4000]))
    assert (clean[config.voltage_col("py")] < 10).sum() == 2


def test_wide_to_long_contiguity_flags():
    df = make_wide([4000] * 5); df.loc[3:, config.TIME_COL] += 20  # a gap
    long = pp.wide_to_long(pp.clean_wide(df)[0])
    py = long[long.Panel == "py"].reset_index(drop=True)
    assert not py.loc[3, "Prev_contiguous"] and not py.loc[2, "Next_contiguous"] and py.loc[1, "Next_contiguous"]


def test_real_dataset_excludes_battery():
    wide = pd.read_csv(config.CLEANED_CSV)
    assert not any(c.startswith(config.BATTERY_COLUMN_PREFIXES) for c in wide.columns)
