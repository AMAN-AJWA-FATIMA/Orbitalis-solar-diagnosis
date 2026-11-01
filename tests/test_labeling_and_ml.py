import numpy as np
import pytest

from src import config, preprocessing as pp
from src.labeling import label_documented_conditions
from src.ml_detector import InvalidTelemetryError, MLDetector, ModelNotFoundError, build_features
from tests.conftest import make_wide


def label(kb, v, i=None):
    wide, _ = pp.clean_wide(make_wide(v, i))
    long = pp.wide_to_long(wide)
    return label_documented_conditions(long, kb.detection_parameters).query("Panel == 'py'").reset_index(drop=True)


def test_type1_boundary(kb):
    df = label(kb, [1199.9, 1200, 1500, 4000, 4000])
    assert df.Type1_Condition.tolist() == [True, False, False, False, False]


def test_type2_requires_three_consecutive_and_low_current(kb):
    v = [4000, 2500, 2500, 2500, 4000, 2500, 2500, 4000]
    i = [100, 10, 10, 10, 100, 10, 10, 10]
    df = label(kb, v, i)
    assert df.Type2_Condition.tolist() == [False, False, True, False, False, False, False, False]


def test_type2_not_triggered_with_high_current(kb):
    df = label(kb, [2500] * 6, [60] * 6)
    assert not df.Type2_Condition.any()


def test_type2_boundaries_inclusive(kb):
    df = label(kb, [2000, 3000, 2000, 3000, 2000], [49.9] * 5)
    assert df.Type2_Condition.tolist() == [False, True, True, True, False]


def test_type2_unverifiable_at_sequence_edge(kb):
    df = label(kb, [2500, 2500, 4000, 4000], [10, 10, 100, 100])
    assert not df.Type2_Condition.any() and df.Type2_Unverifiable.iloc[0]


def test_gap_makes_condition_unverifiable_not_matched(kb):
    wide = make_wide([2500] * 5, [10] * 5); wide.loc[2:, config.TIME_COL] += 100
    long = pp.wide_to_long(pp.clean_wide(wide)[0])
    df = label_documented_conditions(long, kb.detection_parameters).query("Panel == 'py'")
    assert not df.Type2_Condition.iloc[2] and df.Type2_Unverifiable.iloc[2]


def test_nan_voltage_is_neither_condition(kb):
    df = label(kb, [np.nan, 100, np.nan], [10, 10, 10])
    assert df.Type1_Condition.tolist() == [False, True, False]


# ---------------- ML ----------------
@pytest.fixture(scope="module")
def detector():
    return MLDetector.load()


def test_model_loads_and_predicts_valid_labels(detector, kb):
    long = pp.wide_to_long(pp.clean_wide(make_wide([4000, 5000, 1300, 0.0]))[0])
    out = detector.detect(long)
    assert set(out.ML_Status) <= {"Anomaly", "Normal", "Not scored"}
    assert out.ML_Score.notna().all() and set(out.ML_Prediction) <= {-1, 1}


def test_failed_panel_voltage_with_non_idle_current_is_flagged(detector):
    # 18.64 mA is the constant offset current read by the failed RAAVANA +Y panel
    long = pp.wide_to_long(pp.clean_wide(make_wide([1.5] * 20, [18.64] * 20))[0])
    py = detector.detect(long).query("Panel == 'py'")
    assert (py.ML_Status == "Anomaly").mean() > 0.9


def test_known_ml_limitation_failed_panel_with_idle_current_is_missed_by_ml_but_caught_by_rules(detector, kb):
    from src import pipeline
    clean, _ = pp.clean_wide(make_wide([1.5] * 30, [4.66] * 30))
    res = pipeline.analyze(clean, panels=["py"], detector=detector, kb=kb)
    p = res.panel_diagnoses[0]
    assert p.facts["ml_anomaly_count"] == 0                      # ML misses it ...
    assert p.diagnosis.conclusion == "possible_solar_panel_failure"  # ... the rule does not
    assert p.diagnosis.evidence_strength == "Moderate"           # and strength reflects no ML support


def test_missing_values_not_scored(detector):
    long = pp.wide_to_long(pp.clean_wide(make_wide([np.nan, 4000, 4000]))[0])
    out = detector.detect(long).query("Panel == 'py'").reset_index(drop=True)
    assert out.ML_Status[0] == "Not scored" and np.isnan(out.ML_Score[0])


def test_invalid_input_raises(detector):
    import pandas as pd
    with pytest.raises(InvalidTelemetryError):
        build_features(pd.DataFrame({"x": [1]}))


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        MLDetector.load(tmp_path / "no.joblib", tmp_path / "no2.joblib")


def test_expected_rate_exposed(detector):
    assert 0 < detector.expected_anomaly_rate < 1
