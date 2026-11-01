import pytest

from src.conflict_resolution import resolve
from src.inference_engine import ForwardChainingEngine
from tests.test_inference_engine import BASE


@pytest.fixture(scope="module")
def engine(kb):
    return ForwardChainingEngine.from_knowledge_base(kb)


def diag(engine, kb, **upd):
    return resolve(engine.infer({**BASE, **upd}), kb)


def test_normal(engine, kb):
    assert diag(engine, kb).status == "normal"


def test_documented_fault_with_ml_is_strong(engine, kb):
    d = diag(engine, kb, v_below_threshold_count=9, ml_type1_flag_rate=1.0)
    assert (d.status, d.conclusion, d.evidence_strength) == ("possible_documented_fault", "possible_solar_panel_failure", "Strong")


def test_documented_fault_without_ml_is_moderate(engine, kb):
    d = diag(engine, kb, type2_match_count=5, type2_longest_run=5)
    assert d.evidence_strength == "Moderate" and d.conclusion == "possible_solar_cell_failure"


def test_ml_only_is_anomalous_behaviour_not_fault(engine, kb):
    d = diag(engine, kb, ml_anomaly_count=400, ml_anomaly_fraction=0.4)
    assert d.status == "anomalous_behaviour" and d.evidence_strength == "Weak"


def test_conflicting_documented_faults_retained(engine, kb):
    d = diag(engine, kb, v_below_threshold_count=2, type2_match_count=4, type2_longest_run=4)
    assert d.status == "conflicting_evidence" and d.primary.rule_id == "R010"
    assert any(c.rule_id == "R020" for c in d.candidates) and d.outranked and d.notes


def test_priority_beats_lower_priority(engine, kb):
    d = diag(engine, kb, v_below_threshold_count=2, ml_anomaly_count=500, ml_anomaly_fraction=0.5)
    assert d.primary.rule_id == "R010"


def test_insufficient_evidence_cases(engine, kb):
    assert diag(engine, kb, type2_unverifiable_count=1).status == "insufficient_evidence"
    assert diag(engine, kb, type2_match_count=1, type2_longest_run=1).status == "insufficient_evidence"


def test_matched_rules_all_retained(engine, kb):
    d = diag(engine, kb, v_below_threshold_count=3, ml_type1_flag_rate=1.0, ml_anomaly_fraction=0.9, ml_anomaly_count=900)
    assert {"R001", "R004", "R005", "R010"} <= set(d.matched_rule_ids)
    assert d.not_selected  # alternatives carry a reason


def test_no_diagnosis_rule_applicable(kb):
    from src.inference_engine import InferenceResult
    empty = InferenceResult({}, [], [], [], [], 0)
    d = resolve(empty, kb)
    assert d.status == "insufficient_evidence" and d.primary_rule_id is None
