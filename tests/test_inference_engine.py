import pytest

from src.inference_engine import ForwardChainingEngine

BASE = dict(n_samples=1000, v_below_threshold_count=0, type2_match_count=0, type2_unverifiable_count=0,
            type2_longest_run=0, ml_available=True, ml_anomaly_count=0, ml_anomaly_fraction=0.05,
            ml_expected_fraction=0.15, ml_flagged_type1_samples=0, ml_flagged_type2_samples=0,
            ml_type1_flag_rate=0.0, ml_type2_flag_rate=0.0)


@pytest.fixture(scope="module")
def engine(kb):
    return ForwardChainingEngine.from_knowledge_base(kb)


def run(engine, **upd):
    return engine.infer({**BASE, **upd})


def test_no_match_means_normal(engine):
    assert run(engine).matched_ids() == ["R050"]


def test_single_rule_chain_fires_in_order(engine):
    r = run(engine, v_below_threshold_count=5, ml_type1_flag_rate=1.0, ml_anomaly_fraction=0.9)
    ids = r.matched_ids()
    assert ids.index("R001") < ids.index("R005") < ids.index("R010")
    assert r.facts["type1_condition_present"] and r.facts["type1_ml_corroborated"]


def test_forward_chaining_derives_new_facts_until_fixpoint(engine):
    r = run(engine, type2_match_count=4, type2_longest_run=3)
    assert r.facts["type2_condition_present"] is True and "R020" in r.matched_ids()
    assert not r.facts.get("type2_minimal_pattern_only")
    assert len(r.derivation_log) == len(r.matched)


def test_minimal_type2_pattern_blocks_fault_rule(engine):
    r = run(engine, type2_match_count=1, type2_longest_run=1)
    assert "R007" in r.matched_ids() and "R031" in r.matched_ids() and "R020" not in r.matched_ids()


def test_multiple_rules_match_together(engine):
    r = run(engine, v_below_threshold_count=2, type2_match_count=3, type2_longest_run=3)
    assert {"R010", "R020"} <= set(r.matched_ids())


def test_ml_rate_must_exceed_expected(engine):
    assert "R004" not in run(engine, ml_anomaly_count=100, ml_anomaly_fraction=0.10).matched_ids()
    assert "R040" in run(engine, ml_anomaly_count=300, ml_anomaly_fraction=0.30).matched_ids()


def test_unverifiable_candidate(engine):
    assert {"R003", "R030"} <= set(run(engine, type2_unverifiable_count=2).matched_ids())


def test_missing_evidence_reported_not_silently_failed(engine):
    r = run(engine, ml_available=False, ml_anomaly_count=None, ml_anomaly_fraction=None,
            ml_type1_flag_rate=None, ml_type2_flag_rate=None, ml_expected_fraction=None)
    assert "R051" in r.matched_ids() and "R050" not in r.matched_ids()
    assert any(ev.rule_id == "R004" and ev.missing_facts for ev in r.lacking_evidence)


def test_engine_is_data_driven_generic():
    rules = [{"rule_id": "X1", "name": "a", "layer": 1, "logic": "AND", "priority": 1,
              "conditions": [{"fact": "t", "operator": ">", "value": 5}], "evidence_requirements": ["t"],
              "conclusion": {"fact": "hot", "value": True}},
             {"rule_id": "X2", "name": "b", "layer": 2, "logic": "AND", "priority": 1,
              "conditions": [{"fact": "hot", "operator": "is_true"}], "evidence_requirements": [],
              "conclusion": {"fact": "alarm", "value": True}}]
    eng = ForwardChainingEngine(rules)
    assert eng.infer({"t": 9}).facts["alarm"] is True
    assert "alarm" not in eng.infer({"t": 1}).facts
