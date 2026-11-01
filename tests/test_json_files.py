import copy
import json

import pytest

from src import config
from src.knowledge_base import KnowledgeBase, KnowledgeBaseError, load_knowledge_base, validate_knowledge_base


def test_all_files_are_valid_json_and_load(kb):
    assert len(kb.rule_list) >= 1 and kb.errors == []
    for path in (config.COMPONENTS_JSON, config.FAULTS_JSON, config.RULES_JSON):
        json.loads(path.read_text(encoding="utf-8"))


def test_rules_have_required_fields_and_unique_ids(kb):
    ids = [r["rule_id"] for r in kb.rule_list]
    assert len(ids) == len(set(ids))


def test_rule_references_resolve(kb):
    fault_ids = {f["fault_id"] for f in kb.fault_list}
    for r in kb.rule_list:
        if r["fault_id"]:
            assert r["fault_id"] in fault_ids


def test_faults_reference_existing_components(kb):
    comps = {c["component_id"] for c in kb.components["components"]}
    for f in kb.fault_list:
        assert set(f["related_components"]) <= comps


def test_documented_thresholds_match_paper(kb):
    p = kb.detection_parameters
    assert p["type1_solar_panel_failure"]["voltage_below_mv"] == 1200
    t2 = p["type2_solar_cell_failure"]
    assert (t2["voltage_min_mv"], t2["voltage_max_mv"], t2["current_below_ma"], t2["consecutive_samples_required"]) == (2000, 3000, 50, 3)


def test_no_battery_components(kb):
    text = json.dumps(kb.components["components"])
    assert "Vbat" not in text and "Ibatt" not in text


def _kb_with(kb, mutate):
    rules = copy.deepcopy(kb.rules)
    mutate(rules)
    return KnowledgeBase(kb.components, kb.faults, rules)


def test_missing_rule_field_detected(kb):
    bad = _kb_with(kb, lambda r: r["rules"][0].pop("conditions"))
    assert any("conditions" in e for e in validate_knowledge_base(bad))


def test_unknown_operator_and_fact_detected(kb):
    def mut(r):
        r["rules"][0]["conditions"][0]["operator"] = "~~"
        r["rules"][1]["conditions"][0]["fact"] = "no_such_fact"
    errs = validate_knowledge_base(_kb_with(kb, mut))
    assert any("unknown operator" in e for e in errs) and any("undefined fact" in e for e in errs)


def test_unknown_fault_reference_detected(kb):
    errs = validate_knowledge_base(_kb_with(kb, lambda r: r["rules"][0].__setitem__("fault_id", "F999")))
    assert any("F999" in e for e in errs)


def test_empty_and_invalid_files_raise(tmp_path):
    empty = tmp_path / "e.json"; empty.write_text("")
    broken = tmp_path / "b.json"; broken.write_text("{oops")
    with pytest.raises(KnowledgeBaseError):
        load_knowledge_base(components_path=empty)
    with pytest.raises(KnowledgeBaseError):
        load_knowledge_base(components_path=broken)
