"""Loading and validation of the JSON knowledge base."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

VALID_OPERATORS = {">", ">=", "<", "<=", "==", "!=", "is_true", "is_false"}
REQUIRED_RULE_FIELDS = (
    "rule_id", "name", "layer", "conditions", "logic", "conclusion", "priority",
    "severity", "description", "evidence_requirements", "source", "suggested_action",
)
VALID_SEVERITIES = {"none", "low", "medium", "high"}


class KnowledgeBaseError(ValueError):
    """Raised when a knowledge-base file is missing or malformed."""


@dataclass
class KnowledgeBase:
    components: dict[str, Any]
    faults: dict[str, Any]
    rules: dict[str, Any]
    errors: list[str] = field(default_factory=list)

    # convenience accessors ---------------------------------------------------
    @property
    def rule_list(self) -> list[dict]:
        return self.rules.get("rules", [])

    @property
    def fault_list(self) -> list[dict]:
        return self.faults.get("faults", [])

    def fault_by_id(self, fault_id: str) -> dict | None:
        return next((f for f in self.fault_list if f["fault_id"] == fault_id), None)

    def fault_by_conclusion(self, conclusion: str) -> dict | None:
        return next((f for f in self.fault_list if f.get("conclusion") == conclusion), None)

    def rule_by_id(self, rule_id: str) -> dict | None:
        return next((r for r in self.rule_list if r["rule_id"] == rule_id), None)

    @property
    def detection_parameters(self) -> dict[str, Any]:
        return self.faults["detection_parameters"]

    @property
    def sources(self) -> list[dict]:
        return self.components.get("sources", [])


def _read_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise KnowledgeBaseError(f"Knowledge base file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise KnowledgeBaseError(f"Knowledge base file is empty: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise KnowledgeBaseError(f"{path} must contain a JSON object.")
    return data


def validate_rule(rule: dict, known_facts: set[str]) -> list[str]:
    """Structural validation of one rule; returns a list of problems."""
    rid = rule.get("rule_id", "<missing id>")
    problems: list[str] = []
    for key in REQUIRED_RULE_FIELDS:
        if key not in rule:
            problems.append(f"{rid}: missing field '{key}'")
    if problems:
        return problems
    if not rule["conditions"]:
        problems.append(f"{rid}: no conditions")
    if rule["logic"] not in ("AND", "OR"):
        problems.append(f"{rid}: logic must be AND or OR")
    if rule["severity"] not in VALID_SEVERITIES:
        problems.append(f"{rid}: invalid severity '{rule['severity']}'")
    if not isinstance(rule["priority"], int):
        problems.append(f"{rid}: priority must be an integer")
    concl = rule["conclusion"]
    if not isinstance(concl, dict) or "fact" not in concl or "value" not in concl:
        problems.append(f"{rid}: conclusion needs 'fact' and 'value'")
    for c in rule["conditions"]:
        if "fact" not in c or "operator" not in c:
            problems.append(f"{rid}: condition needs 'fact' and 'operator'")
            continue
        if c["operator"] not in VALID_OPERATORS:
            problems.append(f"{rid}: unknown operator '{c['operator']}'")
        elif c["operator"] not in ("is_true", "is_false") and "value" not in c and "value_fact" not in c:
            problems.append(f"{rid}: operator '{c['operator']}' needs a 'value' or 'value_fact'")
        if "value_fact" in c and c["value_fact"] not in known_facts:
            problems.append(f"{rid}: condition compares with undefined fact '{c['value_fact']}'")
        if c["fact"] not in known_facts:
            problems.append(f"{rid}: condition uses undefined fact '{c['fact']}'")
    for fact in rule["evidence_requirements"]:
        if fact not in known_facts:
            problems.append(f"{rid}: evidence requirement uses undefined fact '{fact}'")
    return problems


def validate_knowledge_base(kb: KnowledgeBase) -> list[str]:
    """Cross-file validation: references between rules, faults and components."""
    problems: list[str] = []
    known_facts = set(kb.rules.get("fact_definitions", {}))
    ids = [r.get("rule_id") for r in kb.rule_list]
    if len(ids) != len(set(ids)):
        problems.append("Duplicate rule_id values")
    component_ids = {c["component_id"] for c in kb.components.get("components", [])}
    fault_ids = {f["fault_id"] for f in kb.fault_list}
    for rule in kb.rule_list:
        problems += validate_rule(rule, known_facts)
        for fid in rule.get("related_fault_ids", []) + ([rule["fault_id"]] if rule.get("fault_id") else []):
            if fid not in fault_ids:
                problems.append(f"{rule.get('rule_id')}: references unknown fault '{fid}'")
        for fact in rule.get("strengthening_facts", []):
            if fact not in known_facts:
                problems.append(f"{rule.get('rule_id')}: unknown strengthening fact '{fact}'")
    for fault in kb.fault_list:
        for comp in fault.get("related_components", []):
            if comp not in component_ids:
                problems.append(f"{fault['fault_id']}: references unknown component '{comp}'")
        if fault.get("diagnosis_status") not in kb.faults.get("diagnosis_statuses", {}):
            problems.append(f"{fault['fault_id']}: unknown diagnosis_status")
    return problems


def load_knowledge_base(
    components_path: Path = config.COMPONENTS_JSON,
    faults_path: Path = config.FAULTS_JSON,
    rules_path: Path = config.RULES_JSON,
    validate: bool = True,
) -> KnowledgeBase:
    kb = KnowledgeBase(
        components=_read_json(components_path),
        faults=_read_json(faults_path),
        rules=_read_json(rules_path),
    )
    if validate:
        kb.errors = validate_knowledge_base(kb)
        if kb.errors:
            raise KnowledgeBaseError("Invalid knowledge base:\n- " + "\n- ".join(kb.errors))
    return kb
