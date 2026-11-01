"""Conflict resolution between the diagnosis rules that fired.

Several rules can match the same panel window. The resolution strategy is
documented and deterministic:

1. **Priority** - lower number wins (documented faults before anomalies before
   insufficient-evidence before normal).
2. **Specificity** - the rule with more conditions wins.
3. **Severity** - higher severity wins.
4. **Evidence strength** - corroborated evidence wins.
5. rule id (stable tie-break).

If two *different* documented-fault conclusions match, the status becomes
``conflicting_evidence``; the top-ranked conclusion is still shown first, but
every match is retained for the explanation facility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .inference_engine import InferenceResult, RuleEvaluation
from .knowledge_base import KnowledgeBase

SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
STRENGTH_RANK = {"Strong": 3, "Moderate": 2, "Weak": 1, "Not applicable": 0, "Insufficient": 0}


@dataclass
class Candidate:
    rule_id: str
    rule_name: str
    conclusion: str
    fault_id: str | None
    category: str
    status: str
    priority: int
    severity: str
    specificity: int
    evidence_strength: str
    strengthened_by: list[str]
    evaluation: RuleEvaluation

    def rank_key(self) -> tuple:
        return (self.priority, -self.specificity, -SEVERITY_RANK.get(self.severity, 0),
                -STRENGTH_RANK.get(self.evidence_strength, 0), self.rule_id)


@dataclass
class Diagnosis:
    status: str                       # key of faults.json diagnosis_statuses
    status_label: str
    conclusion: str                   # e.g. possible_solar_panel_failure
    fault_id: str | None
    fault_name: str
    severity: str
    evidence_strength: str
    primary_rule_id: str | None
    candidates: list[Candidate] = field(default_factory=list)
    outranked: list[dict] = field(default_factory=list)
    not_selected: list[dict] = field(default_factory=list)
    matched_rule_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    suggested_action: str = ""
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


def _base_strength(category: str) -> str:
    return {"documented_fault": "Moderate", "anomaly": "Weak"}.get(category, "Not applicable")


def _candidate(ev: RuleEvaluation, kb: KnowledgeBase, facts: dict[str, Any]) -> Candidate:
    rule = ev.rule
    fault = kb.fault_by_id(rule.get("fault_id")) if rule.get("fault_id") else None
    category = fault["category"] if fault else "status"
    strengtheners = rule.get("strengthening_facts", [])
    strength = _base_strength(category)
    if strengtheners and category == "documented_fault":
        strength = "Strong" if all(facts.get(f) for f in strengtheners) else "Moderate"
    if category == "status" and rule.get("diagnosis") == "insufficient_evidence":
        strength = "Insufficient"
    return Candidate(
        rule_id=rule["rule_id"], rule_name=rule["name"], conclusion=rule["diagnosis"],
        fault_id=rule.get("fault_id"), category=category,
        status=fault["diagnosis_status"] if fault else "insufficient_evidence",
        priority=rule["priority"], severity=rule["severity"], specificity=len(rule["conditions"]),
        evidence_strength=strength, strengthened_by=[f for f in strengtheners if facts.get(f)],
        evaluation=ev,
    )


def resolve(inference: InferenceResult, kb: KnowledgeBase) -> Diagnosis:
    """Select the final diagnosis from an :class:`InferenceResult`."""
    statuses = kb.faults["diagnosis_statuses"]
    facts = inference.facts
    candidates = [
        _candidate(ev, kb, facts) for ev in inference.matched if ev.rule.get("is_diagnosis")
    ]
    candidates.sort(key=lambda c: c.rank_key())
    matched_ids = inference.matched_ids()

    # diagnosis rules that were not selected, with the reason
    not_selected: list[dict] = []
    for ev in inference.not_matched + inference.lacking_evidence:
        if not ev.rule.get("is_diagnosis"):
            continue
        if ev.lacks_evidence:
            reason = "required evidence unavailable: " + ", ".join(ev.missing_facts)
        else:
            failed = [c for c in ev.conditions if not c.satisfied]
            reason = "condition not satisfied: " + "; ".join(c.text() for c in failed)
        not_selected.append({"rule_id": ev.rule_id, "name": ev.rule["name"],
                             "conclusion": ev.rule["diagnosis"], "reason": reason})

    if not candidates:
        return Diagnosis(
            status="insufficient_evidence", status_label=statuses["insufficient_evidence"]["label"],
            conclusion="insufficient_evidence", fault_id="F004", fault_name="Insufficient evidence",
            severity="low", evidence_strength="Insufficient", primary_rule_id=None,
            not_selected=not_selected, matched_rule_ids=matched_ids,
            notes=["No diagnosis rule could be applied to the available evidence."],
            suggested_action="Provide voltage and current telemetry and run the anomaly detector.",
            facts=dict(facts),
        )

    top = candidates[0]
    status = top.status
    notes: list[str] = []
    documented = {c.conclusion for c in candidates if c.category == "documented_fault"}
    if len(documented) > 1:
        status = "conflicting_evidence"
        notes.append(
            "Different documented fault conditions matched in the same panel window: "
            + ", ".join(sorted(documented))
            + ". Rules are ranked by priority, specificity, severity and evidence strength; all matches are retained."
        )
    outranked = [
        {"rule_id": c.rule_id, "name": c.rule_name, "conclusion": c.conclusion,
         "reason": f"matched but outranked by {top.rule_id} (priority {c.priority} vs {top.priority}; "
                   f"specificity {c.specificity} vs {top.specificity})"}
        for c in candidates[1:]
    ]
    fault = kb.fault_by_id(top.fault_id) if top.fault_id else None
    return Diagnosis(
        status=status, status_label=statuses[status]["label"], conclusion=top.conclusion,
        fault_id=top.fault_id, fault_name=fault["name"] if fault else top.rule_name,
        severity=top.severity, evidence_strength=top.evidence_strength, primary_rule_id=top.rule_id,
        candidates=candidates, outranked=outranked, not_selected=not_selected,
        matched_rule_ids=matched_ids, notes=notes,
        suggested_action=top.evaluation.rule.get("suggested_action", ""), facts=dict(facts),
    )
