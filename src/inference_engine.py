"""Generic forward-chaining inference engine driven by ``rules.json``.

* Observations of one panel window are converted into *facts*
  (:func:`build_panel_facts`).
* Rules are read from the knowledge base; the engine contains no fault-specific
  logic.
* Execution is stratified by the rule ``layer``: inside a layer the engine
  repeats "evaluate all unfired rules -> assert conclusions" until no new fact
  can be produced (a fixed point); then the next layer starts. A derived fact
  that was never asserted counts as false (closed world); stratification
  guarantees that such negations are evaluated only after every rule that
  could produce the fact has run.
* Rules whose required base facts are missing are reported as lacking evidence
  instead of silently failing.
"""
from __future__ import annotations

import operator as op
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .knowledge_base import KnowledgeBase

_COMPARATORS = {
    ">": op.gt, ">=": op.ge, "<": op.lt, "<=": op.le, "==": op.eq, "!=": op.ne,
}


def _fmt(value: Any) -> str:
    return f"{value:.3g}" if isinstance(value, float) else str(value)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class ConditionResult:
    fact: str
    operator: str
    expected: Any
    actual: Any
    satisfied: bool
    description: str = ""

    expected_label: str = ""

    def text(self) -> str:
        if self.operator in ("is_true", "is_false"):
            exp = ""
        else:
            exp = f" {self.expected_label or self.expected}"
        return f"{self.fact} {self.operator}{exp} (observed: {_fmt(self.actual)})"


@dataclass
class RuleEvaluation:
    rule: dict
    matched: bool
    conditions: list[ConditionResult]
    missing_facts: list[str] = field(default_factory=list)
    iteration: int = 0

    @property
    def rule_id(self) -> str:
        return self.rule["rule_id"]

    @property
    def lacks_evidence(self) -> bool:
        return bool(self.missing_facts)


@dataclass
class InferenceResult:
    facts: dict[str, Any]
    matched: list[RuleEvaluation]
    not_matched: list[RuleEvaluation]
    lacking_evidence: list[RuleEvaluation]
    derivation_log: list[dict]
    iterations: int

    def matched_ids(self) -> list[str]:
        return [m.rule_id for m in self.matched]


# --------------------------------------------------------------------------- #
# Facts from observations
# --------------------------------------------------------------------------- #
def longest_run(mask: np.ndarray, contiguous_next: np.ndarray) -> int:
    """Longest run of ``True`` samples that are consecutive in time."""
    best = run = 0
    for flag, contig in zip(mask, contiguous_next):
        run = (run + 1 if run else 1) if flag else 0
        best = max(best, run)
        if flag and not contig:
            run = 0
    return best


def build_panel_facts(window: pd.DataFrame, expected_ml_rate: float | None = None) -> dict[str, Any]:
    """Convert the (labelled, ML-scored) observations of one panel window into facts.

    ``window`` needs the columns produced by ``labeling.label_documented_conditions``
    and, when available, ``ML_Status``. ``expected_ml_rate`` is the share of samples
    the model flags on normal data (its contamination setting).
    """
    window = window.sort_values("Elapsed_s")
    facts: dict[str, Any] = {"n_samples": int(len(window))}
    t1, t2 = window["Type1_Condition"], window["Type2_Condition"]
    facts["v_below_threshold_count"] = int(t1.sum())
    facts["type2_match_count"] = int(t2.sum())
    facts["type2_unverifiable_count"] = int(window["Type2_Unverifiable"].sum())
    facts["type2_longest_run"] = longest_run(t2.to_numpy(), window["Next_contiguous"].to_numpy())

    facts["ml_expected_fraction"] = expected_ml_rate
    if "ML_Status" in window.columns:
        scored = window["ML_Status"].isin(["Anomaly", "Normal"])
        flagged = window["ML_Status"] == "Anomaly"
        facts["ml_available"] = bool(scored.any())
        facts["ml_anomaly_count"] = int(flagged.sum())
        facts["ml_anomaly_fraction"] = float(flagged.sum() / scored.sum()) if scored.any() else None
        n1, n2 = int(t1.sum()), int(t2.sum())
        facts["ml_flagged_type1_samples"] = int((flagged & t1).sum())
        facts["ml_flagged_type2_samples"] = int((flagged & t2).sum())
        facts["ml_type1_flag_rate"] = facts["ml_flagged_type1_samples"] / n1 if n1 else 0.0
        facts["ml_type2_flag_rate"] = facts["ml_flagged_type2_samples"] / n2 if n2 else 0.0
    else:
        facts["ml_available"] = False
        for key in ("ml_anomaly_count", "ml_anomaly_fraction", "ml_flagged_type1_samples",
                    "ml_flagged_type2_samples", "ml_type1_flag_rate", "ml_type2_flag_rate"):
            facts[key] = None
    return facts


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class ForwardChainingEngine:
    def __init__(self, rules: list[dict], fact_definitions: dict[str, dict] | None = None, max_iterations: int = 50):
        self.rules = sorted(rules, key=lambda r: (r.get("layer", 1), r.get("priority", 99), r["rule_id"]))
        self.derived_facts = {r["conclusion"]["fact"] for r in self.rules}
        self.max_iterations = max_iterations
        self.fact_definitions = fact_definitions or {}

    @classmethod
    def from_knowledge_base(cls, kb: KnowledgeBase) -> "ForwardChainingEngine":
        return cls(kb.rule_list, kb.rules.get("fact_definitions", {}))

    # -- condition / rule evaluation -----------------------------------------
    def _actual(self, facts: dict[str, Any], fact: str) -> Any:
        value = facts.get(fact)
        if value is None and fact in self.derived_facts:
            return False  # closed world for derived facts
        return value

    def _eval_condition(self, cond: dict, facts: dict[str, Any]) -> tuple[ConditionResult, bool]:
        actual = self._actual(facts, cond["fact"])
        operator_name = cond["operator"]
        expected = cond.get("value")
        expected_label = ""
        if "value_fact" in cond:
            ref = cond["value_fact"]
            expected = self._actual(facts, ref)
            expected_label = f"{ref} ({_fmt(expected)})"
        unknown = actual is None or (operator_name not in ("is_true", "is_false") and expected is None)
        satisfied = False
        if not unknown:
            if operator_name == "is_true":
                satisfied = bool(actual) is True
            elif operator_name == "is_false":
                satisfied = bool(actual) is False
            else:
                satisfied = bool(_COMPARATORS[operator_name](actual, expected))
        return ConditionResult(cond["fact"], operator_name, expected, actual, satisfied,
                               cond.get("description", ""), expected_label), unknown

    def evaluate_rule(self, rule: dict, facts: dict[str, Any], iteration: int = 0) -> RuleEvaluation:
        results, unknown_facts = [], []
        for cond in rule["conditions"]:
            res, unknown = self._eval_condition(cond, facts)
            results.append(res)
            if unknown:
                unknown_facts.append(cond["fact"])
        required_missing = [
            f for f in rule.get("evidence_requirements", [])
            if facts.get(f) is None and f not in self.derived_facts
        ]
        missing = sorted(set(required_missing))
        flags = [r.satisfied for r in results]
        matched = (all(flags) if rule["logic"] == "AND" else any(flags)) and not missing
        return RuleEvaluation(rule, matched, results, missing, iteration)

    # -- main loop ---------------------------------------------------------------
    def infer(self, initial_facts: dict[str, Any]) -> InferenceResult:
        facts = dict(initial_facts)
        fired: dict[str, RuleEvaluation] = {}
        log: list[dict] = []
        iterations = 0
        for layer in sorted({r.get("layer", 1) for r in self.rules}):
            layer_rules = [r for r in self.rules if r.get("layer", 1) == layer]
            changed = True
            while changed:
                changed = False
                iterations += 1
                if iterations > self.max_iterations:
                    raise RuntimeError("Forward chaining did not reach a fixed point.")
                for rule in layer_rules:
                    if rule["rule_id"] in fired:
                        continue
                    ev = self.evaluate_rule(rule, facts, iterations)
                    if not ev.matched:
                        continue
                    fired[rule["rule_id"]] = ev
                    fact, value = rule["conclusion"]["fact"], rule["conclusion"]["value"]
                    if facts.get(fact) != value:
                        facts[fact] = value
                        changed = True
                    log.append({
                        "step": len(log) + 1, "layer": layer, "iteration": iterations,
                        "rule_id": rule["rule_id"], "rule_name": rule["name"],
                        "asserted_fact": fact, "value": value,
                        "because": [c.text() for c in ev.conditions if c.satisfied],
                    })
        not_matched, lacking = [], []
        for rule in self.rules:
            if rule["rule_id"] in fired:
                continue
            ev = self.evaluate_rule(rule, facts, iterations)
            (lacking if ev.lacks_evidence else not_matched).append(ev)
        return InferenceResult(
            facts=facts, matched=list(fired.values()), not_matched=not_matched,
            lacking_evidence=lacking, derivation_log=log, iterations=iterations,
        )
