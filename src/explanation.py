"""Explanation facility: evidence collection and dynamic diagnostic reasoning.

Everything shown comes from the actual observations, ML output, facts and
matched rules of the diagnosed window. No values are invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .conflict_resolution import Diagnosis
from .inference_engine import InferenceResult
from .knowledge_base import KnowledgeBase

CAVEATS = [
    "A documented condition indicates a *possible* fault from telemetry only; it is not a confirmed hardware failure.",
    "The thresholds are research-specific conditions (SRC-1) and not universal engineering limits.",
    "The dataset contains no illumination/attitude signal, so operating conditions cannot be verified directly.",
]


@dataclass
class Explanation:
    title: str
    status_label: str
    summary: str
    observations: list[str] = field(default_factory=list)
    ml_result: str = ""
    matched_rules: list[dict] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    suggested_action: str = ""
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def to_markdown(self) -> str:
        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {i}" for i in items) if items else "- (none)"

        rules_md = "\n".join(
            f"- **{r['rule_id']}** - {r['name']} (source: {r['source']})\n"
            + "".join(f"    - {c}\n" for c in r["conditions"])
            for r in self.matched_rules
        ) or "- (none)"
        return (
            f"**{self.title}**\n\n{self.summary}\n\n"
            f"**Evidence**\n{bullets(self.observations)}\n- {self.ml_result}\n\n"
            f"**Matched rules**\n{rules_md}\n"
            f"**Reasoning**\n{bullets(self.reasoning)}\n\n"
            f"**Alternatives not selected**\n{bullets(self.alternatives)}\n\n"
            f"**Suggested inspection**\n{self.suggested_action}\n\n"
            f"**Caveats**\n{bullets(self.caveats)}"
        )


# --------------------------------------------------------------------------- #
def collect_evidence(window: pd.DataFrame, diagnosis: Diagnosis) -> list[dict]:
    """Concrete observed values supporting the diagnosis (for display and DB)."""
    ev: list[dict] = []
    n = len(window)
    ev.append({"feature_name": "samples_analysed", "observed_value": str(n),
               "evidence_description": f"{n} observations of this panel were analysed."})

    t1 = window[window["Type1_Condition"]]
    if len(t1):
        ev.append({"feature_name": "voltage_below_1200_mV",
                   "observed_value": f"{len(t1)} samples ({len(t1) / n:.1%}); V range {t1['Voltage_mV'].min():.2f} to {t1['Voltage_mV'].max():.2f} mV",
                   "evidence_description": f"Voltage satisfies V < 1200 mV (SRC-1 Eq.1); first at t = {t1['Elapsed_s'].min():.0f} s."})
    t2 = window[window["Type2_Condition"]]
    if len(t2):
        run = diagnosis.facts.get("type2_longest_run", "n/a")
        ev.append({"feature_name": "voltage_2000_3000_mV_with_low_current",
                   "observed_value": f"{len(t2)} samples ({len(t2) / n:.1%}); V {t2['Voltage_mV'].min():.0f}-{t2['Voltage_mV'].max():.0f} mV; I max {t2['Current_mA'].max():.2f} mA; longest consecutive run {run}",
                   "evidence_description": f"Voltage within 2000-3000 mV on t-1, t, t+1 and current < 50 mA at t (SRC-1 Eq.2); first at t = {t2['Elapsed_s'].min():.0f} s."})
    unv = window[window["Type2_Unverifiable"]]
    if len(unv):
        ev.append({"feature_name": "type2_candidate_unverifiable",
                   "observed_value": f"{len(unv)} samples",
                   "evidence_description": "Voltage/current fit Eq.2 but a neighbouring sample is missing or not contiguous, so the condition cannot be verified."})
    if "ML_Status" in window.columns:
        scored = window[window["ML_Status"].isin(["Anomaly", "Normal"])]
        flagged = scored[scored["ML_Status"] == "Anomaly"]
        if len(scored):
            desc = (f"Isolation Forest flagged {len(flagged)} of {len(scored)} scored samples ({len(flagged) / len(scored):.1%})."
                    + (f" Lowest anomaly score {flagged['ML_Score'].min():.3f}." if len(flagged) else ""))
            ev.append({"feature_name": "ml_anomaly_samples",
                       "observed_value": f"{len(flagged)} of {len(scored)}", "evidence_description": desc})
    rest = window[(~window["Type1_Condition"]) & (~window["Type2_Condition"])]
    if len(rest) and len(rest) < n:
        ev.append({"feature_name": "voltage_range_other_samples",
                   "observed_value": f"{rest['Voltage_mV'].min():.0f}-{rest['Voltage_mV'].max():.0f} mV",
                   "evidence_description": "Voltage range of samples that did not match a documented condition."})
    return ev


def generate_explanation(
    diagnosis: Diagnosis,
    inference: InferenceResult,
    kb: KnowledgeBase,
    window: pd.DataFrame,
    context: dict[str, str],
    evidence: list[dict],
) -> Explanation:
    facts = diagnosis.facts
    panel = context.get("panel_label", "")
    where = f"{context.get('satellite', '')} {panel} panel, session {context.get('session', '')}"
    title = f"Diagnosis: {diagnosis.status_label}" + (
        f" - {diagnosis.fault_name}" if diagnosis.status not in ("normal",) else "")

    # --- ML sentence ---------------------------------------------------------
    if not facts.get("ml_available"):
        ml_text = "The ML detector produced no result for this window."
    else:
        cnt, frac, exp = facts.get("ml_anomaly_count", 0), facts.get("ml_anomaly_fraction"), facts.get("ml_expected_fraction")
        ml_text = f"Isolation Forest flagged {cnt} of {facts['n_samples']} samples ({frac:.1%})"
        if exp is not None:
            ml_text += (f"; the model flags about {exp:.0%} of normal samples, so this window is "
                        + ("above" if frac > exp else "not above") + " its expected false-alarm rate.")
        else:
            ml_text += "."
        if facts.get("v_below_threshold_count"):
            ml_text += (f" It flagged {facts['ml_flagged_type1_samples']} of the "
                        f"{facts['v_below_threshold_count']} samples meeting the Type 1 condition.")
        if facts.get("type2_match_count"):
            ml_text += (f" It flagged {facts['ml_flagged_type2_samples']} of the "
                        f"{facts['type2_match_count']} samples meeting the Type 2 condition.")

    # --- matched rules with satisfied conditions ------------------------------
    matched_rules = []
    for ev in inference.matched:
        src = ev.rule["source"]
        ref = src["source_id"] + (f" {src['section']}" if src.get("section") else "") + (
            f" {src['equation']}" if src.get("equation") else "")
        matched_rules.append({
            "rule_id": ev.rule_id, "name": ev.rule["name"], "source": ref,
            "is_diagnosis": ev.rule.get("is_diagnosis", False),
            "conditions": [f"{c.description or c.fact}: {c.text()}" for c in ev.conditions],
        })

    # --- reasoning ---------------------------------------------------------------
    reasoning = [
        f"The inference engine converted the observations of the {where} into {len(facts)} facts and "
        f"applied {len(kb.rule_list)} rules by forward chaining ({inference.iterations} passes, "
        f"{len(inference.derivation_log)} rules fired)."
    ]
    for step in inference.derivation_log:
        reasoning.append(f"Step {step['step']}: {step['rule_id']} asserted `{step['asserted_fact']}` because "
                         + "; ".join(step["because"]) + ".")
    if diagnosis.primary:
        p = diagnosis.primary
        reasoning.append(
            f"Conflict resolution ranked {p.rule_id} first (priority {p.priority}, {p.specificity} conditions, "
            f"severity {p.severity}, evidence strength {p.evidence_strength})."
            + (f" Corroborated by: {', '.join(p.strengthened_by)}." if p.strengthened_by else ""))
    reasoning += diagnosis.notes

    alternatives = [f"{o['rule_id']} ({o['conclusion']}): {o['reason']}" for o in diagnosis.outranked]
    alternatives += [f"{a['rule_id']} ({a['conclusion']}): {a['reason']}" for a in diagnosis.not_selected]

    # --- summary ------------------------------------------------------------------
    if diagnosis.status == "possible_documented_fault":
        summary = (f"A documented failure condition was satisfied on the {where}. This is a *possible* fault "
                   f"(evidence strength: {diagnosis.evidence_strength}); it is not confirmed.")
    elif diagnosis.status == "conflicting_evidence":
        summary = f"More than one documented fault condition matched on the {where}; all matches are listed below."
    elif diagnosis.status == "anomalous_behaviour":
        summary = (f"The ML model detected unusual behaviour on the {where}, but no documented fault condition "
                   "explains it. This is anomaly detection, not a physical fault diagnosis.")
    elif diagnosis.status == "insufficient_evidence":
        summary = f"The available evidence for the {where} is insufficient to support a diagnosis."
    else:
        summary = f"No documented condition matched and no ML anomaly was found on the {where}."

    caveats = CAVEATS if diagnosis.status in ("possible_documented_fault", "conflicting_evidence") else CAVEATS[1:]
    return Explanation(
        title=title, status_label=diagnosis.status_label, summary=summary,
        observations=[f"{e['feature_name'].replace('_', ' ')}: {e['observed_value']} - {e['evidence_description']}" for e in evidence],
        ml_result=ml_text, matched_rules=matched_rules, reasoning=reasoning, alternatives=alternatives,
        suggested_action=diagnosis.suggested_action or "No action required.", caveats=caveats,
    )
