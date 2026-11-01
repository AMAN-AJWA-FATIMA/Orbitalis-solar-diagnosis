"""Dynamic knowledge graph (NetworkX) built from the knowledge base and the
actual diagnosis evidence.

Node layers (left to right):
0 Solar panel -> 1 Sensor measurement -> 2 Observed condition / ML anomaly ->
3 Diagnostic rule -> 4 Possible fault / status -> 5 Suggested inspection action.

Node ids are deterministic so nothing is duplicated: rules, faults and actions
are shared between panels, panel-specific nodes carry the satellite/session/panel
in their id.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from .knowledge_base import KnowledgeBase

NODE_TYPES = {
    "solar_panel": "Solar panel", "measurement": "Sensor measurement", "condition": "Observed condition",
    "anomaly": "ML anomaly", "rule": "Diagnostic rule", "fault": "Possible fault / status", "action": "Inspection action",
}
LAYER = {"solar_panel": 0, "measurement": 1, "condition": 2, "anomaly": 2, "rule": 3, "fault": 4, "action": 5}

# Which measurement feeds which observed condition (from the rule definitions).
_CONDITION_FACTS = {
    "type1_condition_present": ("voltage", "Voltage < 1200 mV"),
    "type2_condition_present": ("voltage_current", "V 2000-3000 mV with I < 50 mA"),
    "type2_unverifiable": ("voltage_current", "Type 2 candidate (unverifiable)"),
    "type2_minimal_pattern_only": ("voltage_current", "Type 2 minimum pattern only"),
}


def _add(G: nx.DiGraph, node_id: str, ntype: str, label: str, **attrs) -> None:
    if node_id not in G:
        G.add_node(node_id, ntype=ntype, label=label, layer=LAYER[ntype], on_path=False, **attrs)


def _edge(G: nx.DiGraph, a: str, b: str, label: str) -> None:
    if not G.has_edge(a, b):
        G.add_edge(a, b, label=label, on_path=False)


def knowledge_structure_graph(kb: KnowledgeBase) -> nx.DiGraph:
    """Static view of the knowledge base (rules -> faults -> actions)."""
    G = nx.DiGraph()
    for fault in kb.fault_list:
        _add(G, f"fault:{fault['fault_id']}", "fault", fault["name"], fault_id=fault["fault_id"])
        _add(G, f"action:{fault['fault_id']}", "action", "Inspection action")
        _edge(G, f"fault:{fault['fault_id']}", f"action:{fault['fault_id']}", "recommends")
    for rule in kb.rule_list:
        if not rule.get("is_diagnosis"):
            continue
        _add(G, f"rule:{rule['rule_id']}", "rule", f"{rule['rule_id']} {rule['name']}")
        _edge(G, f"rule:{rule['rule_id']}", f"fault:{rule['fault_id']}", "concludes")
    return G


def build_knowledge_graph(
    kb: KnowledgeBase,
    panel_diagnoses: Iterable,            # iterable of pipeline.PanelDiagnosis
    satellite: str | None = None,
    panel: str | None = None,
    include_normal: bool = False,
    highlight: tuple[str, str, str] | None = None,   # (satellite, session, panel)
) -> nx.DiGraph:
    """Graph for the supplied panel diagnoses, filtered by satellite/panel."""
    G = nx.DiGraph()
    for pd_ in panel_diagnoses:
        if satellite and pd_.satellite != satellite:
            continue
        if panel and pd_.panel != panel:
            continue
        d = pd_.diagnosis
        if d.status == "normal" and not include_normal:
            continue
        facts = d.facts
        key = f"{pd_.satellite}:{pd_.session}:{pd_.panel}"
        is_hl = highlight == (pd_.satellite, pd_.session, pd_.panel)
        path: list[str] = []

        p_id = f"panel:{key}"
        _add(G, p_id, "solar_panel", f"{pd_.satellite} {pd_.panel_label} panel\n{pd_.session}")
        path.append(p_id)
        m_ids = {}
        for m, mlabel in (("voltage", "Voltage (mV)"), ("current", "Current (mA)")):
            m_ids[m] = f"meas:{key}:{m}"
            _add(G, m_ids[m], "measurement", mlabel)
            _edge(G, p_id, m_ids[m], "measures")

        cond_nodes: dict[str, str] = {}
        for fact, (sensors, text) in _CONDITION_FACTS.items():
            if facts.get(fact) or (fact == "type2_unverifiable" and facts.get("type2_unverifiable_count")
                                    and not facts.get("type2_condition_present")):
                count = {"type1_condition_present": facts.get("v_below_threshold_count"),
                         "type2_condition_present": facts.get("type2_match_count"),
                         "type2_unverifiable": facts.get("type2_unverifiable_count"),
                         "type2_minimal_pattern_only": facts.get("type2_longest_run")}[fact]
                c_id = f"cond:{key}:{fact}"
                _add(G, c_id, "condition", f"{text}\n({'longest run' if fact == 'type2_minimal_pattern_only' else 'n'}={count})")
                for s in sensors.split("_"):
                    _edge(G, m_ids[s], c_id, "shows")
                cond_nodes[fact] = c_id
        if facts.get("ml_anomaly_count"):
            a_id = f"anom:{key}"
            _add(G, a_id, "anomaly", f"Isolation Forest anomaly\n(n={facts['ml_anomaly_count']})")
            for s in ("voltage", "current"):
                _edge(G, m_ids[s], a_id, "flagged")
            cond_nodes["ml_anomaly_present"] = a_id

        # diagnosis rules that fired in this window (observation rules R001-R006 are
        # internal derivation steps and are shown through the condition nodes)
        for cand in d.candidates:
            if cand.rule_id == "R050":      # normal operation has no evidence path
                continue
            rule = cand.evaluation.rule
            r_id = f"rule:{cand.rule_id}"
            _add(G, r_id, "rule", f"{cand.rule_id} {rule['name']}")
            linked = False
            for cond in rule["conditions"]:
                src = cond_nodes.get(cond["fact"])
                if src and cond["operator"] == "is_true":
                    _edge(G, src, r_id, "satisfies")
                    linked = True
            if cand.strengthened_by and "ml_anomaly_present" in cond_nodes:
                _edge(G, cond_nodes["ml_anomaly_present"], r_id, "corroborates")
                linked = True
            if not linked:
                _edge(G, p_id, r_id, "evaluated for")
            fault = kb.fault_by_id(cand.fault_id) if cand.fault_id else None
            if fault:
                f_id = f"fault:{fault['fault_id']}"
                _add(G, f_id, "fault", fault["name"], fault_id=fault["fault_id"])
                act = f"action:{fault['fault_id']}"
                _add(G, act, "action", rule.get("suggested_action", "Inspect")[:70])
                _edge(G, r_id, f_id, "concludes")
                _edge(G, f_id, act, "recommends")

        # highlight the evidence path of the primary diagnosis
        if is_hl and d.primary:
            prim = f"rule:{d.primary.rule_id}"
            if prim in G:
                nodes_on = {p_id, prim}
                for anc in nx.ancestors(G, prim):
                    if anc.startswith((f"meas:{key}", f"cond:{key}", f"anom:{key}", "rule:")) or anc == p_id:
                        nodes_on.add(anc)
                for desc in nx.descendants(G, prim):
                    nodes_on.add(desc)
                for n in nodes_on:
                    G.nodes[n]["on_path"] = True
                for a, b in G.edges:
                    if a in nodes_on and b in nodes_on:
                        G.edges[a, b]["on_path"] = True
    return G
