
"""Dynamic knowledge graph for ORBITALIS.

Graph layers:
    0. Solar panel
    1. Sensor measurement
    2. Observed condition / ML anomaly
    3. Diagnostic rule
    4. Possible fault / status
    5. Suggested inspection action

Design principles:
    - Deterministic node IDs prevent duplicate entities.
    - Evidence nodes are specific to satellite/session/panel.
    - Rules, faults, and actions are shared across panels.
    - Multiple distinct actions for the same fault are preserved.
    - Graph construction is separated from evidence interpretation.
    - Primary diagnosis paths can be highlighted independently.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import networkx as nx

from .knowledge_base import KnowledgeBase


# ---------------------------------------------------------------------
# Graph configuration
# ---------------------------------------------------------------------

NODE_TYPES = {
    "solar_panel": "Solar panel",
    "measurement": "Sensor measurement",
    "condition": "Observed condition",
    "anomaly": "ML anomaly",
    "rule": "Diagnostic rule",
    "fault": "Possible fault / status",
    "action": "Inspection action",
}

LAYER = {
    "solar_panel": 0,
    "measurement": 1,
    "condition": 2,
    "anomaly": 2,
    "rule": 3,
    "fault": 4,
    "action": 5,
}

MEASUREMENTS = {
    "voltage": "Voltage (mV)",
    "current": "Current (mA)",
}

# Each entry defines:
# fact name -> (required measurements, human-readable description, count field)
CONDITION_DEFINITIONS = {
    "type1_condition_present": {
        "measurements": ("voltage",),
        "label": "Voltage below 1200 mV",
        "count_field": "v_below_threshold_count",
        "count_label": "samples",
    },
    "type2_condition_present": {
        "measurements": ("voltage", "current"),
        "label": "Voltage 2000–3000 mV with current below 50 mA",
        "count_field": "type2_match_count",
        "count_label": "matches",
    },
    "type2_unverifiable": {
        "measurements": ("voltage", "current"),
        "label": "Type 2 candidate (unverifiable)",
        "count_field": "type2_unverifiable_count",
        "count_label": "candidates",
    },
    "type2_minimal_pattern_only": {
        "measurements": ("voltage", "current"),
        "label": "Type 2 minimum pattern only",
        "count_field": "type2_longest_run",
        "count_label": "longest run",
    },
}


# ---------------------------------------------------------------------
# Deterministic identifiers
# ---------------------------------------------------------------------

def _panel_key(satellite: str, session: str, panel: str) -> str:
    """Create a stable identifier for a panel within a dataset session."""
    return f"{satellite}:{session}:{panel}"


def _panel_node_id(key: str) -> str:
    return f"panel:{key}"


def _measurement_node_id(key: str, measurement: str) -> str:
    return f"measurement:{key}:{measurement}"


def _condition_node_id(key: str, fact: str) -> str:
    return f"condition:{key}:{fact}"


def _anomaly_node_id(key: str) -> str:
    return f"anomaly:{key}"


def _rule_node_id(rule_id: str) -> str:
    return f"rule:{rule_id}"


def _fault_node_id(fault_id: str) -> str:
    return f"fault:{fault_id}"


def _slug(value: str) -> str:
    """Convert an action description into a stable ID-safe suffix."""
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return value.strip("-") or "inspection"


def _action_node_id(fault_id: str, action: str) -> str:
    """Allow different actions for one fault without duplicating shared actions."""
    return f"action:{fault_id}:{_slug(action)}"


# ---------------------------------------------------------------------
# Generic graph operations
# ---------------------------------------------------------------------

def _add_node(
    graph: nx.DiGraph,
    node_id: str,
    node_type: str,
    label: str,
    **attributes: Any,
) -> None:
    """Add or update a graph node using consistent metadata."""
    graph.add_node(
        node_id,
        ntype=node_type,
        label=label,
        layer=LAYER[node_type],
        on_path=False,
        **attributes,
    )


def _add_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    label: str,
) -> None:
    """Add a directed relationship without duplicating an existing edge."""
    if not graph.has_edge(source, target):
        graph.add_edge(
            source,
            target,
            label=label,
            on_path=False,
        )


def _initialise_graph(graph: nx.DiGraph) -> None:
    """Attach graph-level metadata."""
    graph.graph.update(
        {
            "name": "ORBITALIS Knowledge Graph",
            "description": "Satellite solar panel fault diagnosis and evidence graph",
            "layers": dict(LAYER),
        }
    )


# ---------------------------------------------------------------------
# Knowledge-base graph
# ---------------------------------------------------------------------

def knowledge_structure_graph(kb: KnowledgeBase) -> nx.DiGraph:
    """Build a static graph containing diagnosis rules, faults, and actions."""
    graph = nx.DiGraph()
    _initialise_graph(graph)

    # Faults and their default inspection actions.
    for fault in kb.fault_list:
        fault_id = fault["fault_id"]
        fault_node = _fault_node_id(fault_id)

        _add_node(
            graph,
            fault_node,
            "fault",
            fault["name"],
            fault_id=fault_id,
        )

        action = fault.get("suggested_action", "Inspect the affected solar panel")
        action_node = _action_node_id(fault_id, action)

        _add_node(
            graph,
            action_node,
            "action",
            action,
            fault_id=fault_id,
        )

        _add_edge(graph, fault_node, action_node, "recommends")

    # Diagnosis rules and their conclusions.
    for rule in kb.rule_list:
        if not rule.get("is_diagnosis"):
            continue

        rule_id = rule["rule_id"]
        fault_id = rule.get("fault_id")

        _add_node(
            graph,
            _rule_node_id(rule_id),
            "rule",
            f"{rule_id} — {rule['name']}",
            rule_id=rule_id,
        )

        if fault_id and fault_id in {
            fault["fault_id"] for fault in kb.fault_list
        }:
            _add_edge(
                graph,
                _rule_node_id(rule_id),
                _fault_node_id(fault_id),
                "concludes",
            )

    return graph


# ---------------------------------------------------------------------
# Evidence-node construction
# ---------------------------------------------------------------------

def _add_panel_and_measurements(
    graph: nx.DiGraph,
    panel_diagnosis: Any,
    key: str,
) -> tuple[str, dict[str, str]]:
    """Create a panel node and its voltage/current measurement nodes."""
    panel_id = _panel_node_id(key)

    _add_node(
        graph,
        panel_id,
        "solar_panel",
        (
            f"{panel_diagnosis.satellite} "
            f"{panel_diagnosis.panel_label} panel\n"
            f"{panel_diagnosis.session}"
        ),
        satellite=panel_diagnosis.satellite,
        session=panel_diagnosis.session,
        panel=panel_diagnosis.panel,
    )

    measurement_ids = {}

    for measurement, label in MEASUREMENTS.items():
        node_id = _measurement_node_id(key, measurement)

        _add_node(
            graph,
            node_id,
            "measurement",
            label,
            measurement=measurement,
        )

        _add_edge(graph, panel_id, node_id, "measures")
        measurement_ids[measurement] = node_id

    return panel_id, measurement_ids


def _condition_count(facts: dict[str, Any], definition: dict) -> Any:
    """Safely retrieve the count associated with a condition."""
    return facts.get(definition["count_field"])


def _condition_is_present(
    fact: str,
    facts: dict[str, Any],
) -> bool:
    """Determine whether an observed condition should appear in the graph."""
    if facts.get(fact):
        return True

    # An unverifiable Type 2 candidate is shown only when no confirmed
    # Type 2 condition is present.
    if fact == "type2_unverifiable":
        return bool(
            facts.get("type2_unverifiable_count")
            and not facts.get("type2_condition_present")
        )

    return False


def _add_condition_nodes(
    graph: nx.DiGraph,
    key: str,
    facts: dict[str, Any],
    measurement_ids: dict[str, str],
) -> dict[str, str]:
    """Add observed-condition nodes and connect their supporting sensors."""
    condition_ids = {}

    for fact, definition in CONDITION_DEFINITIONS.items():
        if not _condition_is_present(fact, facts):
            continue

        count = _condition_count(facts, definition)
        count_label = definition["count_label"]

        if count is None:
            label = definition["label"]
        else:
            label = f"{definition['label']}\n({count_label} = {count})"

        node_id = _condition_node_id(key, fact)

        _add_node(
            graph,
            node_id,
            "condition",
            label,
            fact=fact,
            count=count,
        )

        for measurement in definition["measurements"]:
            measurement_id = measurement_ids[measurement]
            _add_edge(graph, measurement_id, node_id, "shows")

        condition_ids[fact] = node_id

    return condition_ids


def _add_ml_anomaly_node(
    graph: nx.DiGraph,
    key: str,
    facts: dict[str, Any],
    measurement_ids: dict[str, str],
) -> str | None:
    """Add the ML anomaly node when the detector reports anomalies."""
    anomaly_count = facts.get("ml_anomaly_count", 0)

    if not anomaly_count:
        return None

    node_id = _anomaly_node_id(key)

    _add_node(
        graph,
        node_id,
        "anomaly",
        f"Isolation Forest anomaly\n(n = {anomaly_count})",
        count=anomaly_count,
        detector="Isolation Forest",
    )

    for measurement_id in measurement_ids.values():
        _add_edge(graph, measurement_id, node_id, "flagged")

    return node_id


# ---------------------------------------------------------------------
# Rule, fault, and action construction
# ---------------------------------------------------------------------

def _add_diagnosis_candidate(
    graph: nx.DiGraph,
    candidate: Any,
    panel_id: str,
    condition_ids: dict[str, str],
    anomaly_id: str | None,
    kb: KnowledgeBase,
) -> str | None:
    """Add one fired diagnosis rule and its conclusion."""
    rule_id = candidate.rule_id

    # R050 represents normal operation, which has no fault evidence path.
    if rule_id == "R050":
        return None

    rule = candidate.evaluation.rule
    rule_node = _rule_node_id(rule_id)

    _add_node(
        graph,
        rule_node,
        "rule",
        f"{rule_id} — {rule['name']}",
        rule_id=rule_id,
    )

    linked = False

    # Connect conditions that satisfy the diagnostic rule.
    for condition in rule.get("conditions", []):
        fact = condition.get("fact")
        operator = condition.get("operator")

        if operator != "is_true":
            continue

        source_node = condition_ids.get(fact)

        if source_node:
            _add_edge(graph, source_node, rule_node, "satisfies")
            linked = True

    # ML evidence can strengthen a rule without independently proving a fault.
    if candidate.strengthened_by and anomaly_id:
        _add_edge(graph, anomaly_id, rule_node, "corroborates")
        linked = True

    # Keep evaluated rules visible even when no evidence node is available.
    if not linked:
        _add_edge(graph, panel_id, rule_node, "evaluated for")

    # Add the fault and its suggested action, if the rule has a known fault.
    fault_id = candidate.fault_id

    if not fault_id:
        return rule_node

    fault = kb.fault_by_id(fault_id)

    if not fault:
        return rule_node

    fault_node = _fault_node_id(fault_id)

    _add_node(
        graph,
        fault_node,
        "fault",
        fault["name"],
        fault_id=fault_id,
    )

    _add_edge(graph, rule_node, fault_node, "concludes")

    action = rule.get("suggested_action") or fault.get(
        "suggested_action",
        "Inspect the affected solar panel",
    )

    action_node = _action_node_id(fault_id, action)

    _add_node(
        graph,
        action_node,
        "action",
        action,
        fault_id=fault_id,
    )

    _add_edge(graph, fault_node, action_node, "recommends")

    return rule_node


# ---------------------------------------------------------------------
# Evidence-path highlighting
# ---------------------------------------------------------------------

def _highlight_primary_path(
    graph: nx.DiGraph,
    panel_id: str,
    key: str,
    primary_rule_id: str,
) -> None:
    """Highlight the selected panel's evidence path to its primary rule."""
    rule_id = _rule_node_id(primary_rule_id)

    if panel_id not in graph or rule_id not in graph:
        return

    # Only include evidence belonging to this satellite/session/panel.
    local_prefixes = (
        f"measurement:{key}:",
        f"condition:{key}:",
        f"anomaly:{key}",
    )

    path_nodes = {panel_id, rule_id}

    # Walk backwards from the primary rule through its evidence.
    pending = [rule_id]
    visited = set()

    while pending:
        current = pending.pop()

        if current in visited:
            continue

        visited.add(current)

        for predecessor in graph.predecessors(current):
            is_local_evidence = predecessor == panel_id or predecessor.startswith(
                local_prefixes
            )

            # A shared diagnostic rule is not treated as panel-specific evidence.
            if is_local_evidence:
                path_nodes.add(predecessor)
                pending.append(predecessor)

    # Include the primary rule's conclusion and suggested action.
    for descendant in nx.descendants(graph, rule_id):
        path_nodes.add(descendant)

    for node_id in path_nodes:
        graph.nodes[node_id]["on_path"] = True

    for source, target in graph.edges:
        if source in path_nodes and target in path_nodes:
            graph.edges[source, target]["on_path"] = True


# ---------------------------------------------------------------------
# Public dynamic graph builder
# ---------------------------------------------------------------------

def build_knowledge_graph(
    kb: KnowledgeBase,
    panel_diagnoses: Iterable,
    satellite: str | None = None,
    panel: str | None = None,
    include_normal: bool = False,
    highlight: tuple[str, str, str] | None = None,
) -> nx.DiGraph:
    """Build an evidence-driven graph from actual panel diagnoses.

    Args:
        kb:
            Loaded ORBITALIS knowledge base.
        panel_diagnoses:
            Iterable of pipeline.PanelDiagnosis objects.
        satellite:
            Optional satellite filter.
        panel:
            Optional panel-orientation filter.
        include_normal:
            Include normal diagnoses when True.
        highlight:
            Optional (satellite, session, panel) tuple identifying
            the primary diagnosis path to highlight.

    Returns:
        A directed NetworkX graph containing panel telemetry,
        observed conditions, ML anomalies, diagnostic rules,
        possible faults, and inspection actions.
    """
    graph = nx.DiGraph()
    _initialise_graph(graph)

    for panel_diagnosis in panel_diagnoses:

        # Apply requested filters.
        if satellite and panel_diagnosis.satellite != satellite:
            continue

        if panel and panel_diagnosis.panel != panel:
            continue

        diagnosis = panel_diagnosis.diagnosis

        if diagnosis.status == "normal" and not include_normal:
            continue

        key = _panel_key(
            panel_diagnosis.satellite,
            panel_diagnosis.session,
            panel_diagnosis.panel,
        )

        panel_id, measurement_ids = _add_panel_and_measurements(
            graph,
            panel_diagnosis,
            key,
        )

        condition_ids = _add_condition_nodes(
            graph,
            key,
            diagnosis.facts,
            measurement_ids,
        )

        anomaly_id = _add_ml_anomaly_node(
            graph,
            key,
            diagnosis.facts,
            measurement_ids,
        )

        # Add all fired diagnosis candidates.
        for candidate in diagnosis.candidates:
            _add_diagnosis_candidate(
                graph=graph,
                candidate=candidate,
                panel_id=panel_id,
                condition_ids=condition_ids,
                anomaly_id=anomaly_id,
                kb=kb,
            )

        # Highlight only the requested panel's primary diagnosis path.
        if (
            highlight == (
                panel_diagnosis.satellite,
                panel_diagnosis.session,
                panel_diagnosis.panel,
            )
            and diagnosis.primary
        ):
            _highlight_primary_path(
                graph=graph,
                panel_id=panel_id,
                key=key,
                primary_rule_id=diagnosis.primary.rule_id,
            )

    return graph
