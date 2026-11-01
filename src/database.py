"""SQLite persistence of diagnosis history.

Tables: ``diagnoses`` -> ``diagnosis_evidence`` / ``diagnosis_explanations``
(foreign keys enforced, parameterised queries only). Raw telemetry is not
copied into the database.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    satellite_id        TEXT NOT NULL,
    session             TEXT,
    panel_id            TEXT NOT NULL,
    diagnosis_status    TEXT NOT NULL,
    conclusion          TEXT,
    severity            TEXT,
    evidence_strength   TEXT,
    anomaly_count       INTEGER NOT NULL DEFAULT 0,
    diagnostic_summary  TEXT,
    matched_rules       TEXT,
    source              TEXT,
    facts_json          TEXT
);
CREATE TABLE IF NOT EXISTS diagnosis_evidence (
    evidence_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id         INTEGER NOT NULL REFERENCES diagnoses(diagnosis_id) ON DELETE CASCADE,
    feature_name         TEXT NOT NULL,
    observed_value       TEXT,
    evidence_description TEXT
);
CREATE TABLE IF NOT EXISTS diagnosis_explanations (
    explanation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id     INTEGER NOT NULL REFERENCES diagnoses(diagnosis_id) ON DELETE CASCADE,
    reasoning_text   TEXT,
    suggested_action TEXT,
    explanation_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_diag ON diagnosis_evidence(diagnosis_id);
CREATE INDEX IF NOT EXISTS idx_expl_diag ON diagnosis_explanations(diagnosis_id);
"""


@contextmanager
def connect(db_path: Path | str = config.DB_PATH) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str = config.DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def save_diagnosis(panel_diagnosis: Any, source: str = "dataset", db_path: Path | str = config.DB_PATH) -> int:
    """Persist one ``pipeline.PanelDiagnosis``; returns the new diagnosis_id."""
    init_db(db_path)
    d = panel_diagnosis.diagnosis
    expl = panel_diagnosis.explanation
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO diagnoses (timestamp, satellite_id, session, panel_id, diagnosis_status,
               conclusion, severity, evidence_strength, anomaly_count, diagnostic_summary,
               matched_rules, source, facts_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), panel_diagnosis.satellite,
             panel_diagnosis.session, panel_diagnosis.panel_label, d.status, d.conclusion, d.severity,
             d.evidence_strength, int(d.facts.get("ml_anomaly_count") or 0), expl.summary,
             ",".join(d.matched_rule_ids), source, json.dumps(d.facts, default=str)),
        )
        diagnosis_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO diagnosis_evidence (diagnosis_id, feature_name, observed_value, evidence_description) VALUES (?,?,?,?)",
            [(diagnosis_id, e["feature_name"], e["observed_value"], e["evidence_description"])
             for e in panel_diagnosis.evidence],
        )
        conn.execute(
            "INSERT INTO diagnosis_explanations (diagnosis_id, reasoning_text, suggested_action, explanation_json) VALUES (?,?,?,?)",
            (diagnosis_id, expl.to_markdown(), expl.suggested_action, json.dumps(expl.to_dict(), default=str)),
        )
    return diagnosis_id


def save_analysis(result: Any, db_path: Path | str = config.DB_PATH) -> list[int]:
    """Persist every panel diagnosis of an analysis run."""
    return [save_diagnosis(pd_, result.source, db_path) for pd_ in result.panel_diagnoses]


def list_diagnoses(db_path: Path | str = config.DB_PATH, satellite: str | None = None,
                   status: str | None = None, limit: int = 1000) -> pd.DataFrame:
    init_db(db_path)
    sql = ("SELECT diagnosis_id, timestamp, satellite_id, session, panel_id, diagnosis_status, "
           "severity, evidence_strength, anomaly_count, matched_rules, source FROM diagnoses WHERE 1=1")
    params: list[Any] = []
    if satellite:
        sql += " AND satellite_id = ?"; params.append(satellite)
    if status:
        sql += " AND diagnosis_status = ?"; params.append(status)
    sql += " ORDER BY diagnosis_id DESC LIMIT ?"; params.append(int(limit))
    with connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_diagnosis(diagnosis_id: int, db_path: Path | str = config.DB_PATH) -> dict | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM diagnoses WHERE diagnosis_id = ?", (int(diagnosis_id),)).fetchone()
        if row is None:
            return None
        evidence = [dict(r) for r in conn.execute(
            "SELECT feature_name, observed_value, evidence_description FROM diagnosis_evidence WHERE diagnosis_id = ? ORDER BY evidence_id",
            (int(diagnosis_id),))]
        expl = conn.execute(
            "SELECT reasoning_text, suggested_action, explanation_json FROM diagnosis_explanations WHERE diagnosis_id = ?",
            (int(diagnosis_id),)).fetchone()
    out = dict(row)
    out["evidence"] = evidence
    out["explanation"] = dict(expl) if expl else None
    out["facts"] = json.loads(out.pop("facts_json") or "{}")
    return out


def count_diagnoses(db_path: Path | str = config.DB_PATH) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0])
