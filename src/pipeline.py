"""End-to-end orchestration used by the Streamlit app and the tests.

CSV / dataset (wide) -> validation & cleaning -> long panel observations ->
documented-condition labelling -> ML anomaly detection -> facts -> forward
chaining -> conflict resolution -> evidence -> explanation (-> graph / SQLite).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import config, preprocessing
from .conflict_resolution import Diagnosis, resolve
from .explanation import Explanation, collect_evidence, generate_explanation
from .inference_engine import ForwardChainingEngine, InferenceResult, build_panel_facts
from .knowledge_base import KnowledgeBase, load_knowledge_base
from .labeling import label_documented_conditions
from .ml_detector import MLDetector

WINDOW_KEYS = [config.SATELLITE_COL, config.SESSION_COL, "Panel"]


@dataclass
class PanelDiagnosis:
    satellite: str
    session: str
    panel: str
    panel_label: str
    facts: dict[str, Any]
    inference: InferenceResult
    diagnosis: Diagnosis
    evidence: list[dict]
    explanation: Explanation
    n_samples: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.satellite, self.session, self.panel)


@dataclass
class AnalysisResult:
    observations: pd.DataFrame
    panel_diagnoses: list[PanelDiagnosis]
    kb: KnowledgeBase
    source: str
    warnings: list[str] = field(default_factory=list)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def kpis(self) -> dict[str, Any]:
        obs = self.observations
        faults = [p for p in self.panel_diagnoses
                  if p.diagnosis.status in ("possible_documented_fault", "conflicting_evidence")]
        return {
            "total_observations": int(len(obs)),
            "normal_observations": int((obs["ML_Status"] == config.LABEL_NORMAL).sum()),
            "anomalous_observations": int((obs["ML_Status"] == config.LABEL_ANOMALY).sum()),
            "not_scored": int((obs["ML_Status"] == config.LABEL_NOT_SCORED).sum()),
            "possible_faults": len(faults),
            "panels_analysed": len(self.panel_diagnoses),
            "satellites": sorted(obs[config.SATELLITE_COL].unique().tolist()),
        }

    def diagnosis_table(self) -> pd.DataFrame:
        rows = []
        for p in self.panel_diagnoses:
            d = p.diagnosis
            rows.append({
                "Satellite": p.satellite, "Session": p.session, "Panel": p.panel_label,
                "Status": d.status_label, "Conclusion": d.conclusion.replace("_", " "),
                "Severity": d.severity, "Evidence strength": d.evidence_strength,
                "ML anomalies": p.facts.get("ml_anomaly_count"), "Samples": p.n_samples,
                "Matched rules": ", ".join(d.matched_rule_ids),
            })
        return pd.DataFrame(rows)

    def get(self, key: tuple[str, str, str]) -> PanelDiagnosis | None:
        return next((p for p in self.panel_diagnoses if p.key == key), None)


# --------------------------------------------------------------------------- #
def load_dataset_wide() -> pd.DataFrame:
    """Processed project dataset (runs preprocessing when the CSV is missing)."""
    if not config.CLEANED_CSV.exists():
        preprocessing.run()
    return pd.read_csv(config.CLEANED_CSV, parse_dates=[config.SESSION_DATE_COL, config.DATETIME_COL])


def prepare_upload(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean an uploaded CSV table (raises DataValidationError)."""
    valid, warnings = preprocessing.validate_wide(raw)
    clean, report = preprocessing.clean_wide(valid)
    if report["rows_quarantined_missing_timestamp"]:
        warnings.append(f"{report['rows_quarantined_missing_timestamp']} rows without a usable time stamp were excluded.")
    if report["duplicate_observations_removed"]:
        warnings.append(f"{report['duplicate_observations_removed']} duplicate observations were removed.")
    irregular = sum(s["irregular_intervals"] for s in report["sampling_check"])
    if irregular:
        warnings.append(
            f"{irregular} sampling intervals differ from the expected {report['expected_interval_s']:.0f} s; "
            "consecutive-sample conditions near them are reported as unverifiable.")
    if report["invalid_numeric_values_coerced_to_nan"]:
        warnings.append("Non-numeric sensor values were treated as missing: "
                        + ", ".join(report["invalid_numeric_values_coerced_to_nan"]))
    return clean, warnings


def analyze(
    wide: pd.DataFrame,
    source: str = "dataset",
    satellites: list[str] | None = None,
    sessions: list[str] | None = None,
    panels: list[str] | None = None,
    time_range: tuple[float, float] | None = None,
    detector: MLDetector | None = None,
    kb: KnowledgeBase | None = None,
    warnings: list[str] | None = None,
) -> AnalysisResult:
    """Run the complete hybrid analysis on a cleaned wide table."""
    kb = kb or load_knowledge_base()
    detector = detector or MLDetector.load()
    data = wide
    if satellites:
        data = data[data[config.SATELLITE_COL].isin(satellites)]
    if sessions:
        data = data[data[config.SESSION_COL].isin(sessions)]
    if time_range is not None:
        data = data[data[config.TIME_COL].between(time_range[0], time_range[1])]
    if data.empty:
        raise preprocessing.DataValidationError("No observations remain after applying the selected filters.")

    interval = kb.detection_parameters["type2_solar_cell_failure"]["expected_sampling_interval_s"]
    long = preprocessing.wide_to_long(data, expected_interval_s=float(interval))
    if panels:
        long = long[long["Panel"].isin(panels)]
    labelled = label_documented_conditions(long, kb.detection_parameters)
    scored = detector.detect(labelled)

    engine = ForwardChainingEngine.from_knowledge_base(kb)
    diagnoses: list[PanelDiagnosis] = []
    for (sat, sess, panel), window in scored.groupby(WINDOW_KEYS, sort=False):
        facts = build_panel_facts(window, detector.expected_anomaly_rate)
        inference = engine.infer(facts)
        diagnosis = resolve(inference, kb)
        label = window["Panel_Label"].iloc[0]
        evidence = collect_evidence(window, diagnosis)
        explanation = generate_explanation(
            diagnosis, inference, kb, window,
            {"satellite": sat, "session": sess, "panel_label": label}, evidence)
        diagnoses.append(PanelDiagnosis(sat, sess, panel, label, facts, inference, diagnosis,
                                        evidence, explanation, int(len(window))))
    return AnalysisResult(scored, diagnoses, kb, source, list(warnings or []))
