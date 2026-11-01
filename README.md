# ORBITALIS – Satellite Solar Panel Fault Diagnosis
**A Hybrid Knowledge-Based System for Satellite Solar Panel Fault Diagnosis**

ORBITALIS analyses real on-orbit solar-panel telemetry of BIRDS CubeSats (RAAVANA, UGUISU), detects unusual
behaviour with an Isolation Forest, and diagnoses *possible documented faults* with a forward-chaining
rule engine whose rules come from published research. Every conclusion is explained from the actual evidence.

## Live Application

Access the deployed application:
[Launch ORBITALIS Application](https://orbitalis-solar-diagnosis-kbs.streamlit.app/diagnosis)
The application provides interactive satellite telemetry analysis, anomaly detection, explainable fault diagnosis, knowledge graph visualization, and diagnosis history.

## 1. Problem statement
Reduced power generation of a CubeSat solar panel is hard to notice from raw telemetry; two BIRDS-3 satellites lost
a panel's output for weeks before it was detected (SRC-1, Introduction). Pure anomaly detectors flag *unusual* data but do
not say *why*; pure rules ignore unforeseen behaviour. ORBITALIS combines both.

## 2. Objectives
Detect anomalies (ML) · diagnose documented failures (rules) · resolve conflicting evidence · explain every result ·
visualise telemetry and the diagnostic knowledge graph · keep a diagnosis history.

## 3. Scope and limitations
Included: solar-panel voltage, current, temperature; time-series handling; ML + KBS diagnosis; Streamlit UI; SQLite history.
Excluded by design: battery data (columns are dropped), attitude/communication/power-distribution subsystems, deep learning,
automated repair. **ORBITALIS has not been validated for operational satellite use.** Documented conditions are
research-specific (SRC-1) and are reported as *possible* faults. The dataset has no illumination or attitude signal.

## 4. Dataset
BIRDS EPS dataset (https://github.com/BIRDSOpenSource/EPS_dataset): `dataset/raw/RAAVANA.xlsx` (2 sessions) and
`UGUISU.xlsx` (3 sessions); one worksheet = one orbit sampled every 5 s; 5 panels (+Y, +X, -Z, -X, +Z) x voltage (mV),
current (mA), temperature (°C). No label column exists. The raw files are never modified. Facts found while inspecting
(`python -m src.inspect_data`): 10 rows of UGUISU "10 April 2021" have no time stamp and are quarantined; no duplicates remain.
Other satellites (NEPALISAT, TSURU) can be added by placing their workbooks in `dataset/raw/` (TSURU is sampled every 10 s,
which the 5 s consecutive-sample logic would report as *insufficient evidence* until the interval parameter in `faults.json` is adapted).

## 5. Architecture
```
CSV / workbook -> preprocessing -> long panel observations -> documented-condition labels
   -> Isolation Forest (ML) -> facts -> forward chaining (rules.json) -> conflict resolution
   -> explanation + evidence -> knowledge graph -> SQLite -> Streamlit
```
`src/pipeline.py` orchestrates the modules; `app.py` only renders.

## 6. Technology
Python, pandas, NumPy, scikit-learn (Isolation Forest), joblib, openpyxl, NetworkX, Plotly, Streamlit, SQLite, pytest.

## 7. Machine learning
* Unit of analysis: one panel observation; features `log(1+V)` and `log(1+I)` (log scale separates a failed panel, a few mV,
  from an eclipsed one, about 1300 mV). Temperature is not a feature: it lowered leave-one-session-out F1 and SRC-1 §2.4 excludes it too.
* Fitted on observations without a documented condition only. Chronological, satellite-aware split: the latest session of each
  satellite (RAAVANA 11 March 2021, UGUISU 10 April 2021) is held out; the decision threshold (contamination) is chosen by
  leave-one-session-out validation on training sessions.
* `-1 -> Anomaly`, `1 -> Normal`; observations lacking V or I are "Not scored".
* **Labels are rule-derived** (SRC-1 Eq.1/Eq.2), not operator ground truth.

Held-out result (10,905 observations, 1,694 documented anomalies; reproduce with `python -m src.train_model`):
precision 0.534, recall 0.999, F1 0.696, ROC-AUC 0.958; confusion matrix [[7737, 1474], [2, 1692]]. Many false alarms are expected
because the model deliberately flags 15 % of normal data. Known limitation: a failed panel (about 0 mV) with the eclipse idle
current (about 4.7 mA) is scored normal by the model and is caught only by rule R001 (see `tests/test_labeling_and_ml.py`).

## 8. Knowledge base (`knowledge_base/`)
* `components.json` - panels, sensors, measurements, sources, dataset metadata.
* `faults.json` - fault/status categories and the **detection parameters with their source references**.
* `rules.json` - 14 rules, each with ID, name, layer, conditions, logic, conclusion, priority, severity, description,
  evidence requirements, source, suggested action.

Research-based conditions (SRC-1: Jara Cespedes et al., *Applied Sciences* 12(17):8634, 2022, DOI 10.3390/app12178634), verified against the full text:
* Type 1, Eq.(1): `V < 1200 mV` (R001, R010).
* Type 2, Eq.(2): `2000 <= V(t-1), V(t), V(t+1) <= 3000 mV` and `I(t) < 50 mA`, contiguous 5 s samples (R002, R020).

Other rules combine evidence (ML agreement, minimum-pattern caution, insufficient evidence, normal) and are marked
`ORBITALIS design`; they contain no physical thresholds. Severity values are ORBITALIS classifications, not paper values.
Rules were **not** added to reach a target count. Only SRC-1 has been extracted so far; further papers can be added as new sources/rules.

## 9. Forward chaining and conflict resolution
`src/inference_engine.py` reads rules generically: facts are asserted, rules evaluated until a fixed point, layer by layer
(stratified, so negations are safe). Rules lacking required evidence are reported, not silently skipped.
`src/conflict_resolution.py` ranks matching diagnosis rules by priority, specificity, severity, evidence strength.
Statuses: Normal, Anomalous behaviour, Possible documented fault, Insufficient evidence, Conflicting evidence. Evidence strength
(Strong/Moderate/Weak) is a qualitative corroboration level, **not a probability**. A Type 2 pattern that never exceeds the
paper's minimum of three samples is reported as insufficient evidence (the paper notes it can occur at eclipse transitions).

## 10. Knowledge graph and explanation
`src/knowledge_graph.py` builds a NetworkX graph (panel -> measurement -> observed condition/ML anomaly -> rule -> fault -> action)
from the current diagnoses, filterable by satellite and panel, with the evidence path highlighted. `src/explanation.py` generates
evidence, matched rules and satisfied conditions, reasoning steps, alternatives not selected, inspection advice and caveats.

## 11. Database (`database/orbitalis.db`)
`diagnoses` -> `diagnosis_evidence`, `diagnosis_explanations` (foreign keys, parameterised queries). Raw telemetry is not stored.
The file is created automatically.

## 12. Installation and run
Prerequisite: Python 3.10+.
```
pip install -r requirements.txt
python -m src.inspect_data        # optional: dataset inventory
python -m src.preprocessing       # raw workbooks -> dataset/processed/cleaned_data.csv (+ quality report)
python -m src.train_model         # trains models/model.joblib, writes metrics/metadata
streamlit run app.py              # http://localhost:8501
python -m pytest                  # 75 tests
```
The processed data and trained model are already included, so `streamlit run app.py` works directly.

## 13. Application pages
Overview · Telemetry Analysis (CSV upload, validation, data quality, predictions, download) · Fault Diagnosis ·
Knowledge Graph · Model Performance · Diagnosis History · System Information.
Upload format: `Time Stamp` (elapsed seconds) plus columns such as `Vpy (mV)`, `Ipy (mA)`, `Tpy (°C)` for panels
py, px, mz, mx, pz; optional `Satellite_ID`, `Source_Sheet`.

## 14. References
1. Jara Cespedes, A.J.; Pangestu, B.H.B.; Hanazawa, A.; Cho, M. Performance Evaluation of Machine Learning Methods for
   Anomaly Detection in CubeSat Solar Panels. *Applied Sciences* 2022, 12(17), 8634. https://doi.org/10.3390/app12178634
2. BIRDS EPS Dataset, https://github.com/BIRDSOpenSource/EPS_dataset
