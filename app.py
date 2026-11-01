"""ORBITALIS - Streamlit interface (presentation only; logic lives in src/)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import config, database, pipeline, visualization as viz
from src.knowledge_base import KnowledgeBaseError, load_knowledge_base
from src.knowledge_graph import NODE_TYPES, build_knowledge_graph, knowledge_structure_graph
from src.ml_detector import MLDetector, ModelNotFoundError
from src.preprocessing import DataValidationError

st.set_page_config(page_title="ORBITALIS | Solar Panel Fault Diagnosis", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stMetric"]{background:#111a2e;border:1px solid #22304d;border-radius:10px;padding:14px 16px}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
h1,h2,h3{letter-spacing:-0.01em}
</style>""", unsafe_allow_html=True)

STATUS_COLOR = {"normal": "green", "anomalous_behaviour": "orange", "possible_documented_fault": "red",
                "insufficient_evidence": "yellow", "conflicting_evidence": "violet"}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_kb():
    return load_knowledge_base()


@st.cache_resource(show_spinner=False)
def get_detector():
    return MLDetector.load()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return pipeline.load_dataset_wide()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def badge(status: str, label: str) -> str:
    return f":{STATUS_COLOR.get(status, 'gray')}-badge[{label}]"


# --------------------------------------------------------------------------- #
# State & sidebar controls
# --------------------------------------------------------------------------- #
def active_wide() -> pd.DataFrame | None:
    if st.session_state.get("source") == "Uploaded CSV":
        return st.session_state.get("upload_wide")
    return get_dataset()


def run_analysis(wide: pd.DataFrame, **filters) -> None:
    with st.status("Running hybrid analysis (ML detection, forward chaining, explanation)...", expanded=False) as status:
        try:
            result = pipeline.analyze(
                wide, source=("upload" if st.session_state.get("source") == "Uploaded CSV" else "dataset"),
                detector=get_detector(), kb=get_kb(), warnings=st.session_state.get("upload_warnings", []), **filters)
        except (DataValidationError, ModelNotFoundError, KnowledgeBaseError) as exc:
            status.update(label="Analysis failed", state="error")
            st.session_state["result"] = None
            st.error(str(exc))
            return
        st.session_state["result"] = result
        st.session_state["saved_ids"] = None
        status.update(label=f"Analysis complete: {len(result.panel_diagnoses)} panel windows", state="complete")


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🛰️ ORBITALIS")
        st.caption("Hybrid Knowledge-Based System for Satellite Solar Panel Fault Diagnosis")
        st.radio("Data source", ["Project dataset", "Uploaded CSV"], key="source")
        wide = active_wide()
        if wide is None:
            st.info("Upload a CSV on the Telemetry Analysis page first.")
            return
        sats = sorted(wide[config.SATELLITE_COL].unique())
        sat_sel = st.multiselect("Satellite", sats, default=sats, key="f_sat")
        subset = wide[wide[config.SATELLITE_COL].isin(sat_sel)] if sat_sel else wide
        sess = list(dict.fromkeys(subset[config.SESSION_COL]))
        sess_sel = st.multiselect("Session (one orbit each)", sess, default=sess, key="f_sess")
        panels_available = [p for p in config.PANELS
                            if config.voltage_col(p) in wide.columns and config.current_col(p) in wide.columns]
        panel_sel = st.multiselect("Solar panel", panels_available, default=panels_available,
                                   format_func=lambda p: f"{config.PANELS[p]} panel", key="f_panel")
        lo, hi = float(wide[config.TIME_COL].min()), float(wide[config.TIME_COL].max())
        t_range = st.slider("Time range (s elapsed)", lo, hi, (lo, hi), key="f_time")
        if st.button("Run analysis", type="primary", width="stretch"):
            run_analysis(wide, satellites=sat_sel or None, sessions=sess_sel or None,
                         panels=panel_sel or None, time_range=t_range)
        elif st.session_state.get("result") is None:
            run_analysis(wide)
        res = st.session_state.get("result")
        st.divider()
        st.caption("Last analysis: " + (res.created_utc.replace("T", " ") + " UTC" if res else "none"))
        st.caption("Source: " + st.session_state.get("source", "Project dataset"))


def need_result():
    res = st.session_state.get("result")
    if res is None:
        st.info("No analysis available yet. Choose a data source and press **Run analysis** in the sidebar.")
        st.stop()
    return res


def window_picker(res, key: str):
    opts = [p.key for p in res.panel_diagnoses]
    return res.get(st.selectbox(
        "Panel window", opts, key=key,
        format_func=lambda k: f"{k[0]} · {k[1]} · {config.PANELS.get(k[2], k[2])} panel"))


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.title("Overview")
    st.caption("Anomaly detection (Isolation Forest) combined with research-based diagnostic rules for "
               "BIRDS CubeSat solar panels. Anomaly detection is not fault diagnosis: documented conditions are "
               "reported as *possible* faults only.")
    res = need_result()
    for w in res.warnings:
        st.warning(w)
    k = res.kpis()
    c = st.columns(5)
    c[0].metric("Observations analysed", f"{k['total_observations']:,}")
    c[1].metric("Normal (ML)", f"{k['normal_observations']:,}")
    c[2].metric("Anomalous (ML)", f"{k['anomalous_observations']:,}")
    c[3].metric("Possible documented faults", k["possible_faults"], help="Panel windows with a documented condition")
    c[4].metric("Panel windows", k["panels_analysed"], help=", ".join(k["satellites"]))
    st.subheader("Panel diagnoses")
    st.dataframe(res.diagnosis_table(), width="stretch", hide_index=True)
    st.subheader("Telemetry")
    pd_ = window_picker(res, "ov_window")
    obs = res.observations
    mask = ((obs[config.SATELLITE_COL] == pd_.satellite) & (obs[config.SESSION_COL] == pd_.session)
            & (obs["Panel"] == pd_.panel))
    st.plotly_chart(viz.telemetry_figure(obs[mask], f"{pd_.satellite} · {pd_.session} · {pd_.panel_label} panel"),
                    width="stretch")
    st.caption("Orange dots: Isolation Forest anomalies. Red crosses: samples meeting a documented condition "
               "(SRC-1 Eq.1 / Eq.2). Temperature is shown for context only.")


def page_telemetry() -> None:
    st.title("Telemetry Analysis")
    tab_up, tab_pred, tab_q = st.tabs(["Data source", "Predictions", "Data quality"])
    with tab_up:
        st.write("Upload a CSV with a `Time Stamp` column (elapsed seconds) and per-panel columns such as "
                 "`Vpy (mV)`, `Ipy (mA)`, `Tpy (°C)`. Optional: `Satellite_ID`, `Source_Sheet`. Battery columns are ignored.")
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                raw = pd.read_csv(up)
                clean, warns = pipeline.prepare_upload(raw)
            except (DataValidationError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                st.error(f"Upload rejected: {exc}")
            else:
                st.success(f"File accepted: {len(clean):,} rows.")
                for w in warns:
                    st.warning(w)
                st.dataframe(raw.head(50), width="stretch")
                if st.button("Use this file as data source", type="primary"):
                    st.session_state.update(upload_wide=clean, upload_warnings=warns, source="Uploaded CSV", result=None)
                    st.rerun()
        st.caption("Current source: " + st.session_state.get("source", "Project dataset"))
    with tab_pred:
        res = need_result()
        obs = res.observations
        lo, hi = float(obs["Elapsed_s"].min()), float(obs["Elapsed_s"].max())
        c1, c2 = st.columns([2, 1])
        rng = c1.slider("Filter by elapsed time (s)", lo, hi, (lo, hi), key="pred_time")
        status = c2.multiselect("ML status", [config.LABEL_ANOMALY, config.LABEL_NORMAL, config.LABEL_NOT_SCORED],
                                default=[config.LABEL_ANOMALY, config.LABEL_NORMAL])
        view = obs[obs["Elapsed_s"].between(*rng) & obs["ML_Status"].isin(status)]
        cols = [config.SATELLITE_COL, config.SESSION_COL, "Panel_Label", "Elapsed_s", "Voltage_mV", "Current_mA",
                "Temperature_C", "ML_Status", "ML_Score", "Documented_Type"]
        st.dataframe(view[cols], width="stretch", hide_index=True)
        st.caption(f"{len(view):,} of {len(obs):,} observations. ML score: negative = more anomalous (not a probability).")
        st.download_button("Download results (CSV)", view[cols].to_csv(index=False).encode("utf-8"),
                           "orbitalis_results.csv", "text/csv")
    with tab_q:
        rep = read_json(config.QUALITY_REPORT_JSON)
        if rep is None:
            st.info("Run `python -m src.preprocessing` to generate the data-quality report.")
        else:
            a, b, c = st.columns(3)
            a.metric("Rows read", rep["rows_input"]); b.metric("Rows kept", rep["rows_output"])
            c.metric("Rows quarantined (no time stamp)", rep["rows_quarantined_missing_timestamp"])
            st.dataframe(pd.DataFrame(rep["sampling_check"]), width="stretch", hide_index=True)
            st.caption(rep["note"])


def page_diagnosis() -> None:
    st.title("Fault Diagnosis")
    res = need_result()
    pd_ = window_picker(res, "dg_window")
    d, ex = pd_.diagnosis, pd_.explanation
    st.markdown(f"### {badge(d.status, d.status_label)} {d.fault_name if d.status != 'normal' else ''}")
    st.write(ex.summary)
    c = st.columns(4)
    c[0].metric("Severity", d.severity.capitalize())
    c[1].metric("Evidence strength", d.evidence_strength, help="Qualitative corroboration level, not a calibrated probability.")
    c[2].metric("ML anomalous samples", f"{pd_.facts.get('ml_anomaly_count', 0)} / {pd_.n_samples}")
    c[3].metric("Matched rules", len(d.matched_rule_ids))
    t1, t2, t3, t4 = st.tabs(["Evidence", "Matched rules", "Reasoning", "Alternatives & caveats"])
    with t1:
        st.dataframe(pd.DataFrame(pd_.evidence).rename(columns={
            "feature_name": "Feature", "observed_value": "Observed", "evidence_description": "Description"}),
            width="stretch", hide_index=True)
        st.info(ex.ml_result)
    with t2:
        for r in ex.matched_rules:
            with st.expander(f"{r['rule_id']} · {r['name']}", expanded=r["is_diagnosis"]):
                st.caption(f"Source: {r['source']}")
                for cond in r["conditions"]:
                    st.write("✔ " + cond)
    with t3:
        for line in ex.reasoning:
            st.write("- " + line)
        st.success("**Suggested inspection:** " + ex.suggested_action)
    with t4:
        st.write("**Why other diagnoses were not selected**")
        for a in ex.alternatives or ["(none)"]:
            st.write("- " + a)
        st.write("**Caveats**")
        for cv in ex.caveats:
            st.write("- " + cv)
    if st.button("Save all panel diagnoses of this analysis to history"):
        if st.session_state.get("saved_ids"):
            st.warning("This analysis was already saved.")
        else:
            ids = database.save_analysis(res)
            st.session_state["saved_ids"] = ids
            st.success(f"Saved {len(ids)} diagnoses (IDs {ids[0]}–{ids[-1]}).")


def page_graph() -> None:
    st.title("Knowledge Graph")
    res = need_result()
    c1, c2 = st.columns(2)
    sats = ["All"] + sorted({p.satellite for p in res.panel_diagnoses})
    sat = c1.selectbox("Satellite", sats)
    pans = ["All"] + sorted({p.panel for p in res.panel_diagnoses}, key=list(config.PANELS).index)
    pan = c2.selectbox("Panel", pans, format_func=lambda p: p if p == "All" else config.PANELS[p])
    cand = [p for p in res.panel_diagnoses
            if (sat == "All" or p.satellite == sat) and (pan == "All" or p.panel == pan)]
    if not cand:
        st.info("No panel windows for this filter.")
        return
    focus = st.selectbox("Highlight evidence path of", [p.key for p in cand],
                         format_func=lambda k: f"{k[0]} · {k[1]} · {config.PANELS[k[2]]} panel")
    include_normal = st.toggle("Include normal panels", value=False)
    G = build_knowledge_graph(res.kb, cand, include_normal=include_normal or res.get(focus).diagnosis.status == "normal",
                              highlight=focus)
    if G.number_of_nodes() == 0:
        st.info("All selected panels are normal; there is no diagnostic path to draw.")
    else:
        st.plotly_chart(viz.graph_figure(G), width="stretch")
        st.caption(f"{G.number_of_nodes()} nodes · {G.number_of_edges()} relationships. Amber path = evidence behind the "
                   "selected diagnosis. Drag to pan, scroll to zoom. Legend: " + ", ".join(NODE_TYPES.values()) + ".")
    with st.expander("Static knowledge-base structure (rules → faults → actions)"):
        st.plotly_chart(viz.graph_figure(knowledge_structure_graph(res.kb)), width="stretch")


def page_performance() -> None:
    st.title("Model Performance")
    m, meta = read_json(config.METRICS_PATH), read_json(config.METADATA_PATH)
    if m is None:
        st.info("No evaluation available. Run `python -m src.train_model`.")
        return
    st.caption("Isolation Forest evaluated on held-out sessions "
               + ", ".join(f"{s['satellite']} {s['session']}" for s in m["evaluation_sessions"])
               + ". Labels are derived from the published Eq.(1)/(2) definitions, not independent ground truth.")
    c = st.columns(5)
    c[0].metric("Precision", f"{m['precision']:.3f}"); c[1].metric("Recall", f"{m['recall']:.3f}")
    c[2].metric("F1-score", f"{m['f1']:.3f}"); c[3].metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
    c[4].metric("Avg. precision", f"{m.get('average_precision', float('nan')):.3f}")
    a, b = st.columns(2)
    with a:
        st.subheader("Confusion matrix")
        st.plotly_chart(viz.confusion_figure(m["confusion_matrix"]["matrix"], m["confusion_matrix"]["labels"]), width="stretch")
    with b:
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(m["classification_report"]).T.round(3), width="stretch")
        st.write(f"Recall Type 1: **{m['recall_type_1']:.3f}** (n={m['n_type_1']}) · "
                 f"Recall Type 2: **{m['recall_type_2']:.3f}** (n={m['n_type_2']})")
    st.subheader("Anomaly distribution (held-out sessions)")
    dist = pd.DataFrame(m["anomaly_distribution_test"])
    dist["Window"] = dist["Satellite_ID"] + " · " + dist["Panel_Label"]
    st.plotly_chart(viz.distribution_figure(dist, "Window", "ML_Status"), width="stretch")
    with st.expander("Validation used to choose the threshold (training sessions only)"):
        st.dataframe(pd.DataFrame(meta["contamination_validation"]), width="stretch", hide_index=True)
        st.caption(meta["feature_selection"])
    st.subheader("Limitations")
    for lim in m["limitations"]:
        st.write("- " + lim)


def page_history() -> None:
    st.title("Diagnosis History")
    c1, c2 = st.columns(2)
    sat = c1.text_input("Filter by satellite", "").strip().upper() or None
    status = c2.selectbox("Filter by status", ["All"] + list(get_kb().faults["diagnosis_statuses"]))
    df = database.list_diagnoses(satellite=sat, status=None if status == "All" else status)
    if df.empty:
        st.info("No saved diagnoses yet. Save an analysis from the Fault Diagnosis page.")
        return
    st.dataframe(df, width="stretch", hide_index=True)
    sel = st.selectbox("Open diagnosis", df["diagnosis_id"].tolist())
    rec = database.get_diagnosis(int(sel))
    st.markdown(f"### {badge(rec['diagnosis_status'], rec['diagnosis_status'].replace('_', ' ').title())} "
                f"{rec['satellite_id']} · {rec['session']} · {rec['panel_id']} panel")
    st.caption(f"Saved {rec['timestamp']} · matched rules: {rec['matched_rules'] or '-'}")
    t1, t2, t3 = st.tabs(["Explanation", "Evidence", "Facts"])
    t1.markdown(rec["explanation"]["reasoning_text"] if rec["explanation"] else "(no explanation stored)")
    t2.dataframe(pd.DataFrame(rec["evidence"]), width="stretch", hide_index=True)
    t3.json(rec["facts"])


def page_system() -> None:
    st.title("System Information")
    kb = get_kb()
    st.markdown("**Architecture:** CSV/dataset → preprocessing → Isolation Forest → facts → forward-chaining "
                "inference (JSON rules) → conflict resolution → explanation → knowledge graph → SQLite → Streamlit.")
    st.subheader("Knowledge base rules")
    st.dataframe(pd.DataFrame([{
        "Rule": r["rule_id"], "Name": r["name"], "Layer": r["layer"], "Priority": r["priority"],
        "Severity": r["severity"], "Concludes": r["conclusion"]["fact"],
        "Source": r["source"]["source_id"] + (" " + r["source"].get("equation", "") if r["source"].get("equation") else "")}
        for r in kb.rule_list]), width="stretch", hide_index=True)
    st.subheader("Sources")
    for s in kb.sources:
        st.write(f"- **{s['source_id']}** {s['citation']} {s.get('url', '')}")
    meta = read_json(config.METADATA_PATH)
    if meta:
        st.subheader("Model")
        st.json({k: meta[k] for k in ("algorithm", "n_estimators", "contamination", "features", "created_utc",
                                      "training_sessions", "test_sessions")})
    st.subheader("Scope and limitations")
    st.write("- Solar-panel telemetry only; battery columns are excluded.\n"
             "- Documented conditions are research-specific (SRC-1) and are reported as *possible* faults.\n"
             "- The system has not been validated for operational satellite deployment.")


sidebar()
nav = st.navigation([
    st.Page(page_overview, title="Overview", icon="🛰️", default=True),
    st.Page(page_telemetry, title="Telemetry Analysis", icon="📈", url_path="telemetry"),
    st.Page(page_diagnosis, title="Fault Diagnosis", icon="🩺", url_path="diagnosis"),
    st.Page(page_graph, title="Knowledge Graph", icon="🕸️", url_path="graph"),
    st.Page(page_performance, title="Model Performance", icon="📊", url_path="performance"),
    st.Page(page_history, title="Diagnosis History", icon="🗂️", url_path="history"),
    st.Page(page_system, title="System Information", icon="⚙️", url_path="system"),
])
nav.run()
