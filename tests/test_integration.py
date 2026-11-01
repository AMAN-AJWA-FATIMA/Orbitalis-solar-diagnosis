"""End-to-end flow on the real dataset and on an upload."""
import io

import pandas as pd
import pytest

from src import config, database, pipeline
from src.knowledge_graph import build_knowledge_graph
from src.preprocessing import DataValidationError
from src.visualization import graph_figure, telemetry_figure


@pytest.fixture(scope="module")
def result():
    return pipeline.analyze(pipeline.load_dataset_wide())


def test_all_five_stages_produce_output(result):
    assert result.kpis()["total_observations"] == len(result.observations) > 0
    assert len(result.panel_diagnoses) == 25
    assert {"ML_Status", "Type1_Condition", "Documented_Type"} <= set(result.observations.columns)


def test_known_failures_found_in_real_data(result):
    raavana_y = result.get(("RAAVANA", "13 Feb 2021", "py")).diagnosis
    assert raavana_y.conclusion == "possible_solar_panel_failure" and raavana_y.severity == "high"
    uguisu_y = result.get(("UGUISU", "13 Feb 2021", "py")).diagnosis
    assert uguisu_y.conclusion == "possible_solar_cell_failure"


def test_battery_not_in_pipeline(result):
    assert not any("bat" in c.lower() for c in result.observations.columns)


def test_explanation_is_dynamic(result):
    a = result.get(("RAAVANA", "13 Feb 2021", "py")).explanation.to_markdown()
    b = result.get(("UGUISU", "13 Feb 2021", "py")).explanation.to_markdown()
    assert "R010" in a and "R020" in b and a != b and "1079" in a


def test_graph_reflects_diagnosis_and_highlights_path(result):
    focus = ("RAAVANA", "13 Feb 2021", "py")
    G = build_knowledge_graph(result.kb, result.panel_diagnoses, satellite="RAAVANA", highlight=focus)
    assert "rule:R010" in G and "fault:F001" in G and G.nodes["rule:R010"]["on_path"]
    assert "rule:R020" not in {n for n in G if G.nodes[n]["on_path"]}
    assert len(set(G.nodes)) == G.number_of_nodes()
    assert graph_figure(G).data
    only_u = build_knowledge_graph(result.kb, result.panel_diagnoses, satellite="UGUISU")
    assert not any(n.startswith("panel:RAAVANA") for n in only_u)


def test_charts_build(result):
    obs = result.observations
    assert telemetry_figure(obs[obs.Panel == "py"].head(200)).data


def test_sqlite_roundtrip(result, tmp_path):
    db = tmp_path / "i.db"
    ids = database.save_analysis(result, db)
    assert len(ids) == 25
    assert database.get_diagnosis(ids[3], db)["explanation"]["reasoning_text"]


def test_upload_flow_and_errors():
    wide = pipeline.load_dataset_wide()
    sample = wide[(wide.Satellite_ID == "UGUISU") & (wide.Source_Sheet == "13 Feb 2021")].drop(
        columns=["Satellite_ID", "Source_Sheet", "Session_Date", "Datetime"])
    clean, warns = pipeline.prepare_upload(pd.read_csv(io.StringIO(sample.to_csv(index=False))))
    res = pipeline.analyze(clean, source="upload")
    assert res.get(("UPLOAD", "upload", "py")).diagnosis.conclusion == "possible_solar_cell_failure"
    with pytest.raises(DataValidationError):
        pipeline.prepare_upload(pd.DataFrame({"a": [1, 2]}))


def test_wrong_sampling_interval_gives_no_unsupported_fault():
    from tests.conftest import make_wide
    df = make_wide([2500.0] * 40, [10.0] * 40, step=7.0)
    clean, _ = pipeline.prepare_upload(df)
    res = pipeline.analyze(clean, panels=["py"])
    d = res.panel_diagnoses[0].diagnosis
    assert d.conclusion != "possible_solar_cell_failure"


def test_filters_applied():
    wide = pipeline.load_dataset_wide()
    res = pipeline.analyze(wide, satellites=["RAAVANA"], panels=["py"], time_range=(0, 1000))
    assert len(res.panel_diagnoses) == 2 and res.observations["Elapsed_s"].max() <= 1000
    with pytest.raises(DataValidationError):
        pipeline.analyze(wide, satellites=["NOPE"])
