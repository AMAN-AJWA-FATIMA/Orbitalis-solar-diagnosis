import sqlite3

import pytest

from src import database, pipeline
from tests.conftest import make_wide


@pytest.fixture()
def result():
    from src import preprocessing as pp
    clean, _ = pp.clean_wide(make_wide([2.0] * 30, [10.0] * 30))
    return pipeline.analyze(clean, source="upload", panels=["py"])


def test_save_and_retrieve(result, tmp_path):
    db = tmp_path / "t.db"
    ids = database.save_analysis(result, db)
    assert len(ids) == 1 and database.count_diagnoses(db) == 1
    rec = database.get_diagnosis(ids[0], db)
    assert rec["satellite_id"] == "TESTSAT" and rec["diagnosis_status"] == "possible_documented_fault"
    assert rec["matched_rules"].startswith("R001") and rec["facts"]["v_below_threshold_count"] == 30


def test_evidence_and_explanation_stored(result, tmp_path):
    db = tmp_path / "t.db"
    rec = database.get_diagnosis(database.save_analysis(result, db)[0], db)
    assert rec["evidence"] and "R010" in rec["explanation"]["reasoning_text"]
    assert rec["explanation"]["suggested_action"]


def test_list_filters(result, tmp_path):
    db = tmp_path / "t.db"
    database.save_analysis(result, db)
    assert len(database.list_diagnoses(db, satellite="TESTSAT")) == 1
    assert database.list_diagnoses(db, status="normal").empty


def test_unknown_id_returns_none(tmp_path):
    assert database.get_diagnosis(999, tmp_path / "t.db") is None


def test_foreign_key_enforced_and_parameterised(tmp_path):
    db = tmp_path / "t.db"; database.init_db(db)
    with pytest.raises(sqlite3.IntegrityError):
        with database.connect(db) as c:
            c.execute("INSERT INTO diagnosis_evidence (diagnosis_id, feature_name) VALUES (42, 'x')")
    assert database.list_diagnoses(db, satellite="x'; DROP TABLE diagnoses;--").empty
    assert database.count_diagnoses(db) == 0
