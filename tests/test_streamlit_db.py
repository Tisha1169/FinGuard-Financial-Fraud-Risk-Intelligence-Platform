"""Tests for streamlit_app/db.py's pure helper logic - no DB needed.
_parse_evidence specifically guards against a real bug found during Phase
10 manual testing: rules_triggered.evidence coming back as a JSON-encoded
string instead of a parsed object from raw SQLAlchemy/pandas queries.
"""
import pandas as pd

from streamlit_app.db import _parse_evidence


def test_parse_evidence_parses_json_string_in_dataframe():
    df = pd.DataFrame({"rule_id": ["R1"], "evidence": ['{"threshold": 3, "count": 5}']})
    result = _parse_evidence(df)
    assert result.iloc[0]["evidence"] == {"threshold": 3, "count": 5}


def test_parse_evidence_leaves_already_parsed_dict_alone_in_dataframe():
    df = pd.DataFrame({"rule_id": ["R1"], "evidence": [{"threshold": 3}]})
    result = _parse_evidence(df)
    assert result.iloc[0]["evidence"] == {"threshold": 3}


def test_parse_evidence_handles_empty_dataframe():
    df = pd.DataFrame({"rule_id": [], "evidence": []})
    result = _parse_evidence(df)
    assert result.empty


def test_parse_evidence_parses_json_string_in_row_list():
    rows = [{"rule_id": "R1", "evidence": '{"a": 1}'}, {"rule_id": "R2", "evidence": '{"b": 2}'}]
    result = _parse_evidence(rows)
    assert result[0]["evidence"] == {"a": 1}
    assert result[1]["evidence"] == {"b": 2}


def test_parse_evidence_leaves_already_parsed_dict_alone_in_row_list():
    rows = [{"rule_id": "R1", "evidence": {"a": 1}}]
    result = _parse_evidence(rows)
    assert result[0]["evidence"] == {"a": 1}


def test_parse_evidence_handles_empty_row_list():
    assert _parse_evidence([]) == []
