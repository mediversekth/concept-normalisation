"""Check the notebook's actual scoring and truth-alignment functions without services."""
from concept_normalisation import notebook_evaluation as evaluation

import numpy as np
import pandas as pd
import pytest


def test_set_recall_is_not_hit_rate():
    result = evaluation.score_ids(["a", "x", "b"], {"a", "b", "c", "d"}, [1, 3], 5)
    assert result == {"Top-1 Accuracy": 1.0, "MRR@5": 1.0, "Recall@1": .25, "Recall@3": .5}


def test_reciprocal_rank_and_shared_cutoff():
    score = evaluation.score_ids
    assert score(["x", "a"], {"a"}, [1, 5], 5)["MRR@5"] == .5
    result = score(["1", "2", "3", "4", "5", "a"], {"a"}, [5], 5)
    assert all(value == 0 for value in result.values())
    assert all(value == 0 for value in score([], {"a"}, [1, 5], 5).values())
    with pytest.raises(ValueError, match="Unlabelled"):
        score([], set(), [1], 5)


def test_saved_order_and_concept_deduplication():
    values = np.array([{"sctid": "x", "score": .1}, {"sctid": "x", "score": .9},
                       {"sctid": "a", "score": .8}], dtype=object)
    ids = evaluation.candidate_ids(values, "sctid")
    assert ids == ["x", "a"]
    assert evaluation.score_ids(ids, {"a"}, [2], 5)["MRR@5"] == .5


def test_null_and_malformed_predictions():
    ids = evaluation.candidate_ids
    for empty in [None, np.nan, pd.NA, []]:
        assert ids(empty, "sctid") == []
    assert ids('[{"sctid": "123"}]', "sctid") == ["123"]
    with pytest.raises(ValueError):
        ids([{"wrong_key": "123"}], "sctid")
    with pytest.raises(ValueError):
        ids({"sctid": "123"}, "sctid")


def test_large_ids_and_legacy_integral_strings():
    normalise = evaluation.normalise_concept
    assert normalise("129032061000119103") == "129032061000119103"
    assert normalise("66657009.0") == "66657009"
    assert evaluation.normalise_id("564.00") == "564.00"
    with pytest.raises(ValueError, match="rounded"):
        normalise("129032061000119104.0")
    with pytest.raises(ValueError, match="Float"):
        normalise(123.0)


def test_truth_join_does_not_depend_on_row_order():
    truth = pd.DataFrame({"icd10": ["B", "A"], "snomed": ["2|3", "1"]})
    data = pd.DataFrame({"icd10": ["A", "B", "A"], "diagnosis_text": ["a", "b", "a"]})
    result = evaluation.attach_gold(data, truth, "icd10")
    assert result.gold.tolist() == [frozenset({"1"}), frozenset({"2", "3"}), frozenset({"1"})]
    assert result.query_key.nunique() == 2


def test_restore_exact_icd_key_from_record_identity():
    truth = pd.DataFrame({"diagnosisid": ["10"], "patientunitstayid": ["20"],
                          "diagnosisstring": ["a"], "icd9": ["564.00"], "snomed": ["14760008"]})
    data = truth.copy()
    data["icd9"] = "564.0"
    data["snomed"] = "14760008.0"
    result = evaluation.attach_gold(data, truth, "icd9")
    assert result.icd9.iloc[0] == "564.00"
    assert result.icd_key_restored.iloc[0]
    assert result.gold.iloc[0] == frozenset({"14760008"})


def test_conflicting_or_stale_truth_fails():
    truth = pd.DataFrame({"icd10": ["A", "A"], "snomed": ["1", "2"]})
    data = pd.DataFrame({"icd10": ["A"], "snomed": ["9"], "diagnosis_text": ["a"]})
    with pytest.raises(ValueError, match="Conflicting"):
        evaluation.attach_gold(data, truth, "icd10")
    with pytest.raises(ValueError, match="disagree"):
        evaluation.attach_gold(data, truth.iloc[:1], "icd10")


def test_aggregation_keeps_row_and_query_weighting_separate():
    data = pd.DataFrame({
        "row_id": [0, 1, 2],
        "query_key": [("repeated", "A"), ("repeated", "A"), ("rare", "B")],
        "diagnosis_text": ["repeated", "repeated", "rare"],
        "gold": [frozenset({"1"})] * 3,
        "algorithm_graphrag_matches": [[{"sctid": "1"}], [{"sctid": "1"}], []],
    })
    per_row, per_query, scores = evaluation.evaluate_methods(
        {"ICD9": data}, ks=[1, 3], mrr_depth=3,
    )
    assert len(per_row) == 3
    assert len(per_query) == 2
    scores = scores.set_index("view")
    assert scores.loc["row", "Top-1 Accuracy"] == pytest.approx(2 / 3)
    assert scores.loc["unique query", "Top-1 Accuracy"] == .5
    assert "MRR@3" in scores
    assert "MRR@5" not in scores


def test_missing_logs_preserve_coverage(tmp_path):
    data = pd.DataFrame({
        "query_key": [("query", "A")],
        "diagnosis_text": ["query"],
        "gold": [frozenset({"1"})],
        "algorithm_graphrag_matches": [[{"sctid": "1"}]],
    })
    traces, coverage = evaluation.load_traces(
        {"ICD9": data}, {"ICD9": {"log": tmp_path / "missing.jsonl"}},
    )
    assert traces == []
    assert coverage.status.tolist() == ["no matching log"]
    assert coverage.row_count.tolist() == [1]


def test_outside_pool_output_stays_visible_in_trace_analysis():
    trace = {
        "dataset": "ICD10",
        "query_key": ("query", "A"),
        "diagnosis_text": "query",
        "gold": frozenset({"1"}),
        "entry": {
            "candidates": [{"sctid": "2"}],
            "llm_candidates": [{"sctid": "2"}],
            "matches": [{"sctid": "1"}],
        },
    }
    stages, outcomes, membership = evaluation.analyse_traces([trace])
    assert stages["any target present"].tolist() == [0, 0, 1, 1]
    assert outcomes.outcome.tolist() == ["Outside-pool output"]
    assert membership.outside_pool_ids.tolist() == ["1"]
    assert evaluation.select_examples([trace], {"ICD10": []}) == [trace]
