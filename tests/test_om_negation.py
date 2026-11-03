"""Unit tests for the ontology-mapping negation guard."""

from unittest.mock import Mock

import pandas as pd
import pytest

from metaharmonizer.engine.ontology_mapping_engine import OntoMapEngine
from metaharmonizer.utils.negation import detect_negation, polarity_compatible


def _engine(category: str = "treatment", top_k: int = 2):
    engine = OntoMapEngine.__new__(OntoMapEngine)
    engine.category = category
    engine.top_k = top_k
    engine._test_or_prod = "prod"
    engine.ground_truth_map = {}
    engine.negation_guard = True
    engine._logger = Mock()
    return engine


@pytest.mark.parametrize(
    ("query", "trigger"),
    [
        ("no aspirin", "no"),
        ("  NONE (antihypertensive treatment-naive)", "none"),
        ("not taking insulin", "not"),
        ("without metformin", "without"),
        ("antihypertensive treatment-naive", "treatment-naive"),
        ("systemic-therapy-naïve", "naïve"),
    ],
)
def test_detects_explicit_treatment_negation(query, trigger):
    assert detect_negation(query, category="treatment") == trigger


@pytest.mark.parametrize(
    "query",
    [
        "metformin",
        "insulin treatment",
        "Noonan syndrome",
        "notch inhibitor",
        "aspirin not otherwise specified",
    ],
)
def test_does_not_guard_affirmative_terms(query):
    assert detect_negation(query, category="treatment") is None


def test_naive_suffix_is_not_guarded_outside_treatment_category():
    assert detect_negation("naive T cell", category="phenotype") is None


def test_treatment_category_is_case_insensitive():
    assert detect_negation(
        "antihypertensive treatment-naive",
        category="Treatment",
    ) == "treatment-naive"


def test_without_inside_official_disease_name_is_not_guarded():
    query = "acute myeloid leukemia without maturation"
    assert detect_negation(query, category="disease") is None


@pytest.mark.parametrize(
    ("query", "candidate", "expected"),
    [
        ("no metformin", "Metformin", False),
        ("no metformin use", "No Metformin Use", True),
        ("metformin", "No Metformin Use", False),
        ("metformin", "Metformin", True),
        ("antihypertensive treatment-naive", "Treatment Naive", True),
    ],
)
def test_candidate_polarity_compatibility(query, candidate, expected):
    assert polarity_compatible(
        query, candidate, category="treatment") is expected


def test_pairwise_guard_removes_mismatches_and_compacts_candidates():
    engine = _engine(top_k=2)
    engine._test_or_prod = "test"
    engine.ground_truth_map = {"no metformin": "No Metformin Use"}
    results = pd.DataFrame({
        "original_value": ["no metformin"],
        "curated_ontology": ["No Metformin Use"],
        "match_level": [3],
        "stage": [2.0],
        "match1": ["Metformin"],
        "match1_score": [0.91],
        "match2": ["Insulin"],
        "match2_score": [0.86],
        "match3": ["No Metformin Use"],
        "match3_score": [0.82],
        "match3_similarity_score": [0.80],
        "match3_reranker_score": [0.82],
        "match4": ["Metformin Treatment-Naive"],
        "match4_score": [0.71],
        "match4_similarity_score": [0.70],
        "match4_reranker_score": [0.71],
    })

    result = engine._apply_negation_guard(results)

    assert result.loc[0, "match1"] == "No Metformin Use"
    assert result.loc[0, "match1_score"] == 0.82
    assert result.loc[0, "match1_similarity_score"] == 0.80
    assert result.loc[0, "match1_reranker_score"] == 0.82
    assert result.loc[0, "match2"] == "Metformin Treatment-Naive"
    assert result.loc[0, "match2_score"] == 0.71
    assert not any(
        column.startswith(("match3", "match4"))
        for column in result.columns
    )
    assert result.loc[0, "match_level"] == 1


def test_candidate_retrieval_uses_at_most_double_top_k():
    engine = _engine(top_k=3)
    engine.corpus = [f"term-{i}" for i in range(20)]
    assert engine._candidate_top_k() == 6

    engine.negation_guard = False
    assert engine._candidate_top_k() == 3


def test_candidate_retrieval_respects_available_count():
    engine = _engine(top_k=3)
    engine.corpus = ["one", "two", "three", "four"]
    assert engine._candidate_top_k() == 4
    assert engine._candidate_top_k(20) == 6


def test_pairwise_guard_filters_negated_candidate_for_affirmed_query():
    engine = _engine(top_k=1)
    results = pd.DataFrame({
        "original_value": ["metformin"],
        "curated_ontology": ["Not Found"],
        "match_level": [99],
        "stage": [2.0],
        "match1": ["No Metformin Use"],
        "match1_score": [0.91],
    })

    result = engine._apply_negation_guard(results)

    assert pd.isna(result.loc[0, "match1"])
    assert result.loc[0, "match1_score"] == 0.0


def test_disabled_guard_does_not_filter_candidates():
    engine = _engine(top_k=1)
    engine.negation_guard = False
    results = pd.DataFrame({
        "original_value": ["no metformin"],
        "curated_ontology": ["Not Found"],
        "match_level": [99],
        "stage": [2.0],
        "match1": ["Metformin"],
        "match1_score": [0.91],
    })

    result = engine._apply_negation_guard(results)

    assert result.loc[0, "match1"] == "Metformin"


def test_run_matches_query_before_filtering_candidate():
    engine = _engine(top_k=1)
    engine.query = ["no metformin"]
    engine.corpus = ["Metformin", "No Metformin Use"]
    engine.output_dir = None
    engine.s2_strategy = "lm"
    engine.s3_strategy = None
    engine.s4_strategy = None
    engine.other_params = {"skip_stage25": True}
    engine._map_shortname_to_fullname = lambda queries: {
        query: query for query in queries
    }
    model = Mock()
    model.get_match_results.return_value = pd.DataFrame({
        "original_value": ["no metformin"],
        "curated_ontology": ["Not Found"],
        "match_level": [99],
        "match1": ["Metformin"],
        "match1_score": [0.91],
    })
    engine._om_model_from_strategy = Mock(return_value=model)

    result = engine.run()

    engine._om_model_from_strategy.assert_called_once()
    model.get_match_results.assert_called_once()
    assert model.get_match_results.call_args.kwargs["top_k"] == 2
    assert result.loc[0, "query"] == "no metformin"
    assert pd.isna(result.loc[0, "match1"])
    assert result.loc[0, "match1_score"] == 0.0
    assert result.loc[0, "polarity"] == "NEGATED"


def test_run_prefers_exact_match_over_negation_guard():
    engine = _engine(top_k=1)
    engine.query = ["no treatment"]
    engine.corpus = ["no treatment"]
    engine.negation_guard = True
    engine.output_dir = None
    engine._om_model_from_strategy = Mock(
        side_effect=AssertionError("matcher should not be called")
    )

    result = engine.run()

    assert result.loc[0, "query"] == "no treatment"
    assert result.loc[0, "match1"] == "no treatment"
    assert result.loc[0, "stage"] == 1.0
    assert result.loc[0, "polarity"] == "NEGATED"
    assert result.loc[0, "negation_trigger"] == "no"
    engine._om_model_from_strategy.assert_not_called()
