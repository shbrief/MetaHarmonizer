"""Regression tests for OntoMapEngine._build_exact_df() (Stage 1).

Stage 1 previously set ``match1`` from ``curated_ontology``, which broke both
modes from one line:

* prod -- ``ground_truth_map`` is the placeholder ``{query: "Not Found"}``, so
  every exact corpus match was returned as ``"Not Found"`` with
  ``match1_score = 1.00``: a silent failure at maximum confidence, affecting
  ~32% of queries on the ukbb_efo benchmark.
* test -- the ground truth was substituted into ``match1`` and
  ``match_level`` was hardcoded to 1, so Stage 1 scored 100% by construction
  and the benchmark could not falsify it.

``match1`` must be the corpus label the query matched, in both modes.
"""
import pandas as pd

from metaharmonizer.engine.ontology_mapping_engine import OntoMapEngine


CORPUS = ["Senile cataract", "Asthma", "Pain", "Scoliosis"]


def _engine(ground_truth_map, mode):
    """Minimal stub; __init__ is heavy and irrelevant to this logic."""
    e = OntoMapEngine.__new__(OntoMapEngine)
    e.corpus = CORPUS
    e.ground_truth_map = ground_truth_map
    e._test_or_prod = mode
    e.top_k = 5
    return e


# ── prod mode ───────────────────────────────────────────────────────────────

def test_prod_returns_corpus_label_not_placeholder():
    """The original bug: "Not Found" returned at score 1.00 for exact hits."""
    queries = ["Pain", "Scoliosis"]
    gt = {q: "Not Found" for q in queries}       # what prod mode builds
    df = _engine(gt, "prod")._build_exact_df(queries)

    assert list(df["match1"]) == ["Pain", "Scoliosis"]
    assert "Not Found" not in set(df["match1"])


def test_prod_score_is_unchanged():
    queries = ["Pain"]
    df = _engine({"Pain": "Not Found"}, "prod")._build_exact_df(queries)
    assert df["match1_score"].iloc[0] == 1.00


def test_prod_curated_column_stays_not_found():
    """ref_match is an eval column; prod has no curated value to report."""
    df = _engine({"Pain": "Not Found"}, "prod")._build_exact_df(["Pain"])
    assert df["curated_ontology"].iloc[0] == "Not Found"


# ── test mode ───────────────────────────────────────────────────────────────

def test_test_mode_does_not_substitute_ground_truth():
    """'Senile cataract' is in the corpus but curated to a different label."""
    gt = {"Senile cataract": "age-related cataract"}
    df = _engine(gt, "test")._build_exact_df(["Senile cataract"])

    assert df["match1"].iloc[0] == "Senile cataract"
    assert df["match1"].iloc[0] != gt["Senile cataract"]


def test_test_mode_scores_a_divergent_curation_as_a_miss():
    """Previously hardcoded to 1, making the stage unfalsifiable."""
    gt = {"Senile cataract": "age-related cataract"}
    df = _engine(gt, "test")._build_exact_df(["Senile cataract"])
    assert df["match_level"].iloc[0] == 99


def test_test_mode_scores_an_agreeing_curation_as_a_hit():
    gt = {"Asthma": "Asthma"}
    df = _engine(gt, "test")._build_exact_df(["Asthma"])
    assert df["match_level"].iloc[0] == 1


def test_case_insensitive_match_returns_canonical_corpus_casing():
    gt = {"asthma": "asthma"}
    df = _engine(gt, "test")._build_exact_df(["asthma"])
    assert df["match1"].iloc[0] == "Asthma"


# ── shape ───────────────────────────────────────────────────────────────────

def test_empty_input_returns_empty_frame():
    df = _engine({}, "prod")._build_exact_df([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_stage_column_is_one():
    df = _engine({"Pain": "Not Found"}, "prod")._build_exact_df(["Pain"])
    assert df["stage"].iloc[0] == 1.0
