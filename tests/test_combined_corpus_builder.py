"""Tests for CombinedCorpusBuilder — static+dynamic union corpus assembly.

Network is mocked at two seams: CorpusBuilder.build_sync (dynamic branches) and
OLSDb.get_term (static terms), so these run fully offline.
"""
import pandas as pd
import pytest

from metaharmonizer.knowledge_db.combined_corpus_builder import (
    CombinedCorpusBuilder,
    normalize_obo_id,
    _split_enum_field,
    _split_allowedvalues_enum,
    field_has_corpus_layer,
    build_schema_corpus,
    schema_to_ontology_terms,
    merge_match_results,
    run_ontology_partitioned,
)


def _rec(obo_id: str, label: str = "") -> dict:
    prefix = obo_id.split(":", 1)[0]
    return {
        "obo_id": obo_id,
        "label": label or obo_id,
        "ontology_prefix": prefix,
        "ontology_name": prefix.lower(),
        "iri": f"http://purl.obolibrary.org/obo/{obo_id.replace(':', '_')}",
        "short_form": obo_id.replace(":", "_"),
        "description": None,
        "type": "class",
    }


# --------------------------- pure helpers ---------------------------------- #

class TestPureHelpers:
    def test_normalize_underscore_to_colon(self):
        assert normalize_obo_id("EFO_0000408") == "EFO:0000408"

    def test_normalize_colon_untouched(self):
        assert normalize_obo_id("NCIT:C7057") == "NCIT:C7057"

    def test_normalize_only_first_separator(self):
        # Local ids containing an underscore must survive intact.
        assert normalize_obo_id("FOO_bar_baz") == "FOO:bar_baz"

    def test_split_pipe_field(self):
        assert _split_enum_field("NCIT:C7057|MONDO:0000001") == [
            "NCIT:C7057", "MONDO:0000001"]

    def test_split_drops_na_and_blanks(self):
        assert _split_enum_field("NCIT:C1|NA| |MONDO:2") == ["NCIT:C1", "MONDO:2"]

    def test_split_empty_variants(self):
        assert _split_enum_field(None) == []
        assert _split_enum_field("NA") == []
        assert _split_enum_field(float("nan")) == []
        assert _split_enum_field("") == []

    def test_dedupe_first_wins(self):
        recs = [_rec("NCIT:C1", "a"), _rec("NCIT:C1", "dup"), _rec("NCIT:C2")]
        out = CombinedCorpusBuilder._dedupe(recs)
        assert [r["obo_id"] for r in out] == ["NCIT:C1", "NCIT:C2"]
        assert out[0]["label"] == "a"  # first occurrence retained

    def test_partition_by_ontology(self):
        recs = [_rec("NCIT:C1"), _rec("MONDO:2"), _rec("NCIT:C3")]
        groups = CombinedCorpusBuilder.partition_by_ontology(recs)
        assert set(groups) == {"ncit", "mondo"}
        assert [r["obo_id"] for r in groups["ncit"]] == ["NCIT:C1", "NCIT:C3"]

    def test_to_dataframe_column_order(self):
        df = CombinedCorpusBuilder.to_dataframe([_rec("NCIT:C1")])
        assert list(df.columns) == [
            "iri", "ontology_name", "ontology_prefix", "short_form",
            "description", "label", "obo_id", "type"]


# --------------------------- build() union --------------------------------- #

class _FakeNCI:
    """Mock NCI EVSREST client: each root -> one descendant; concepts carry a synonym."""
    def __init__(self):
        self.descendants_calls = []
        self.concept_calls = []

    async def get_descendants(self, code, client=None, max_level=50, page_size=1000):
        self.descendants_calls.append((code, max_level))
        return [{"code": code + "d", "name": f"child of {code}"}]

    async def get_custom_concepts_by_codes(self, codes, client=None):
        self.concept_calls.append(list(codes))
        return {c: {"name": f"label:{c}", "synonyms": [{"name": f"syn:{c}"}]}
                for c in codes}


@pytest.fixture
def builder(monkeypatch):
    b = CombinedCorpusBuilder()

    # Mock OLS dynamic-branch expansion: each root -> root record + one descendant.
    def fake_build_sync(root_term_id, include_root=True, include_hierarchy=False):
        out = []
        if include_root:
            out.append(_rec(root_term_id, f"root:{root_term_id}"))
        out.append(_rec(root_term_id + "d", f"child:{root_term_id}"))
        return out

    monkeypatch.setattr(b._builder, "build_sync", fake_build_sync)

    # Mock OLS static-term fetch: get_term returns a minimal raw OLS dict.
    async def fake_get_term(obo_id, ontology=None, client=None):
        return {"obo_id": obo_id, "label": f"static:{obo_id}",
                "ontology_prefix": obo_id.split(":", 1)[0]}

    monkeypatch.setattr(b._ols, "get_term", fake_get_term)

    # Preset NCI client so NCIt routing never hits the network / constructs NCIDb.
    b._nci_client = _FakeNCI()
    return b


class TestBuild:
    def test_multiple_dynamic_roots_union(self, builder):
        recs = builder.build(dynamic_roots=["NCIT:C7057", "MONDO:0000001"])
        ids = {r["obo_id"] for r in recs}
        assert ids == {"NCIT:C7057", "NCIT:C7057d",
                       "MONDO:0000001", "MONDO:0000001d"}

    def test_static_plus_dynamic(self, builder):
        recs = builder.build(dynamic_roots=["NCIT:C1908"],
                             static_terms=["NCIT:C41132"])
        ids = [r["obo_id"] for r in recs]
        assert "NCIT:C41132" in ids           # static merged in
        assert "NCIT:C1908" in ids            # dynamic root
        assert "NCIT:C1908d" in ids           # dynamic descendant
        # static term appears after dynamic branch
        assert ids.index("NCIT:C41132") > ids.index("NCIT:C1908")

    def test_static_overlap_deduped(self, builder):
        # Static term equal to a dynamic descendant must not duplicate.
        recs = builder.build(dynamic_roots=["NCIT:C1"],
                             static_terms=["NCIT:C1d"])
        ids = [r["obo_id"] for r in recs]
        assert ids.count("NCIT:C1d") == 1

    def test_underscore_root_normalized(self, builder):
        recs = builder.build(dynamic_roots=["EFO_0000408"])
        assert any(r["obo_id"] == "EFO:0000408" for r in recs)

    def test_from_schema_row_combined_field(self, builder):
        row = {
            "dynamic.enum": "NCIT:C7057|MONDO:0000001",
            "static.enum": "NCIT:C115935",
            "dynamic.enum.property": "descendant",
            "corpus.type": "dynamic_enum|static_enum",
        }
        recs = builder.build_from_schema_row(row)
        ids = {r["obo_id"] for r in recs}
        assert {"NCIT:C7057", "MONDO:0000001", "NCIT:C115935"} <= ids

    def test_from_schema_row_pandas_series(self, builder):
        row = pd.Series({
            "dynamic.enum": "NCIT:C1908",
            "static.enum": "NCIT:C41132",
            "dynamic.enum.property": float("nan"),  # -> default "descendant"
        })
        recs = builder.build_from_schema_row(row)
        assert {"NCIT:C1908", "NCIT:C41132"} <= {r["obo_id"] for r in recs}

    def test_ncit_root_routes_to_evsrest_not_ols(self, builder, monkeypatch):
        ols_called = []
        monkeypatch.setattr(builder._builder, "build_sync",
                            lambda *a, **k: ols_called.append(a) or [])
        recs = builder.build(dynamic_roots=["NCIT:C7057"])
        assert builder._nci_client.descendants_calls == [("C7057", 50)]
        assert ols_called == []  # NCIt must not touch OLS
        assert {r["obo_id"] for r in recs} == {"NCIT:C7057", "NCIT:C7057d"}
        assert recs[0]["ontology_prefix"] == "NCIT"

    def test_non_ncit_root_routes_to_ols_not_evsrest(self, builder):
        recs = builder.build(dynamic_roots=["MONDO:0000001"])
        assert builder._nci_client.descendants_calls == []  # OLS-only path
        assert {r["obo_id"] for r in recs} == {"MONDO:0000001", "MONDO:0000001d"}

    def test_children_property_uses_maxlevel_1_for_ncit(self, builder):
        builder.build(dynamic_roots=["NCIT:C7057"], prop="children")
        assert builder._nci_client.descendants_calls == [("C7057", 1)]

    def test_mixed_corpus_partitions_for_engine(self, builder):
        recs = builder.build(dynamic_roots=["NCIT:C7057", "MONDO:0000001"])
        groups = builder.partition_by_ontology(recs)
        # Each group is single-ontology -> individually loadable by OntoMapEngine.
        assert set(groups) == {"ncit", "mondo"}
        for recs_g in groups.values():
            prefixes = {r["obo_id"].split(":", 1)[0].lower() for r in recs_g}
            assert len(prefixes) == 1


# --------------------- merge_match_results (pure) -------------------------- #

def _result_frame(rows, top_k=3):
    """Build an OntoMapEngine-shaped result frame from (query, [(label, score)...])."""
    records = []
    for query, cands in rows:
        rec = {"query": query, "ref_match": "Not Found"}
        for i in range(1, top_k + 1):
            if i <= len(cands):
                rec[f"match{i}"], rec[f"match{i}_score"] = cands[i - 1]
            else:
                rec[f"match{i}"], rec[f"match{i}_score"] = None, 0.0
        records.append(rec)
    return pd.DataFrame(records)


class TestMergeMatchResults:
    def test_pools_and_ranks_across_ontologies(self):
        ncit = _result_frame([("crohn", [("Crohn Disease NCIT", 0.80),
                                          ("Colitis", 0.55)])])
        mondo = _result_frame([("crohn", [("Crohn disease MONDO", 0.91),
                                           ("IBD", 0.60)])])
        merged = merge_match_results([ncit, mondo], top_k=3)
        row = merged.iloc[0]
        # Highest score across both frames wins the top slot.
        assert row["match1"] == "Crohn disease MONDO"
        assert row["match1_score"] == 0.91
        assert row["match2"] == "Crohn Disease NCIT"
        assert row["match3"] == "IBD"  # 0.60 > Colitis 0.55

    def test_dedupes_same_label_keeps_max(self):
        f1 = _result_frame([("x", [("Same", 0.5)])])
        f2 = _result_frame([("x", [("Same", 0.7)])])
        merged = merge_match_results([f1, f2], top_k=3)
        row = merged.iloc[0]
        assert row["match1"] == "Same" and row["match1_score"] == 0.7
        assert row["match2"] is None  # only one distinct candidate

    def test_drops_not_found_sentinels(self):
        f = _result_frame([("y", [("Not Found", 1.0), ("Real", 0.4)])])
        merged = merge_match_results([f], top_k=3)
        row = merged.iloc[0]
        assert row["match1"] == "Real"

    def test_query_order_and_union(self):
        f1 = _result_frame([("a", [("A1", 0.9)]), ("b", [("B1", 0.8)])])
        f2 = _result_frame([("b", [("B2", 0.85)]), ("c", [("C1", 0.7)])])
        merged = merge_match_results([f1, f2], top_k=2)
        assert list(merged["query"]) == ["a", "b", "c"]

    def test_empty_input(self):
        assert merge_match_results([]).empty


# --------------------- allowedvalues / corpus-layer helpers ---------------- #

class TestAllowedValuesHelpers:
    def test_enum_pipe_list(self):
        assert _split_allowedvalues_enum("Female|Male") == ["Female", "Male"]

    def test_regex_excluded(self):
        assert _split_allowedvalues_enum("[0-9]+") == []
        assert _split_allowedvalues_enum("^\\d+(\\.\\d+)?$") == []
        assert _split_allowedvalues_enum(".+") == []

    def test_na_and_blank(self):
        assert _split_allowedvalues_enum("NA") == []
        assert _split_allowedvalues_enum(float("nan")) == []

    def test_field_has_corpus_layer(self):
        assert field_has_corpus_layer({"dynamic.enum": "NCIT:C1"})
        assert field_has_corpus_layer({"allowedvalues": "A|B"})
        assert field_has_corpus_layer({"static.enum": "NCIT:C1"})
        assert not field_has_corpus_layer({"allowedvalues": "[0-9]+"})  # regex only
        assert not field_has_corpus_layer({"allowedvalues": "NA",
                                           "dynamic.enum": "NA"})


# --------------------- to_ontology_terms (pure) ---------------------------- #

class TestToOntologyTerms:
    def test_labels_and_synonyms(self):
        recs = [
            {"label": "asthma", "synonyms": ["bronchial asthma"]},
            {"label": "Healthy", "synonyms": ["Well", "healthy"]},  # syn==label (ci) dropped
        ]
        ot = CombinedCorpusBuilder.to_ontology_terms(recs)
        assert ot["labels"] == ["asthma", "Healthy"]
        assert ot["synonym_lookup"]["bronchial asthma"] == "asthma"
        assert ot["synonym_lookup"]["Well"] == "Healthy"
        assert "healthy" not in ot["synonym_lookup"]  # equals its own label

    def test_label_dedup_and_first_synonym_wins(self):
        recs = [
            {"label": "A", "synonyms": ["x"]},
            {"label": "A", "synonyms": ["y"]},   # dup label
            {"label": "B", "synonyms": ["x"]},   # x already claimed by A
        ]
        ot = CombinedCorpusBuilder.to_ontology_terms(recs)
        assert ot["labels"] == ["A", "B"]
        assert ot["synonym_lookup"]["x"] == "A"


# --------------------- build_field_corpus (mocked engine) ------------------ #

class TestBuildFieldCorpus:
    def test_custom_enum_labels_only(self):
        # No network: custom labels, no static ids.
        b = CombinedCorpusBuilder()
        recs = b.build_field_corpus(
            {"col.name": "diet", "allowedvalues": "vegan|omnivore",
             "static.enum": "NA", "dynamic.enum": "NA"},
            fetch_static_synonyms=False,
        )
        assert [r["label"] for r in recs] == ["vegan", "omnivore"]
        assert all(r["obo_id"] == "" for r in recs)  # free-text

    def test_static_enum_positional_ids(self):
        b = CombinedCorpusBuilder()
        recs = b.build_field_corpus(
            {"col.name": "sex", "allowedvalues": "Female|Male",
             "static.enum": "NCIT:C16576|NCIT:C20197", "dynamic.enum": "NA"},
            fetch_static_synonyms=False,
        )
        assert recs[0]["label"] == "Female" and recs[0]["obo_id"] == "NCIT:C16576"
        assert recs[1]["label"] == "Male" and recs[1]["obo_id"] == "NCIT:C20197"

    def test_regex_field_yields_nothing(self):
        b = CombinedCorpusBuilder()
        recs = b.build_field_corpus(
            {"col.name": "bmi", "allowedvalues": "[0-9]+", "static.enum": "NA",
             "dynamic.enum": "NA"},
            fetch_static_synonyms=False,
        )
        assert recs == []

    def test_combined_dynamic_plus_static(self, builder):
        # builder fixture mocks build_sync (dynamic) + get_term (static fetch).
        recs = builder.build_field_corpus(
            {"col.name": "disease", "allowedvalues": "NA",
             "static.enum": "NCIT:C115935",
             "dynamic.enum": "NCIT:C7057", "dynamic.enum.property": "descendant"},
            fetch_static_synonyms=True,
        )
        ids = {r["obo_id"] for r in recs}
        assert "NCIT:C7057" in ids and "NCIT:C7057d" in ids  # dynamic branch
        assert "NCIT:C115935" in ids                         # standalone static


# --------------------- build_schema_corpus (orchestrator) ------------------ #

class TestBuildSchemaCorpus:
    def test_skips_non_corpus_fields_and_projects(self):
        # All corpus-bearing fields here are custom labels (no network).
        df = pd.DataFrame([
            {"col.name": "diet", "allowedvalues": "vegan|omnivore",
             "static.enum": "NA", "dynamic.enum": "NA",
             "dynamic.enum.property": "NA"},
            {"col.name": "bmi", "allowedvalues": "[0-9]+", "static.enum": "NA",
             "dynamic.enum": "NA", "dynamic.enum.property": "NA"},   # regex -> skip
            {"col.name": "note", "allowedvalues": "NA", "static.enum": "NA",
             "dynamic.enum": "NA", "dynamic.enum.property": "NA"},    # plain -> skip
        ])
        # diet uses no network (custom labels); disable synonym fetch anyway.
        corpora = build_schema_corpus(df, fetch_static_synonyms=False)
        assert set(corpora) == {"diet"}
        ot = schema_to_ontology_terms(corpora)
        assert ot["diet"]["labels"] == ["vegan", "omnivore"]


# --------------------- run_ontology_partitioned (mocked engine) ------------ #

class TestRunOntologyPartitioned:
    def test_one_engine_per_ontology_then_merge(self, builder, monkeypatch):
        recs = builder.build(dynamic_roots=["NCIT:C7057", "MONDO:0000001"])
        constructed = []

        class FakeEngine:
            def __init__(self, *, corpus_category, query_ls, top_k,
                         corpus_df, ontology_source, ground_truth_map=None,
                         **kw):
                # Each engine must receive a single-ontology corpus.
                prefixes = {c.split(":", 1)[0].lower()
                            for c in corpus_df["obo_id"] if ":" in str(c)}
                assert len(prefixes) == 1, prefixes
                self.ontology_source = ontology_source
                self.query_ls = query_ls
                self.top_k = top_k
                constructed.append(ontology_source)

            def run(self):
                # NCIT weak, MONDO strong -> MONDO should win the merge.
                score = 0.9 if self.ontology_source == "mondo" else 0.6
                return _result_frame(
                    [(q, [(f"{q}-{self.ontology_source}", score)])
                     for q in self.query_ls],
                    top_k=self.top_k,
                )

        monkeypatch.setattr(
            "metaharmonizer.engine.ontology_mapping_engine.OntoMapEngine",
            FakeEngine,
        )
        merged, partials = run_ontology_partitioned(
            recs, corpus_category="disease", query_ls=["crohn"],
            top_k=2, return_partials=True,
        )
        assert set(constructed) == {"ncit", "mondo"}
        assert set(partials) == {"ncit", "mondo"}
        row = merged.iloc[0]
        assert row["match1"] == "crohn-mondo" and row["match1_score"] == 0.9
        assert row["match2"] == "crohn-ncit"
