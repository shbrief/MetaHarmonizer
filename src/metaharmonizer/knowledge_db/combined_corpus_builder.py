"""Build a union corpus from static-enum term(s) plus dynamic-enum branch(es).

Realizes the schema's combined ``corpus.type = "dynamic_enum|static_enum"``
fields (see the MetaHarmonizer schema registry). For such a field the corpus is::

    final corpus = {static.enum terms} ∪ (descendants|children of each dynamic.enum root)

Concretely, ``curatedMetagenomicData``'s ``disease`` field declares::

    static.enum           = NCIT:C115935            (e.g. "Healthy")
    dynamic.enum          = NCIT:C7057|MONDO:0000001
    dynamic.enum.property = descendant

which this builder turns into one term list: the "Healthy" node merged with every
descendant of ``NCIT:C7057`` and every descendant of ``MONDO:0000001``.

Roots may span ontologies (NCIT + MONDO above); OLS4 serves them all, so building
is uniform. Consumption is not: ``OntoMapEngine`` rejects a single ``corpus_df``
whose codes span more than one ontology prefix (it partitions by prefix for its
SQLite/FAISS table naming). Use :meth:`partition_by_ontology` /
:meth:`save_partitioned` to emit one loadable corpus per ontology, then run the
engine once per ontology and merge candidates. A single-ontology combined field
(e.g. ``treatment``: static ``NCIT:C41132`` + dynamic ``NCIT:C1908``) loads
directly as one corpus.

Usage::

    builder = CombinedCorpusBuilder()
    records = builder.build(
        dynamic_roots=["NCIT:C7057", "MONDO:0000001"],
        static_terms=["NCIT:C115935"],
        prop="descendant",
    )
    # single ontology -> one CSV; mixed -> one CSV per ontology
    builder.save_partitioned(records, "data/corpus/disease")
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

import httpx
import pandas as pd

from metaharmonizer.custom_logger.custom_logger import CustomLogger
from metaharmonizer.knowledge_db.corpus_builder import CorpusBuilder
from metaharmonizer.knowledge_db.db_clients.ols_db import OLSDb
from metaharmonizer._async_utils import run_async

# Column order expected by OntoMapEngine's corpus CSV loader
# (mirrors OntoMapEngine._build_ols_corpus_csv).
_CORPUS_CSV_COLUMNS = [
    "iri", "ontology_name", "ontology_prefix", "short_form",
    "description", "label", "obo_id", "type",
]


def normalize_obo_id(term: str) -> str:
    """Normalize an ontology id to OBO ``PREFIX:LOCAL`` form.

    Accepts both ``EFO_0000408`` (underscore, as some schema columns store it)
    and ``EFO:0000408`` (colon). Only the *first* separator is treated as the
    prefix boundary, so ids whose local part contains ``_`` survive intact.
    """
    term = term.strip()
    if ":" in term:
        return term
    if "_" in term:
        prefix, local = term.split("_", 1)
        return f"{prefix}:{local}"
    return term


def _split_enum_field(value) -> list[str]:
    """Split a ``|``-delimited schema enum field into clean tokens.

    Returns ``[]`` for empty / ``NA`` cells. ``|`` is the standardized schema
    delimiter for multi-node dynamic.enum and multi-value static.enum fields.
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if text in ("", "NA"):
        return []
    return [t.strip() for t in text.split("|") if t.strip() not in ("", "NA")]


# Regex metacharacters that mark an `allowedvalues` cell as a pattern rather
# than an enum list. Mirrors the discrimination in build_schema_artifacts.py
# and tableToLinkmlSchema() so the three stay consistent.
_ALLOWEDVALUES_REGEX_RE = re.compile(r"^\^|\$$|\[|\]|\{|\}|\+")


def _split_allowedvalues_enum(value) -> list[str]:
    """Return the enum labels in an ``allowedvalues`` cell, or ``[]``.

    Empty / ``NA`` cells and regex patterns yield ``[]`` (regex fields are
    validation-only and never populate the corpus). A ``|``-list yields its
    trimmed labels.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if text in ("", "NA"):
        return []
    if _ALLOWEDVALUES_REGEX_RE.search(text):
        return []
    return [t.strip() for t in text.split("|") if t.strip() not in ("", "NA")]


class CombinedCorpusBuilder:
    """Union a static term set with one or more dynamic ontology branches.

    Ontology routing: NCIt terms (``NCIT:`` prefix) are expanded via the NCI
    EVSREST API (matching ``OntoMapEngine``'s NCIt corpus), all other ontologies
    via the EBI OLS4 API. Both are public; no API key is required.
    """

    def __init__(self):
        self._builder = CorpusBuilder()
        self._ols = self._builder._ols
        self._nci_client = None  # lazy — only created when an NCIt term appears
        self.logger = CustomLogger().custlogger(loglevel="INFO")

    @staticmethod
    def _is_ncit(term: str) -> bool:
        """True for an ``NCIT:``-prefixed (or ``NCIT_``) ontology id."""
        return normalize_obo_id(term).split(":", 1)[0].upper() == "NCIT"

    def _nci(self):
        """Lazily construct the NCI EVSREST client (no UMLS key needed here)."""
        if self._nci_client is None:
            from metaharmonizer.knowledge_db.db_clients.nci_db import NCIDb
            self._nci_client = NCIDb(os.getenv("UMLS_API_KEY"))
        return self._nci_client

    # ----------------------------- core API -----------------------------------

    def build(
        self,
        dynamic_roots: Iterable[str] = (),
        static_terms: Iterable[str] = (),
        prop: str = "descendant",
        include_root: bool = True,
        include_hierarchy: bool = False,
    ) -> list[dict]:
        """Build the union corpus.

        Parameters
        ----------
        dynamic_roots : iterable of str
            Root term ids whose branch is expanded (e.g. ``["NCIT:C7057",
            "MONDO:0000001"]``). Underscore ids are normalized to colon form.
        static_terms : iterable of str
            Individual ontology terms merged in as-is (the ``static.enum``
            column of a combined field).
        prop : str
            ``"descendant"`` (default, transitive) or ``"children"`` (direct
            subclasses only), matching ``dynamic.enum.property``.
        include_root : bool
            Include each dynamic root term itself (descendant mode only).
        include_hierarchy : bool
            Enrich each dynamic-branch record with parent/child labels
            (passed through to :class:`CorpusBuilder`). Costs ~2x API calls.

        Returns
        -------
        list[dict]
            Term records (CorpusBuilder schema), de-duplicated by ``obo_id``
            with first occurrence winning. Dynamic-branch terms come first,
            then any static terms not already present.
        """
        dynamic_roots = [normalize_obo_id(r) for r in dynamic_roots]
        static_terms = [normalize_obo_id(t) for t in static_terms]
        prop = (prop or "descendant").strip().lower()

        records: list[dict] = []
        for root in dynamic_roots:
            branch = self._build_branch(
                root, prop=prop, include_root=include_root,
                include_hierarchy=include_hierarchy)
            self.logger.info(
                f"Dynamic root {root} ({prop}): {len(branch)} terms")
            records.extend(branch)

        if static_terms:
            static_records = self._fetch_single_records(static_terms)
            self.logger.info(
                f"Static terms: {len(static_records)}/{len(static_terms)} fetched")
            records.extend(static_records)

        deduped = self._dedupe(records)
        self.logger.info(
            f"Combined corpus: {len(deduped)} unique terms "
            f"({len(dynamic_roots)} dynamic root(s), "
            f"{len(static_terms)} static term(s))")
        return deduped

    def build_from_schema_row(self, row, **kwargs) -> list[dict]:
        """Build from a schema-table row (dict or pandas Series).

        Reads ``dynamic.enum``, ``static.enum`` and ``dynamic.enum.property``
        (the standardized ``|``-delimited columns). Extra keyword arguments are
        forwarded to :meth:`build` (e.g. ``include_hierarchy=True``).
        """
        get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
        dynamic_roots = _split_enum_field(get("dynamic.enum"))
        static_terms = _split_enum_field(get("static.enum"))
        prop = get("dynamic.enum.property") or "descendant"
        if isinstance(prop, float) and pd.isna(prop):
            prop = "descendant"
        return self.build(
            dynamic_roots=dynamic_roots,
            static_terms=static_terms,
            prop=str(prop),
            **kwargs,
        )

    def build_field_corpus(
        self,
        row,
        *,
        include_hierarchy: bool = False,
        fetch_static_synonyms: bool = True,
    ) -> list[dict]:
        """Build the unified target corpus for one schema field.

        Unions all three corpus-bearing layers of a field into one term list:

        1. ``dynamic.enum`` roots -> descendant (or direct-child) branch;
        2. ``allowedvalues`` enum labels -> one record each. When ``static.enum``
           supplies a positional id for a label (custom/static enum), the label
           is bound to that id and, if *fetch_static_synonyms*, enriched with the
           ontology term's synonyms; custom labels with no id become free-text
           records (empty ``obo_id``);
        3. ``static.enum`` standalone terms -> fetched as-is (the combined
           ``dynamic_enum|static_enum`` case, where ``allowedvalues`` is empty and
           ``static.enum`` holds extra ontology terms such as "Healthy").

        Regex ``allowedvalues`` contribute nothing (validation-only). Records are
        de-duplicated by ``obo_id`` (or label, for free-text terms).

        Parameters
        ----------
        row : dict or pandas.Series
            A schema-table row.
        include_hierarchy : bool
            Passed to the dynamic-branch build (parent/child label enrichment).
        fetch_static_synonyms : bool
            Fetch ontology synonyms for static.enum-backed labels and standalone
            static terms. Set False for an offline / no-network build.
        """
        get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)

        records: list[dict] = []

        # 1. dynamic.enum branch
        dynamic_roots = _split_enum_field(get("dynamic.enum"))
        if dynamic_roots:
            prop = get("dynamic.enum.property") or "descendant"
            if isinstance(prop, float) and pd.isna(prop):
                prop = "descendant"
            records.extend(self.build(
                dynamic_roots=dynamic_roots,
                prop=str(prop),
                include_hierarchy=include_hierarchy,
            ))

        # 2 & 3. allowedvalues enum labels and static.enum terms
        av_labels = _split_allowedvalues_enum(get("allowedvalues"))
        se_tokens = _split_enum_field(get("static.enum"))

        if av_labels:
            # static.enum is positional to the labels when the counts line up
            # (custom_enum / static_enum); otherwise labels are id-less.
            positional = se_tokens if len(se_tokens) == len(av_labels) else []
            fetch_ids = [normalize_obo_id(t) for t in positional if t] \
                if fetch_static_synonyms else []
            id_to_rec = {}
            if fetch_ids:
                id_to_rec = {r["obo_id"]: r
                             for r in self._fetch_single_records(fetch_ids)}
            for i, label in enumerate(av_labels):
                raw_id = positional[i] if i < len(positional) else ""
                oid = normalize_obo_id(raw_id) if raw_id else ""
                records.append(self._label_record(label, oid, id_to_rec.get(oid)))
        elif se_tokens:
            # combined field: static.enum holds standalone terms, no labels
            static_ids = [normalize_obo_id(t) for t in se_tokens]
            if fetch_static_synonyms:
                records.extend(self._fetch_single_records(static_ids))
            else:
                records.extend(self._label_record("", oid) for oid in static_ids)

        deduped = self._dedupe(records)
        self.logger.info(
            f"Field corpus '{get('col.name')}': {len(deduped)} terms "
            f"({len(dynamic_roots)} dynamic root(s), {len(av_labels)} enum "
            f"label(s), {len(se_tokens)} static term(s))")
        return deduped

    @staticmethod
    def _label_record(label: str, obo_id: str = "", base: Optional[dict] = None) -> dict:
        """Build a corpus record for an enum label / static term.

        *label* is the preferred (curated) label kept as-is. *obo_id* binds it to
        an ontology term; *base* is an optional fetched record whose synonyms /
        description enrich the result (the curated label still wins).
        """
        prefix = obo_id.split(":", 1)[0] if ":" in obo_id else ""
        base = base or {}
        preferred = label or base.get("label", "")
        return {
            "iri": base.get("iri", "") or (
                f"http://purl.obolibrary.org/obo/{obo_id.replace(':', '_')}"
                if obo_id else ""),
            "ontology_name": base.get("ontology_name", "") or prefix.lower(),
            "ontology_prefix": base.get("ontology_prefix", "") or prefix,
            "short_form": base.get("short_form", "") or (
                obo_id.replace(":", "_") if obo_id else ""),
            "label": preferred,
            "obo_id": obo_id,
            "definitions": base.get("definitions", []),
            "description": base.get("description"),
            "synonyms": base.get("synonyms", []),
            "parents": base.get("parents", []),
            "children": base.get("children", []),
            "roles": base.get("roles", []),
            "type": "class",
        }

    # --------------------------- ontology routing ----------------------------

    def _build_branch(self, root: str, *, prop: str, include_root: bool,
                      include_hierarchy: bool) -> list[dict]:
        """Expand one dynamic root into its branch, routed by ontology.

        NCIt roots go through NCI EVSREST; all others through OLS4.
        """
        if self._is_ncit(root):
            return self._build_ncit_branch(root, prop=prop, include_root=include_root)
        if prop == "children":
            return self._build_children(root, include_root=include_root)
        return self._builder.build_sync(
            root_term_id=root,
            include_root=include_root,
            include_hierarchy=include_hierarchy,
        )

    def _build_ncit_branch(self, root: str, *, prop: str,
                           include_root: bool) -> list[dict]:
        """NCIt branch via EVSREST: root + descendants (or direct children)."""
        code = root.split(":", 1)[-1]
        max_level = 1 if prop == "children" else 50
        # Both EVSREST calls run in one event loop (single run_async) so the
        # client's rate limiter is not reused across loops.
        return run_async(self._async_ncit_branch(code, max_level, include_root))

    async def _async_ncit_branch(self, code: str, max_level: int,
                                 include_root: bool) -> list[dict]:
        nci = self._nci()
        async with httpx.AsyncClient() as client:
            descendants = await nci.get_descendants(
                code, client=client, max_level=max_level)
            codes: list[str] = []
            seen: set[str] = set()
            for c in ([code] if include_root else []) \
                    + [d.get("code") for d in descendants]:
                if c and c not in seen:
                    seen.add(c)
                    codes.append(c)
            concept_map = await nci.get_custom_concepts_by_codes(codes, client=client)
        records = [self._nci_record(c, concept_map.get(c, {})) for c in codes]
        return [r for r in records if r is not None]

    def _fetch_single_records(self, ids: Iterable[str]) -> list[dict]:
        """Fetch single-term records for *ids*, routing NCIt -> EVSREST, else OLS."""
        norm = [normalize_obo_id(i) for i in ids if i]
        ncit = [i for i in norm if self._is_ncit(i)]
        ols = [i for i in norm if not self._is_ncit(i)]
        records: list[dict] = []
        if ols:
            records.extend(run_async(self._fetch_terms(ols)))
        if ncit:
            records.extend(self._fetch_ncit_terms(ncit))
        return records

    def _fetch_ncit_terms(self, ids: list[str]) -> list[dict]:
        """Fetch NCIt single-term records via EVSREST concept lookup."""
        codes = [i.split(":", 1)[-1] for i in ids]
        concept_map = run_async(self._nci().get_custom_concepts_by_codes(codes))
        records = [self._nci_record(c, concept_map.get(c, {})) for c in codes]
        return [r for r in records if r is not None]

    @staticmethod
    def _nci_record(code: str, concept: dict) -> Optional[dict]:
        """Convert an NCI EVSREST concept payload into a corpus record.

        Mirrors the record shape ``OntoMapEngine`` writes for NCIt terms.
        Returns None when the concept has no label (e.g. a failed fetch).
        """
        label = str((concept or {}).get("name") or "").strip()
        if not label:
            return None

        def _names(items) -> list[str]:
            out, seen = [], set()
            for it in items or []:
                if isinstance(it, dict):
                    v = str(it.get("name") or "").strip()
                    if v and v.lower() not in seen:
                        seen.add(v.lower())
                        out.append(v)
            return out

        first_def = ""
        for it in concept.get("definitions", []) or []:
            if isinstance(it, dict):
                d = str(it.get("definition") or "").strip()
                if d:
                    first_def = d
                    break

        roles = []
        for it in concept.get("roles", []) or []:
            if isinstance(it, dict):
                roles.append({
                    "type": str(it.get("type") or "").strip(),
                    "related_name": str(it.get("relatedName") or "").strip(),
                    "related_code": str(it.get("relatedCode") or "").strip(),
                })

        return {
            "iri": f"http://purl.obolibrary.org/obo/NCIT_{code}",
            "ontology_name": "ncit",
            "ontology_prefix": "NCIT",
            "short_form": f"NCIT_{code}",
            "label": label,
            "obo_id": f"NCIT:{code}",
            "definitions": [first_def] if first_def else [],
            "description": first_def or None,
            "synonyms": _names(concept.get("synonyms")),
            "parents": _names(concept.get("parents")),
            "children": _names(concept.get("children")),
            "roles": roles,
            "type": "class",
        }

    # ----------------------------- helpers ------------------------------------

    async def _fetch_terms(self, obo_ids: list[str]) -> list[dict]:
        """Fetch single-term records for each id via OLS (parsed to the corpus schema)."""
        records = []
        async with httpx.AsyncClient() as client:
            for obo_id in obo_ids:
                try:
                    raw = await self._ols.get_term(obo_id, client=client)
                except Exception as exc:  # noqa: BLE001 - one bad id shouldn't sink the corpus
                    self.logger.warning(f"Static term {obo_id} skipped: {exc}")
                    continue
                records.append(CorpusBuilder._parse_term(raw))
        return records

    def _build_children(self, root: str, include_root: bool) -> list[dict]:
        """Direct-children corpus for ``dynamic.enum.property == "children"``."""
        return run_async(self._async_children(root, include_root))

    async def _async_children(self, root: str, include_root: bool) -> list[dict]:
        records: list[dict] = []
        async with httpx.AsyncClient() as client:
            root_raw = await self._ols.get_term(root, client=client)
            if include_root:
                records.append(CorpusBuilder._parse_term(root_raw))
            href = (root_raw.get("_links", {}).get("children", {}) or {}).get("href")
            while href:
                resp = await self._ols.fetch_one(client, href)
                if resp is None:
                    raise RuntimeError(
                        f"OLS children({root}): page fetch failed after "
                        f"{len(records)} terms. Aborting to avoid an "
                        f"incomplete corpus.")
                data = resp.json()
                for term in data.get("_embedded", {}).get("terms", []):
                    records.append(CorpusBuilder._parse_term(term))
                href = data.get("_links", {}).get("next", {}).get("href")
        return records

    @staticmethod
    def _dedupe(records: list[dict]) -> list[dict]:
        """De-duplicate (first occurrence wins), preserving order.

        Keyed by ``obo_id`` when present; custom / free-text terms carry no
        ontology id, so they fall back to a label key so they are neither
        dropped nor collapsed with an ontology term of the same label.
        """
        seen: set[str] = set()
        out: list[dict] = []
        for rec in records:
            oid = rec.get("obo_id") or rec.get("iri")
            key = oid if oid else "label::" + str(rec.get("label", "")).strip().lower()
            if key in ("", "label::") or key in seen:
                continue
            seen.add(key)
            out.append(rec)
        return out

    @staticmethod
    def partition_by_ontology(records: list[dict]) -> dict[str, list[dict]]:
        """Group records by ontology prefix (from ``obo_id``).

        The engine requires a single-ontology corpus per run; this splits a
        mixed corpus into per-ontology groups keyed by lowercased prefix
        (``"ncit"``, ``"mondo"``, ...).
        """
        groups: dict[str, list[dict]] = {}
        for rec in records:
            obo_id = rec.get("obo_id") or ""
            prefix = obo_id.split(":", 1)[0].lower() if ":" in obo_id else ""
            groups.setdefault(prefix or "unknown", []).append(rec)
        return groups

    @staticmethod
    def to_ontology_terms(records: list[dict]) -> dict:
        """Project corpus records into the R validator's ``ontology_terms`` shape.

        Returns ``{"labels": [...], "synonym_lookup": {synonym: preferred_label}}``
        as consumed by ``validateDataAgainstSchema`` (via ``.validate_dynamic_enum``).
        Labels are de-duplicated case-insensitively preserving order; a synonym
        already equal to a label, or already claimed by an earlier term, is not
        overwritten.
        """
        labels: list[str] = []
        seen_labels: set[str] = set()
        synonym_lookup: dict[str, str] = {}
        for rec in records:
            label = str(rec.get("label", "")).strip()
            if not label:
                continue
            key = label.lower()
            if key not in seen_labels:
                seen_labels.add(key)
                labels.append(label)
        label_keys = seen_labels
        for rec in records:
            label = str(rec.get("label", "")).strip()
            if not label:
                continue
            for syn in rec.get("synonyms", []):
                s = str(syn).strip()
                if not s or s.lower() in label_keys or s in synonym_lookup:
                    continue
                synonym_lookup[s] = label
        return {"labels": labels, "synonym_lookup": synonym_lookup}

    # ----------------------------- persistence --------------------------------

    @staticmethod
    def to_dataframe(records: list[dict]) -> pd.DataFrame:
        """Records -> corpus CSV DataFrame with the engine's column order."""
        df = pd.DataFrame(records)
        for col in _CORPUS_CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[_CORPUS_CSV_COLUMNS]

    def save(self, records: list[dict], output_csv: str,
             root_term_id: str = "") -> Path:
        """Save a single combined corpus CSV (+ rich JSON sidecar).

        Suitable when the corpus is single-ontology (loadable directly by
        ``OntoMapEngine``). For mixed corpora use :meth:`save_partitioned`.
        """
        df = self.to_dataframe(records)
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        # Rich JSON alongside, reusing CorpusBuilder's envelope format.
        self._builder.save(records, str(path.with_suffix(".json")),
                           root_term_id=root_term_id)
        self.logger.info(f"Saved combined corpus ({len(df)} terms) to {path}")
        return path.resolve()

    def save_partitioned(self, records: list[dict], out_prefix: str) -> dict[str, Path]:
        """Save one corpus CSV per ontology: ``<out_prefix>_<ontology>_corpus.csv``.

        Returns a mapping ``{ontology: csv_path}``. Each file is single-ontology
        and therefore directly loadable by ``OntoMapEngine``; run the engine per
        file and merge the candidate lists downstream.
        """
        groups = self.partition_by_ontology(records)
        written: dict[str, Path] = {}
        for ontology, recs in groups.items():
            csv_path = f"{out_prefix}_{ontology}_corpus.csv"
            written[ontology] = self.save(recs, csv_path)
        if len(groups) > 1:
            self.logger.info(
                f"Mixed-ontology corpus split into {len(groups)} files "
                f"({', '.join(groups)}); run OntoMapEngine per ontology.")
        return written


def field_has_corpus_layer(row) -> bool:
    """True if a schema row declares any corpus-bearing layer.

    Corpus layers are ``dynamic.enum``, enum ``allowedvalues`` (not a regex), and
    ``static.enum``. Regex-only / plain fields have no target corpus.
    """
    get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
    return bool(
        _split_enum_field(get("dynamic.enum"))
        or _split_allowedvalues_enum(get("allowedvalues"))
        or _split_enum_field(get("static.enum"))
    )


def build_schema_corpus(
    schema_df: pd.DataFrame,
    *,
    fields: Optional[Iterable[str]] = None,
    include_hierarchy: bool = False,
    fetch_static_synonyms: bool = True,
) -> dict[str, list[dict]]:
    """Build the unified target corpus for every corpus-bearing field in a schema.

    Returns ``{field_name: records}``. Fields with no corpus layer (regex-only,
    free-text, plain) are skipped. ``fields`` restricts to a subset.
    """
    builder = CombinedCorpusBuilder()
    only = set(fields) if fields is not None else None
    corpora: dict[str, list[dict]] = {}
    for _, row in schema_df.iterrows():
        name = row.get("col.name") if hasattr(row, "get") else row["col.name"]
        if only is not None and name not in only:
            continue
        if not field_has_corpus_layer(row):
            continue
        corpora[name] = builder.build_field_corpus(
            row,
            include_hierarchy=include_hierarchy,
            fetch_static_synonyms=fetch_static_synonyms,
        )
    return corpora


def schema_to_ontology_terms(corpora: dict[str, list[dict]]) -> dict[str, dict]:
    """Project a ``{field: records}`` map into ``{field: ontology_terms}``.

    The result is JSON-serializable and matches what the R
    ``validateDataAgainstSchema(ontology_terms=...)`` argument expects (after the
    ``loadOntologyTerms`` loader converts the object into named vectors).
    """
    return {
        field: CombinedCorpusBuilder.to_ontology_terms(records)
        for field, records in corpora.items()
    }


def build_combined_corpus(
    dynamic_roots: Iterable[str] = (),
    static_terms: Iterable[str] = (),
    prop: str = "descendant",
    include_hierarchy: bool = False,
) -> list[dict]:
    """Module-level convenience wrapper around :class:`CombinedCorpusBuilder`."""
    return CombinedCorpusBuilder().build(
        dynamic_roots=dynamic_roots,
        static_terms=static_terms,
        prop=prop,
        include_hierarchy=include_hierarchy,
    )


# --------------------------------------------------------------------------- #
# Mixed-ontology mapping: run OntoMapEngine per ontology, merge candidates.
#
# OntoMapEngine requires a single-ontology corpus per run. For a mixed combined
# corpus we run it once per ontology partition and merge the per-ontology top-k
# candidate lists into one ranked top-k per query. Stage 2/3 scores are cosine
# similarities from the same embedding models across ontologies, so they are
# directly comparable — this mirrors the engine's own cross-stage candidate
# merge (max score per distinct match, re-ranked).
# --------------------------------------------------------------------------- #

_MATCH_COL_RE = re.compile(r"^match(\d+)$")
_MATCH_SENTINELS = {"", "not found", "none", "nan", "na"}


def merge_match_results(
    results: Iterable[pd.DataFrame],
    top_k: int = 5,
    query_col: str = "query",
    carry_cols: Iterable[str] = ("ref_match",),
) -> pd.DataFrame:
    """Merge per-ontology OntoMapEngine result frames into one ranked top-k.

    For each query, pools all ``matchN`` candidates across the frames, keeps the
    highest ``matchN_score`` per distinct match label, ranks by score descending,
    and re-emits ``match1..top_k`` / ``match1_score..top_k_score``. Query order
    follows first appearance. ``carry_cols`` (e.g. ``ref_match``) are copied from
    the first frame that has a non-null value for the query.

    Sentinel non-matches ("Not Found", blank, NaN) are dropped before ranking.
    """
    frames = [f for f in results if f is not None and len(f) > 0]
    if not frames:
        return pd.DataFrame()

    order: list = []
    seen: set = set()
    pooled: dict = {}   # query -> {label: best_score}
    carried: dict = {}  # query -> {carry_col: value}

    for frame in frames:
        match_cols = [c for c in frame.columns if _MATCH_COL_RE.match(c)]
        for _, row in frame.iterrows():
            q = row[query_col]
            if q not in seen:
                seen.add(q)
                order.append(q)
            bucket = pooled.setdefault(q, {})
            for mcol in match_cols:
                label = row.get(mcol)
                if label is None or (isinstance(label, float) and pd.isna(label)):
                    continue
                label = str(label).strip()
                if not label or label.lower() in _MATCH_SENTINELS:
                    continue
                try:
                    score = float(row.get(f"{mcol}_score"))
                except (TypeError, ValueError):
                    score = 0.0
                if label not in bucket or score > bucket[label]:
                    bucket[label] = score
            cbucket = carried.setdefault(q, {})
            for col in carry_cols:
                if col in frame.columns and col not in cbucket:
                    val = row.get(col)
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        cbucket[col] = val

    rows = []
    for q in order:
        ranked = sorted(pooled[q].items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        rec: dict = {query_col: q}
        rec.update(carried.get(q, {}))
        for i in range(1, top_k + 1):
            if i <= len(ranked):
                rec[f"match{i}"] = ranked[i - 1][0]
                rec[f"match{i}_score"] = round(ranked[i - 1][1], 4)
            else:
                rec[f"match{i}"] = None
                rec[f"match{i}_score"] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def run_ontology_partitioned(
    records: list[dict],
    corpus_category: str,
    query_ls: list[str],
    *,
    top_k: int = 5,
    ground_truth_map: Optional[dict] = None,
    engine_kwargs: Optional[dict] = None,
    return_partials: bool = False,
):
    """Map ``query_ls`` against a (possibly mixed) combined corpus.

    Partitions ``records`` by ontology, runs one :class:`OntoMapEngine` per
    ontology with that single-ontology corpus (satisfying the engine's
    single-prefix requirement), then merges the per-ontology results into one
    ranked top-k per query via :func:`merge_match_results`.

    Parameters
    ----------
    records : list[dict]
        Combined corpus records from :meth:`CombinedCorpusBuilder.build`.
    corpus_category : str
        Semantic category passed to the engine (e.g. ``"disease"``).
    query_ls : list[str]
        Terms to map.
    top_k : int
        Candidates per query in the merged result.
    ground_truth_map : dict, optional
        Query -> curated label. Supplying it runs the engine in test mode;
        omitting it runs prod mode.
    engine_kwargs : dict, optional
        Extra keyword arguments forwarded to every ``OntoMapEngine`` (e.g.
        ``s2_method``, ``s3_strategy``, ``skip_stage25``).
    return_partials : bool
        When True, also return the per-ontology result frames.

    Returns
    -------
    pandas.DataFrame, or (DataFrame, dict[str, DataFrame])
        The merged result, and optionally the ``{ontology: frame}`` partials.
    """
    from metaharmonizer.engine.ontology_mapping_engine import OntoMapEngine

    groups = CombinedCorpusBuilder.partition_by_ontology(records)
    partials: dict[str, pd.DataFrame] = {}
    for ontology, recs in groups.items():
        corpus_df = CombinedCorpusBuilder.to_dataframe(recs)
        engine = OntoMapEngine(
            corpus_category=corpus_category,
            query_ls=query_ls,
            top_k=top_k,
            corpus_df=corpus_df,
            ontology_source=ontology,
            ground_truth_map=ground_truth_map,
            **(engine_kwargs or {}),
        )
        partials[ontology] = engine.run()

    merged = merge_match_results(partials.values(), top_k=top_k)
    return (merged, partials) if return_partials else merged
