"""Offline corpus builder for OLS-hosted ontologies (UBERON, MONDO, HP, etc.).

Fetches all descendants of a root term from EBI OLS4, saves them to JSON,
then populates the synonym and RAG SQLite tables so the pipeline can run
fully offline.

Usage::

    python scripts/build_ols_corpus.py --term UBERON:0001062 --category bodysite --ontology uberon
    python scripts/build_ols_corpus.py --term MONDO:0000001 --category disease --ontology mondo --force-rebuild
    python scripts/build_ols_corpus.py --term UBERON:0001062 --category bodysite --ontology uberon --include-hierarchy
"""

import argparse
import sys

from src.KnowledgeDb.corpus_builder import CorpusBuilder
from src.KnowledgeDb.concept_table_builder import ConceptTableBuilder
from src._paths import corpus_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an OLS corpus and populate pipeline concept tables."
    )
    parser.add_argument(
        "--term",
        required=True,
        help="Root OBO term ID, e.g. UBERON:0001062 or MONDO:0000001",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Pipeline category name: disease, bodysite, or treatment",
    )
    parser.add_argument(
        "--ontology",
        required=True,
        help="Ontology source name, e.g. uberon, mondo (used for table/index naming)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Clear existing tables and re-fetch all terms",
    )
    parser.add_argument(
        "--no-root",
        action="store_true",
        help="Exclude the root term itself from the corpus",
    )
    parser.add_argument(
        "--include-hierarchy",
        action="store_true",
        help=(
            "Fetch parent and child labels for each term and include them in the "
            "RAG context string.  Produces richer context at the cost of ~2x API calls."
        ),
    )
    args = parser.parse_args()
    args.ontology = args.ontology.lower()
    args.category = args.category.lower()

    # ── Step 1: fetch corpus from OLS ────────────────────────────────────────
    print(f"\nBuilding corpus for {args.term} (category={args.category}) ...")
    if args.include_hierarchy:
        print("  [hierarchy mode] fetching parent/child labels for each term")
    builder = CorpusBuilder()
    records = builder.build_sync(
        root_term_id=args.term,
        include_root=not args.no_root,
        include_hierarchy=args.include_hierarchy,
    )

    if not records:
        sys.exit("ERROR: No terms returned from OLS. Check the term ID and network.")

    # ── Step 2: save corpus JSON ──────────────────────────────────────────────
    corpus_key = f"{args.ontology}_{args.category}"
    json_path = corpus_path(args.category, args.ontology, ".json")
    builder.save(records, str(json_path), root_term_id=args.term)
    print(f"Corpus saved to: {json_path}  ({len(records)} terms)")

    # ── Step 3: populate SQLite synonym + RAG tables ──────────────────────────
    print(f"\nPopulating concept tables for '{args.ontology}_{args.category}' ...")
    concept_builder = ConceptTableBuilder(args.category, args.ontology)
    concept_builder.build_from_json(str(json_path), force_rebuild=args.force_rebuild)

    stats = concept_builder.get_stats()
    print(
        f"\nDone.\n"
        f"  RAG records    : {stats['rag_records']}\n"
        f"  Synonym records: {stats['synonym_records']}\n"
        f"  Unique codes   : {stats['unique_codes']}\n"
    )
    print(
        f"The pipeline can now be run with category='{args.category}', "
        f"ontology_source='{args.ontology}' without any external API calls."
    )


if __name__ == "__main__":
    main()
