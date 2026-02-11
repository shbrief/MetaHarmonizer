#!/usr/bin/env python
"""Build an ontology descendant corpus as JSON.

Examples::

    python scripts/build_corpus.py NCIT:C3262 -o data/corpus/neoplasm.json
    python scripts/build_corpus.py MONDO:0005070 -o data/corpus/mondo_disease.json
    python scripts/build_corpus.py NCIT:C3262 --no-root -o output.json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.KnowledgeDb.corpus_builder import CorpusBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build ontology descendant corpus via EBI OLS API"
    )
    parser.add_argument(
        "term_id", help="Root term OBO ID (e.g. NCIT:C3262, MONDO:0000001)"
    )
    parser.add_argument(
        "--ontology",
        help="Ontology short name (auto-detected from prefix if omitted)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output JSON file path"
    )
    parser.add_argument(
        "--no-root",
        action="store_true",
        help="Exclude the root term from the output",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="OLS API page size (default: 200)",
    )

    args = parser.parse_args()

    builder = CorpusBuilder(page_size=args.page_size)
    records = builder.build_sync(
        root_term_id=args.term_id,
        ontology=args.ontology,
        include_root=not args.no_root,
    )

    path = builder.save(
        records,
        args.output,
        root_term_id=args.term_id,
        ontology=args.ontology or "",
    )
    print(f"Wrote {len(records)} terms to {path}")


if __name__ == "__main__":
    main()
