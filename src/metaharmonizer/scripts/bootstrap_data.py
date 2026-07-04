"""Populate the user data cache with large data files shipped in the repo.

Large corpora (e.g. the merged EFO corpus) are intentionally *not* bundled
inside the wheel — they live in the repo's ``data/`` tree and are resolved at
runtime from :data:`metaharmonizer._paths.DATA_DIR`
(default ``~/.metaharmonizer/data``, overridable via ``METAHARMONIZER_DATA_DIR``).

This script copies those files from the repo into the active ``DATA_DIR`` so
the engines can auto-resolve them. It is idempotent and skips files that are
already present (use ``--force`` to overwrite). When ``DATA_DIR`` already points
at the repo ``data/`` dir (source == destination) there is nothing to do.

Usage::

    python -m metaharmonizer.scripts.bootstrap_data            # copy missing
    python -m metaharmonizer.scripts.bootstrap_data --force    # overwrite
    python -m metaharmonizer.scripts.bootstrap_data --source /path/to/repo/data
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from metaharmonizer._paths import DATA_DIR

# Files to mirror from the repo data tree into DATA_DIR, as paths relative to
# the data root. Add new shipped-but-unbundled data files here.
_MANIFEST: tuple[str, ...] = (
    "corpus/retrieved_ontologies/efo_phenotype_corpus.csv",
)


def _default_source() -> Path:
    """Repo ``data/`` dir, resolved relative to this file in an editable checkout.

    ``<repo>/src/metaharmonizer/scripts/bootstrap_data.py`` → ``<repo>/data``.
    """
    return Path(__file__).resolve().parents[3] / "data"


def bootstrap(source: Path, dest: Path, *, force: bool = False) -> int:
    """Copy manifest files from ``source`` to ``dest``. Returns files copied."""
    source = source.expanduser().resolve()
    dest = dest.expanduser().resolve()
    if source == dest:
        print(f"[bootstrap] DATA_DIR already points at the source ({dest}); nothing to do.")
        return 0

    copied = 0
    for rel in _MANIFEST:
        src_file = source / rel
        dst_file = dest / rel
        if not src_file.exists():
            print(f"[bootstrap] WARNING: source missing, skipping: {src_file}", file=sys.stderr)
            continue
        if dst_file.exists() and not force:
            print(f"[bootstrap] exists, skipping (use --force): {dst_file}")
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        print(f"[bootstrap] copied {rel} -> {dst_file}")
        copied += 1
    print(f"[bootstrap] done: {copied} file(s) copied into {dest}")
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=_default_source(),
                        help="Repo data/ directory to copy from (default: auto-detected).")
    parser.add_argument("--dest", type=Path, default=DATA_DIR,
                        help="Target DATA_DIR (default: metaharmonizer._paths.DATA_DIR).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite files that already exist in the destination.")
    args = parser.parse_args(argv)
    bootstrap(args.source, args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
