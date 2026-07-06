"""Export/import the ontology knowledge DB as a single portable archive (#80).

Building the knowledge DB (SQLite corpus + FAISS indexes) needs a ``UMLS_API_KEY``
and is slow; independently-built indexes can also drift (GPU float
non-determinism). "Build once, distribute the artifact" avoids both — a KB
built on one machine can seed another (e.g. the ``MetaHarmonizer_App``
``engine_cache`` volume) without a rebuild.

Layout exported (from :data:`metaharmonizer._paths.KNOWLEDGE_DB_DIR`, default
``~/.metaharmonizer/KnowledgeDb``)::

    vector_db.sqlite                 # consistent snapshot via VACUUM INTO
    faiss_indexes/<...>.index        # one per (strategy, method, source, category)
    faiss_indexes/<...>.index.ids.npy
    manifest.json                    # versions + per-index metadata + sha256

Usage::

    python -m metaharmonizer.scripts.knowledge_db export -o kb.mhkb.tar.gz
    python -m metaharmonizer.scripts.knowledge_db export -o kb.mhkb.tar.gz --category disease
    python -m metaharmonizer.scripts.knowledge_db import kb.mhkb.tar.gz [--force]

Correctness notes:
- The SQLite snapshot uses ``VACUUM INTO`` (never a raw copy) because the live
  DB runs in WAL mode and a file copy can capture a torn state.
- ``import`` verifies every file's sha256 against the manifest and hard-fails on
  mismatch, so the engine never silently serves wrong vectors. It also compares
  ``kb_format_version`` (hard-fail) and ``package_version`` (warn only), and
  extracts atomically (temp dir → swap) so a failed import can't corrupt an
  existing KB. Archive members are path-traversal-checked before extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Bumped only when the on-disk KB format or the embedding contract changes in a
# way that makes an older archive unsafe to load — NOT on every package release.
KB_FORMAT_VERSION = 1

_MANIFEST_NAME = "manifest.json"
_SQLITE_NAME = "vector_db.sqlite"
_FAISS_SUBDIR = "faiss_indexes"


# --------------------------------------------------------------------------- #
# Path resolution (honours env overrides at call time, so tests can redirect)  #
# --------------------------------------------------------------------------- #
def _kb_dir() -> Path:
    from metaharmonizer import _paths

    return Path(os.environ.get("KNOWLEDGE_DB_DIR", _paths.KNOWLEDGE_DB_DIR))


def _sqlite_path(kb: Path) -> Path:
    return kb / _SQLITE_NAME


def _faiss_dir(kb: Path) -> Path:
    return kb / _FAISS_SUBDIR


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("metaharmonizer")
    except Exception:  # noqa: BLE001
        return "0.0.0+unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_index_name(stem: str) -> dict[str, str]:
    """Parse ``{strategy}_{method}_{ontology_source}_{category}`` from a FAISS
    index filename stem. Best-effort: unrecognised shapes still round-trip, they
    just carry a coarser ``raw`` label (custom-corpus hash suffixes fall here)."""
    parts = stem.split("_")
    if len(parts) >= 4:
        strategy, method, source = parts[0], parts[1], parts[2]
        category = "_".join(parts[3:])
        return {
            "strategy": strategy,
            "method": method,
            "ontology_source": source,
            "category": category,
        }
    return {"raw": stem}


# --------------------------------------------------------------------------- #
# Export                                                                        #
# --------------------------------------------------------------------------- #
def _snapshot_sqlite(src: Path, dst: Path) -> None:
    """Write a consistent snapshot of the (possibly WAL-mode, possibly open)
    SQLite DB using ``VACUUM INTO`` — never a raw file copy."""
    conn = sqlite3.connect(str(src))
    try:
        # VACUUM INTO refuses to overwrite an existing file.
        if dst.exists():
            dst.unlink()
        conn.execute("VACUUM INTO ?", (str(dst),))
    finally:
        conn.close()


def _collect_indexes(faiss_dir: Path, category: str | None) -> list[Path]:
    if not faiss_dir.is_dir():
        return []
    out: list[Path] = []
    for idx in sorted(faiss_dir.glob("*.index")):
        meta = _parse_index_name(idx.stem)
        if category and meta.get("category") != category:
            continue
        out.append(idx)
        ids = idx.with_suffix(".index.ids.npy")
        if ids.exists():
            out.append(ids)
    return out


def cmd_export(args: argparse.Namespace) -> int:
    kb = _kb_dir()
    sqlite_src = _sqlite_path(kb)
    if not sqlite_src.exists():
        print(f"error: no knowledge DB found at {sqlite_src}", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        # 1. Consistent SQLite snapshot.
        snap = stage / _SQLITE_NAME
        _snapshot_sqlite(sqlite_src, snap)

        # 2. FAISS indexes (+ ids), optionally filtered by category.
        faiss_out = stage / _FAISS_SUBDIR
        faiss_out.mkdir(parents=True, exist_ok=True)
        indexes = _collect_indexes(_faiss_dir(kb), args.category)
        for f in indexes:
            shutil.copy2(f, faiss_out / f.name)

        # 3. Manifest: versions + per-index metadata + sha256 of every file.
        files_meta: list[dict] = []
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                rel = f.relative_to(stage).as_posix()
                entry = {"path": rel, "sha256": _sha256(f), "bytes": f.stat().st_size}
                if rel.endswith(".index"):
                    entry.update(_parse_index_name(Path(rel).stem))
                files_meta.append(entry)

        manifest = {
            "kb_format_version": KB_FORMAT_VERSION,
            "package_version": _package_version(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "category_filter": args.category,
            "files": files_meta,
        }
        (stage / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

        # 4. Archive (manifest first so it can be read without full extraction).
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(stage / _MANIFEST_NAME, arcname=_MANIFEST_NAME)
            for f in sorted(stage.rglob("*")):
                if f.is_file() and f.name != _MANIFEST_NAME:
                    tar.add(f, arcname=f.relative_to(stage).as_posix())

    n_idx = sum(1 for m in files_meta if m["path"].endswith(".index"))
    print(f"exported {n_idx} index(es) + corpus -> {out_path} ({out_path.stat().st_size} bytes)")
    return 0


# --------------------------------------------------------------------------- #
# Import                                                                        #
# --------------------------------------------------------------------------- #
def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract with a path-traversal guard: reject absolute paths, ``..`` escapes,
    and non-regular members."""
    dest = dest.resolve()
    for member in tar.getmembers():
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"refusing unusual archive member: {member.name}")
        target = (dest / member.name).resolve()
        if os.path.commonpath([str(dest), str(target)]) != str(dest):
            raise ValueError(f"path traversal blocked: {member.name}")
    # ``filter="data"`` (py>=3.12; backported to 3.10.12/3.11.4) is a second line
    # of defence on top of the explicit checks above. Fall back cleanly on older
    # runtimes where the argument is unavailable.
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)  # noqa: S202 — members validated above


def cmd_import(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    if not archive.exists():
        print(f"error: archive not found: {archive}", file=sys.stderr)
        return 2

    kb = _kb_dir()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, stage)

        manifest_path = stage / _MANIFEST_NAME
        if not manifest_path.exists():
            print("error: archive has no manifest.json", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text())

        # Compatibility: hard-fail on KB-format drift, warn on package drift.
        fmt = manifest.get("kb_format_version")
        if fmt != KB_FORMAT_VERSION:
            print(
                f"error: KB format version mismatch (archive {fmt} != engine "
                f"{KB_FORMAT_VERSION}); refusing to import.",
                file=sys.stderr,
            )
            return 3
        pkg = manifest.get("package_version")
        if pkg and pkg != _package_version():
            print(
                f"warning: archive built with metaharmonizer {pkg}, running "
                f"{_package_version()} — importing anyway (same KB format).",
                file=sys.stderr,
            )

        # Verify every file's checksum before touching the live KB.
        for entry in manifest.get("files", []):
            f = stage / entry["path"]
            if not f.exists():
                print(f"error: manifest lists missing file: {entry['path']}", file=sys.stderr)
                return 3
            actual = _sha256(f)
            if actual != entry["sha256"]:
                print(
                    f"error: checksum mismatch for {entry['path']} — corrupt archive; "
                    "aborting so the engine never serves wrong vectors.",
                    file=sys.stderr,
                )
                return 3

        # Refuse to clobber an existing KB unless --force.
        target_sqlite = _sqlite_path(kb)
        if target_sqlite.exists() and not args.force:
            print(
                f"error: a knowledge DB already exists at {kb}; pass --force to replace it.",
                file=sys.stderr,
            )
            return 4

        # Atomic-ish install: build the new KB tree in a sibling temp dir, then
        # swap directories so a crash mid-copy can't leave a half KB.
        kb.parent.mkdir(parents=True, exist_ok=True)
        staged_kb = Path(tempfile.mkdtemp(prefix=".kb_new_", dir=str(kb.parent)))
        try:
            shutil.copy2(stage / _SQLITE_NAME, staged_kb / _SQLITE_NAME)
            src_faiss = stage / _FAISS_SUBDIR
            if src_faiss.is_dir():
                shutil.copytree(src_faiss, staged_kb / _FAISS_SUBDIR)
            shutil.copy2(manifest_path, staged_kb / _MANIFEST_NAME)

            backup = None
            if kb.exists():
                backup = kb.parent / (kb.name + ".old")
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(kb, backup)
            os.replace(staged_kb, kb)
            if backup and backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            shutil.rmtree(staged_kb, ignore_errors=True)
            raise

    print(f"imported knowledge DB -> {kb}")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m metaharmonizer.scripts.knowledge_db",
        description="Export/import the ontology knowledge DB as a portable archive.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Bundle the knowledge DB into an archive.")
    exp.add_argument("-o", "--output", required=True, help="Output .mhkb.tar.gz path.")
    exp.add_argument("--category", default=None, help="Only export indexes for this category.")
    exp.set_defaults(func=cmd_export)

    imp = sub.add_parser("import", help="Install a knowledge-DB archive.")
    imp.add_argument("archive", help="Path to a .mhkb.tar.gz archive.")
    imp.add_argument("--force", action="store_true", help="Replace an existing KB.")
    imp.set_defaults(func=cmd_import)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
