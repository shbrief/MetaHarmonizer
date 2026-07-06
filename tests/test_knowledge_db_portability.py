"""Round-trip + safety tests for the knowledge-DB export/import CLI (#80).

These use a synthetic KNOWLEDGE_DB_DIR (a tiny real SQLite file + fake FAISS
index blobs), so no UMLS_API_KEY or built corpus is required.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tarfile
from pathlib import Path

import pytest

from metaharmonizer.scripts import knowledge_db as kb


def _make_kb(root: Path) -> Path:
    """Create a minimal fake knowledge DB under ``root`` and return its dir."""
    kbdir = root / "KnowledgeDb"
    (kbdir / "faiss_indexes").mkdir(parents=True)
    # A real (tiny) SQLite DB so VACUUM INTO works.
    conn = sqlite3.connect(str(kbdir / "vector_db.sqlite"))
    conn.execute("CREATE TABLE concept (id INTEGER, term TEXT)")
    conn.execute("INSERT INTO concept VALUES (1, 'lung')")
    conn.commit()
    conn.close()
    # Fake FAISS index pair. The name matches what the engine actually writes:
    # faiss_sqlite_pipeline builds `{strategy}_{method}_{source}_{category}` and
    # cleans the method with `.replace('-', '_')`, so `sap-bert` lands on disk as
    # `sap_bert` (an underscore *inside* the method token).
    idx = kbdir / "faiss_indexes" / "rag_sap_bert_ncit_disease.index"
    idx.write_bytes(b"FAKE-FAISS-INDEX-BYTES")
    idx.with_suffix(".index.ids.npy").write_bytes(b"FAKE-IDS")
    return kbdir


@pytest.fixture
def kb_env(tmp_path, monkeypatch):
    kbdir = _make_kb(tmp_path / "src")
    monkeypatch.setenv("KNOWLEDGE_DB_DIR", str(kbdir))
    return kbdir


def test_export_then_import_roundtrip(tmp_path, kb_env, monkeypatch):
    archive = tmp_path / "kb.mhkb.tar.gz"
    assert kb.main(["export", "-o", str(archive)]) == 0
    assert archive.exists()

    # Import into a fresh, empty KNOWLEDGE_DB_DIR.
    dest = tmp_path / "dest" / "KnowledgeDb"
    monkeypatch.setenv("KNOWLEDGE_DB_DIR", str(dest))
    assert kb.main(["import", str(archive)]) == 0

    assert (dest / "vector_db.sqlite").exists()
    assert (dest / "faiss_indexes" / "rag_sap_bert_ncit_disease.index").exists()
    assert (dest / "faiss_indexes" / "rag_sap_bert_ncit_disease.index.ids.npy").exists()
    # The restored SQLite is queryable.
    conn = sqlite3.connect(str(dest / "vector_db.sqlite"))
    assert conn.execute("SELECT term FROM concept WHERE id=1").fetchone()[0] == "lung"
    conn.close()


def test_manifest_records_index_metadata(tmp_path, kb_env):
    archive = tmp_path / "kb.mhkb.tar.gz"
    kb.main(["export", "-o", str(archive)])
    with tarfile.open(archive) as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read())
    assert manifest["kb_format_version"] == kb.KB_FORMAT_VERSION
    idx = next(f for f in manifest["files"] if f["path"].endswith(".index"))
    assert idx["ontology_source"] == "ncit"
    assert idx["category"] == "disease"
    assert idx["strategy"] == "rag"
    assert idx["method"] == "sap_bert"


def test_category_filter_excludes_other_indexes(tmp_path, kb_env):
    # Add a second-category index.
    (kb_env / "faiss_indexes" / "rag_sap_bert_uberon_bodysite.index").write_bytes(b"X")
    archive = tmp_path / "kb.mhkb.tar.gz"
    kb.main(["export", "-o", str(archive), "--category", "disease"])
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert any("ncit_disease.index" in n for n in names)
    assert not any("uberon_bodysite.index" in n for n in names)


def test_import_detects_checksum_mismatch(tmp_path, kb_env, monkeypatch, capsys):
    archive = tmp_path / "kb.mhkb.tar.gz"
    kb.main(["export", "-o", str(archive)])

    # Tamper: rewrite the sqlite entry inside the archive, keep the old manifest.
    tampered = tmp_path / "bad.mhkb.tar.gz"
    with tarfile.open(archive) as src, tarfile.open(tampered, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read() if m.isfile() else b""
            if m.name == "vector_db.sqlite":
                data = data + b"CORRUPT"
                m.size = len(data)
            import io

            dst.addfile(m, io.BytesIO(data))

    dest = tmp_path / "dest" / "KnowledgeDb"
    monkeypatch.setenv("KNOWLEDGE_DB_DIR", str(dest))
    rc = kb.main(["import", str(tampered)])
    assert rc == 3
    assert "checksum mismatch" in capsys.readouterr().err


def test_import_blocks_path_traversal(tmp_path, monkeypatch, capsys):
    # Hand-craft a malicious archive with a '..' member.
    evil = tmp_path / "evil.mhkb.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("pwned")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../escape.txt")
        # minimal manifest so we reach extraction first
    dest = tmp_path / "dest" / "KnowledgeDb"
    monkeypatch.setenv("KNOWLEDGE_DB_DIR", str(dest))
    with pytest.raises(ValueError, match="path traversal"):
        kb.main(["import", str(evil)])


def test_import_refuses_existing_without_force(tmp_path, kb_env, monkeypatch, capsys):
    archive = tmp_path / "kb.mhkb.tar.gz"
    kb.main(["export", "-o", str(archive)])
    dest = _make_kb(tmp_path / "dest2")  # a KB already exists here
    monkeypatch.setenv("KNOWLEDGE_DB_DIR", str(dest))
    assert kb.main(["import", str(archive)]) == 4
    assert "already exists" in capsys.readouterr().err
    # With --force it succeeds.
    assert kb.main(["import", str(archive), "--force"]) == 0
