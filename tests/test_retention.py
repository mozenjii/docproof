"""Tests for opaque ids and the retention sweep, including what it must never delete."""

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pytest

from docproof.retention import (
    META,
    RetentionPolicy,
    is_document_id,
    is_expired,
    new_document_id,
    store_document,
    sweep,
)

UTC = dt.UTC
T0 = dt.datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
POLICY = RetentionPolicy(dt.timedelta(days=30))


def test_ids_are_random_and_well_formed() -> None:
    ids = {new_document_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(is_document_id(i) for i in ids)
    for bad in ("doc_", "doc_XYZ", "invoice.pdf", "../doc_" + "0" * 24, "doc_" + "0" * 25):
        assert not is_document_id(bad)


def test_expiry_boundary_is_strict() -> None:
    assert not is_expired(T0, T0 + dt.timedelta(days=30), POLICY)
    assert is_expired(T0, T0 + dt.timedelta(days=30, seconds=1), POLICY)


def test_timezones_are_compared_as_instants() -> None:
    karachi = dt.timezone(dt.timedelta(hours=5))
    stored = dt.datetime(2026, 1, 1, 17, 0, tzinfo=karachi)  # the same instant as T0
    assert not is_expired(stored, T0 + dt.timedelta(days=30), POLICY)


def test_naive_times_are_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        is_expired(dt.datetime(2026, 1, 1), T0, POLICY)  # naive on purpose
    with pytest.raises(ValueError, match="timezone-aware"):
        is_expired(T0, dt.datetime(2026, 1, 1), POLICY)


@pytest.mark.parametrize("days", [0, -1])
def test_policy_must_be_positive(days: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        RetentionPolicy(dt.timedelta(days=days))


def test_store_writes_content_and_meta_under_an_opaque_id(tmp_path: Path) -> None:
    doc_id = store_document(tmp_path, "Jane Roe invoice.pdf", b"%PDF-1.7", T0)
    folder = tmp_path / doc_id
    assert is_document_id(doc_id)
    assert "Jane" not in doc_id
    assert (folder / "Jane Roe invoice.pdf").read_bytes() == b"%PDF-1.7"
    assert json.loads((folder / META).read_text())["stored_at"] == T0.isoformat()


@pytest.mark.parametrize("name", ["../../escape.pdf", "C:/Windows/x.pdf", "a/b/c.pdf"])
def test_store_keeps_only_the_file_name(tmp_path: Path, name: str) -> None:
    store = tmp_path / "store"
    doc_id = store_document(store, name, b"x", T0)
    assert {p.name for p in (store / doc_id).iterdir()} == {Path(name).name, META}
    assert not (tmp_path / "escape.pdf").exists()


@pytest.mark.parametrize("name", ["", ".", "..", META])
def test_store_refuses_unusable_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="unusable"):
        store_document(tmp_path, name, b"x", T0)


def test_store_refuses_naive_time(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        store_document(tmp_path, "a.pdf", b"x", dt.datetime(2026, 1, 1))


def test_sweep_deletes_old_keeps_new_and_never_guesses(tmp_path: Path) -> None:
    old = store_document(tmp_path, "a.pdf", b"a", T0)
    new = store_document(tmp_path, "b.pdf", b"b", T0 + dt.timedelta(days=20))
    broken = store_document(tmp_path, "c.pdf", b"c", T0)
    (tmp_path / broken / META).write_text("{not json", encoding="utf-8")
    naive = store_document(tmp_path, "d.pdf", b"d", T0)
    (tmp_path / naive / META).write_text(json.dumps({"stored_at": "2026-01-01T12:00:00"}))
    missing = store_document(tmp_path, "e.pdf", b"e", T0)
    (tmp_path / missing / META).unlink()
    (tmp_path / "notes.txt").write_text("not a document folder")
    (tmp_path / "doc_not_hex_at_all_xxxxxxxx").mkdir()

    now = T0 + dt.timedelta(days=45)
    preview = sweep(tmp_path, POLICY, now, dry_run=True)
    assert preview.deleted == [old]
    assert (tmp_path / old).exists()

    report = sweep(tmp_path, POLICY, now)
    assert report.deleted == [old]
    assert report.kept == [new]
    assert sorted(report.unreadable) == sorted([broken, naive, missing])
    assert sorted(report.foreign) == sorted(["notes.txt", "doc_not_hex_at_all_xxxxxxxx"])
    assert not (tmp_path / old).exists()
    for survivor in (new, broken, naive, missing):
        assert (tmp_path / survivor).exists()


def test_sweep_does_not_follow_symlinks(tmp_path: Path) -> None:
    store, outside = tmp_path / "store", tmp_path / "outside"
    store.mkdir()
    outside.mkdir()
    victim = outside / "keep.txt"
    victim.write_text("must survive")
    old = store_document(store, "a.pdf", b"a", T0)
    linked_id = new_document_id()
    try:
        os.symlink(outside, store / linked_id, target_is_directory=True)
        os.symlink(victim, store / old / "link.txt")
    except OSError:
        pytest.skip("creating symlinks needs developer mode on this Windows machine")
    report = sweep(store, POLICY, T0 + dt.timedelta(days=45))
    assert linked_id in report.foreign
    assert report.deleted == [old]
    assert victim.read_text() == "must survive"


def test_missing_store_is_an_error_not_an_empty_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sweep(tmp_path / "nope", POLICY, T0)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions exist only on Windows")
def test_sweep_does_not_follow_windows_junctions(tmp_path: Path) -> None:
    # Junctions need no special rights, and is_symlink() is False for them on Python 3.12.
    import _winapi

    store, outside = tmp_path / "store", tmp_path / "outside"
    store.mkdir()
    outside.mkdir()
    (outside / META).write_text(json.dumps({"stored_at": T0.isoformat()}), encoding="utf-8")
    (outside / "keep.txt").write_text("must survive")
    linked_id = new_document_id()
    _winapi.CreateJunction(str(outside), str(store / linked_id))
    report = sweep(store, POLICY, T0 + dt.timedelta(days=45))
    assert linked_id in report.foreign
    assert report.deleted == []
    assert (outside / "keep.txt").read_text() == "must survive"
