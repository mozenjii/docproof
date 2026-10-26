"""Tests for opaque ids and the retention sweep, including what it must never delete."""

import datetime as dt
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from docproof.retention import (
    CONTENT,
    DELETING,
    INCOMING,
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
LATER = T0 + dt.timedelta(days=45)
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


def test_naive_times_are_refused(tmp_path: Path) -> None:
    naive = dt.datetime(2026, 1, 1)  # naive on purpose
    with pytest.raises(ValueError, match="timezone-aware"):
        is_expired(naive, T0, POLICY)
    with pytest.raises(ValueError, match="timezone-aware"):
        is_expired(T0, naive, POLICY)
    with pytest.raises(ValueError, match="timezone-aware"):
        store_document(tmp_path, b"x", naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        sweep(tmp_path, POLICY, naive)


@pytest.mark.parametrize("days", [0, -1])
def test_policy_must_be_positive(days: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        RetentionPolicy(dt.timedelta(days=days))


def test_store_keeps_content_and_time_but_not_the_file_name(tmp_path: Path) -> None:
    doc_id = store_document(tmp_path, b"%PDF-1.7", T0, suffix=".pdf")
    folder = tmp_path / doc_id
    assert is_document_id(doc_id)
    assert sorted(p.name for p in folder.iterdir()) == [f"{CONTENT}.pdf", META]
    assert (folder / f"{CONTENT}.pdf").read_bytes() == b"%PDF-1.7"
    assert json.loads((folder / META).read_text())["stored_at"] == T0.isoformat()
    assert not any(p.name.startswith(INCOMING) for p in tmp_path.iterdir())


@pytest.mark.parametrize("suffix", ["pdf", ".PDF", ".toolong", "./x", ".p df", "::$DATA", ". "])
def test_store_refuses_malformed_suffixes(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(ValueError, match="suffix"):
        store_document(tmp_path, b"x", T0, suffix=suffix)


def test_sweep_deletes_old_keeps_new_and_never_guesses(tmp_path: Path) -> None:
    old = store_document(tmp_path, b"a", T0)
    new = store_document(tmp_path, b"b", T0 + dt.timedelta(days=20))
    broken = store_document(tmp_path, b"c", T0)
    (tmp_path / broken / META).write_text("{not json", encoding="utf-8")
    naive = store_document(tmp_path, b"d", T0)
    (tmp_path / naive / META).write_text(json.dumps({"stored_at": "2026-01-01T12:00:00"}))
    missing = store_document(tmp_path, b"e", T0)
    (tmp_path / missing / META).unlink()
    future = store_document(tmp_path, b"f", T0)
    (tmp_path / future / META).write_text(json.dumps({"stored_at": "2099-01-01T00:00:00+00:00"}))
    (tmp_path / "notes.txt").write_text("not a document folder")
    (tmp_path / "doc_not_hex_at_all_xxxxxxxx").mkdir()

    preview = sweep(tmp_path, POLICY, LATER, dry_run=True)
    assert preview.deleted == [old]
    assert (tmp_path / old).exists()

    report = sweep(tmp_path, POLICY, LATER)
    assert report.deleted == [old]
    assert report.kept == [new]
    assert report.unreadable == sorted([broken, naive, missing, future])
    assert report.foreign == sorted(["notes.txt", "doc_not_hex_at_all_xxxxxxxx"])
    assert report.failed == []
    assert not (tmp_path / old).exists()
    for survivor in (new, broken, naive, missing, future):
        assert (tmp_path / survivor).exists()


def test_an_interrupted_store_can_still_be_dated_and_swept(tmp_path: Path) -> None:
    doc_id = store_document(tmp_path, b"a", T0)
    # Simulate a crash before the final rename: the staging directory holds meta and content.
    (tmp_path / doc_id).rename(tmp_path / f"{INCOMING}{doc_id}")
    assert sweep(tmp_path, POLICY, T0 + dt.timedelta(days=1)).kept == [doc_id]
    assert sweep(tmp_path, POLICY, LATER).deleted == [doc_id]
    assert list(tmp_path.iterdir()) == []


def test_a_half_finished_delete_is_finished_next_time(tmp_path: Path) -> None:
    doc_id = store_document(tmp_path, b"a", T0)
    tombstone = tmp_path / f"{DELETING}{doc_id}"
    (tmp_path / doc_id).rename(tombstone)
    (tombstone / META).unlink()  # the part already removed before the interruption
    report = sweep(tmp_path, POLICY, T0)  # even within the retention period: it was decided
    assert report.deleted == [doc_id]
    assert not tombstone.exists()


def test_one_stuck_document_does_not_stop_the_sweep(tmp_path: Path) -> None:
    stuck = store_document(tmp_path, b"a", T0, suffix=".pdf")
    other = store_document(tmp_path, b"b", T0, suffix=".pdf")
    locked = tmp_path / stuck / f"{CONTENT}.pdf"
    os.chmod(locked, stat.S_IREAD)
    handle = locked.open("rb")  # an open file cannot be deleted on Windows
    try:
        report = sweep(tmp_path, POLICY, LATER)
    finally:
        handle.close()
        tombstone = tmp_path / f"{DELETING}{stuck}"
        for path in (tombstone / f"{CONTENT}.pdf", locked):
            if path.exists():
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    assert other in report.deleted
    if sys.platform == "win32":
        assert report.failed == [stuck]
        assert sweep(tmp_path, POLICY, LATER).deleted == [stuck]  # retried once released
    else:
        assert stuck in report.deleted  # POSIX lets an owner delete read-only open files


def test_sweep_does_not_follow_symlinks(tmp_path: Path) -> None:
    store, outside = tmp_path / "store", tmp_path / "outside"
    store.mkdir()
    outside.mkdir()
    victim = outside / "keep.txt"
    victim.write_text("must survive")
    old = store_document(store, b"a", T0)
    linked_id = new_document_id()
    try:
        os.symlink(outside, store / linked_id, target_is_directory=True)
        os.symlink(victim, store / old / "link.txt")
    except OSError:
        pytest.skip("creating symlinks needs developer mode on this Windows machine")
    report = sweep(store, POLICY, LATER)
    assert linked_id in report.foreign
    assert report.deleted == [old]
    assert victim.read_text() == "must survive"


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
    _winapi.CreateJunction(str(outside), str(store / f"{DELETING}{new_document_id()}"))
    report = sweep(store, POLICY, LATER)
    assert linked_id in report.foreign
    assert report.deleted == []
    assert (outside / "keep.txt").read_text() == "must survive"


def test_missing_store_is_an_error_not_an_empty_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sweep(tmp_path / "nope", POLICY, T0)


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    doc_id = store_document(tmp_path, b"a", T0)
    tomb = store_document(tmp_path, b"b", T0)
    (tmp_path / tomb).rename(tmp_path / f"{DELETING}{tomb}")
    before = sorted(p.name for p in tmp_path.iterdir())
    report = sweep(tmp_path, POLICY, LATER, dry_run=True)
    assert report.deleted == sorted([doc_id, tomb])
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    shutil.rmtree(tmp_path / f"{DELETING}{tomb}")


def test_a_document_that_cannot_be_renamed_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = store_document(tmp_path, b"a", T0)

    def refuse(self: Path, target: Path) -> Path:
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "rename", refuse)
    report = sweep(tmp_path, POLICY, LATER)
    assert report.failed == [doc_id]
    assert (tmp_path / doc_id).exists()


def test_a_failed_removal_leaves_a_tombstone_that_the_next_sweep_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = store_document(tmp_path, b"a", T0)

    def fail(path: Path) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr("docproof.retention.shutil.rmtree", fail)
    assert sweep(tmp_path, POLICY, LATER).failed == [doc_id]
    assert (tmp_path / f"{DELETING}{doc_id}").exists()
    monkeypatch.undo()
    assert sweep(tmp_path, POLICY, LATER).deleted == [doc_id]
    assert list(tmp_path.iterdir()) == []
