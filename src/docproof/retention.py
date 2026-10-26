"""Opaque document ids and a retention sweep for stored documents.

A stored document is a directory ``<store>/<doc_id>/`` holding ``meta.json`` (when it was stored)
and ``document<suffix>``. The id is random and the uploader's file name is not kept, because a name
like "Jane Roe invoice.pdf" is itself personal data. The sweep deletes every document older than
the policy allows.

Both directions are crash-safe. Storing writes ``meta.json`` first into ``.incoming-<id>/`` and
renames the finished directory into place, so a crash leaves a staging directory that can still be
dated. Deleting renames ``<id>`` to ``.deleting-<id>`` before removing it, so a delete that stops
halfway is finished by the next sweep instead of leaving a document that can no longer be dated.

Deliberate boundary: the retention period is a required argument with no default. How long a
document may be kept is a product and legal decision that has not been made, and a default here
would quietly become the answer. The sweep only touches directories it created, never follows a
symlink or junction, and never deletes something it cannot date; those are reported for a person.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_ID = re.compile(r"doc_[0-9a-f]{24}")
_SUFFIX = re.compile(r"\.[a-z0-9]{1,5}")
META = "meta.json"
CONTENT = "document"
INCOMING = ".incoming-"
DELETING = ".deleting-"


def new_document_id() -> str:
    """Return a fresh random document id such as ``doc_3f9a…`` (96 random bits)."""
    return f"doc_{secrets.token_hex(12)}"


def is_document_id(text: str) -> bool:
    """Return whether ``text`` has the exact shape of a document id."""
    return bool(_ID.fullmatch(text))


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How long stored documents are kept.

    Attributes:
        max_age: Documents stored longer ago than this are deleted. Must be positive.
    """

    max_age: dt.timedelta

    def __post_init__(self) -> None:
        """Reject a zero or negative period, which would delete everything at once."""
        if self.max_age <= dt.timedelta(0):
            raise ValueError("max_age must be positive")


def _aware(moment: dt.datetime, what: str) -> dt.datetime:
    """Reject naive datetimes: comparing local and UTC times silently shifts retention by hours."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{what} must be timezone-aware")
    return moment


def is_expired(stored_at: dt.datetime, now: dt.datetime, policy: RetentionPolicy) -> bool:
    """Decide whether a document has outlived the policy.

    Args:
        stored_at: When it was stored; timezone-aware.
        now: The current time; timezone-aware.
        policy: The retention policy.

    Returns:
        ``True`` once ``now - stored_at`` is strictly greater than ``policy.max_age``.

    Raises:
        ValueError: If either time is naive.
    """
    return _aware(now, "now") - _aware(stored_at, "stored_at") > policy.max_age


def store_document(store: Path, content: bytes, now: dt.datetime, *, suffix: str = "") -> str:
    """Store one document under a fresh opaque id.

    Args:
        store: The store directory; created if missing.
        content: The document bytes.
        now: Storage time; timezone-aware.
        suffix: File type such as ``".pdf"``: a dot and 1-5 lowercase letters or digits, or empty.
            The original file name is never stored.

    Returns:
        The new document id.

    Raises:
        ValueError: If ``now`` is naive or ``suffix`` is malformed.
    """
    stored_at = _aware(now, "now")
    if suffix and not _SUFFIX.fullmatch(suffix):
        raise ValueError("suffix must be a dot and 1-5 lowercase letters or digits")
    store.mkdir(parents=True, exist_ok=True)
    doc_id = new_document_id()
    staging = store / f"{INCOMING}{doc_id}"
    staging.mkdir()
    # The time goes first: a crash after this line leaves something the sweep can still date.
    (staging / META).write_text(json.dumps({"stored_at": stored_at.isoformat()}), encoding="utf-8")
    (staging / f"{CONTENT}{suffix}").write_bytes(content)
    staging.rename(store / doc_id)
    return doc_id


@dataclass(slots=True)
class SweepReport:
    """What a sweep did. Every list is sorted.

    Attributes:
        deleted: Ids of documents deleted, including half-finished deletes completed now.
        kept: Ids of documents still within the policy.
        unreadable: Ids whose storage time is missing, malformed, naive or in the future; kept
            and flagged for a person.
        failed: Ids whose deletion raised an error (a locked or read-only file, for example);
            the sweep carried on and the next one retries.
        foreign: Entries the store did not create (other names, files, symlinks, junctions);
            never touched.
    """

    deleted: list[str] = field(default_factory=list[str])
    kept: list[str] = field(default_factory=list[str])
    unreadable: list[str] = field(default_factory=list[str])
    failed: list[str] = field(default_factory=list[str])
    foreign: list[str] = field(default_factory=list[str])


def _stored_at(folder: Path, now: dt.datetime) -> dt.datetime | None:
    """Read a document's storage time, or ``None`` if it cannot be trusted."""
    meta = folder / META
    if meta.is_symlink() or not meta.is_file():
        return None
    try:
        raw = json.loads(meta.read_text(encoding="utf-8"))
        stored_at = dt.datetime.fromisoformat(raw["stored_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if stored_at.tzinfo is None or stored_at > now:
        return None
    return stored_at


def _remove(folder: Path, report: SweepReport, doc_id: str) -> None:
    """Delete a directory, recording success or failure instead of raising."""
    try:
        shutil.rmtree(folder)
    except OSError:
        report.failed.append(doc_id)
    else:
        report.deleted.append(doc_id)


def _is_plain_dir(entry: Path) -> bool:
    """A real directory, not a link to one. Junctions slip past is_symlink() on Python 3.12."""
    return entry.is_dir() and not entry.is_symlink() and not entry.is_junction()


def _sweep_entry(
    entry: Path, policy: RetentionPolicy, now: dt.datetime, report: SweepReport, dry_run: bool
) -> None:
    """Classify one entry of the store and act on it."""
    name = entry.name
    if not _is_plain_dir(entry):
        report.foreign.append(name)
        return
    if name.startswith(DELETING) and is_document_id(name.removeprefix(DELETING)):
        doc_id = name.removeprefix(DELETING)
        if dry_run:
            report.deleted.append(doc_id)
        else:
            _remove(entry, report, doc_id)
        return
    staged = name.startswith(INCOMING) and is_document_id(name.removeprefix(INCOMING))
    if not (staged or is_document_id(name)):
        report.foreign.append(name)
        return
    doc_id = name.removeprefix(INCOMING)
    stored_at = _stored_at(entry, now)
    if stored_at is None:
        report.unreadable.append(doc_id)
    elif not is_expired(stored_at, now, policy):
        report.kept.append(doc_id)
    elif dry_run:
        report.deleted.append(doc_id)
    else:
        try:
            # Renaming first makes the decision durable: if removal stops halfway, the next
            # sweep sees the tombstone and finishes it rather than finding an undatable remnant.
            tombstone = entry.rename(entry.with_name(f"{DELETING}{doc_id}"))
        except OSError:
            report.failed.append(doc_id)
            return
        _remove(tombstone, report, doc_id)


def sweep(
    store: Path, policy: RetentionPolicy, now: dt.datetime, *, dry_run: bool = False
) -> SweepReport:
    """Delete stored documents older than the policy allows.

    Args:
        store: The store directory.
        policy: The retention policy.
        now: The current time; timezone-aware.
        dry_run: Report what would be deleted without deleting or renaming anything.

    Returns:
        What was deleted, kept, unreadable, failed or foreign. A failure on one document never
        stops the sweep of the others.

    Raises:
        ValueError: If ``now`` is naive.
        FileNotFoundError: If ``store`` does not exist; a missing store is not an empty one.
    """
    _aware(now, "now")
    if not store.is_dir():
        raise FileNotFoundError(f"retention store {store} does not exist")
    report = SweepReport()
    for entry in sorted(store.iterdir()):
        _sweep_entry(entry, policy, now, report, dry_run)
    for bucket in (report.deleted, report.kept, report.unreadable, report.failed, report.foreign):
        bucket.sort()
    return report
