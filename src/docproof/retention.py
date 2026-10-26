"""Opaque document ids and a retention sweep for stored documents.

A stored document is a directory ``<store>/<doc_id>/`` holding ``meta.json`` (when it was stored)
and its files. The id is random, so it reveals nothing: not the file name, the customer or the date.
The sweep deletes every document older than the policy allows.

Deliberate boundary: the retention period is a required argument with no default. How long a
document may be kept is a product and legal decision that has not been made, and a default here
would quietly become the answer. The sweep is careful about what it deletes: only directories whose
name is a document id, never through a symlink, and never a document whose age it cannot read.
Unreadable entries are reported, not guessed.
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
META = "meta.json"


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


def store_document(store: Path, name: str, content: bytes, now: dt.datetime) -> str:
    """Store one document under a fresh opaque id.

    Args:
        store: The store directory; created if missing.
        name: File name to store the content under. Only its final component is used, so a name
            like ``../x`` cannot escape the document's directory.
        content: The document bytes.
        now: Storage time; timezone-aware.

    Returns:
        The new document id.

    Raises:
        ValueError: If ``now`` is naive or ``name`` has no usable file name.
    """
    stored_at = _aware(now, "now")
    safe_name = Path(name).name
    if safe_name in {"", ".", "..", META}:
        raise ValueError(f"unusable file name {name!r}")
    store.mkdir(parents=True, exist_ok=True)
    doc_id = new_document_id()
    folder = store / doc_id
    folder.mkdir()
    (folder / safe_name).write_bytes(content)
    (folder / META).write_text(json.dumps({"stored_at": stored_at.isoformat()}), encoding="utf-8")
    return doc_id


@dataclass(slots=True)
class SweepReport:
    """What a sweep did.

    Attributes:
        deleted: Ids of documents deleted.
        kept: Ids of documents still within the policy.
        unreadable: Ids whose storage time could not be read; kept and flagged for a person.
        foreign: Entries that are not document directories (wrong name, a file, a symlink or
            junction);
            never touched.
    """

    deleted: list[str] = field(default_factory=list[str])
    kept: list[str] = field(default_factory=list[str])
    unreadable: list[str] = field(default_factory=list[str])
    foreign: list[str] = field(default_factory=list[str])


def _stored_at(folder: Path) -> dt.datetime | None:
    """Read a document's storage time, or ``None`` if it is missing or malformed."""
    meta = folder / META
    if meta.is_symlink() or not meta.is_file():
        return None
    try:
        raw = json.loads(meta.read_text(encoding="utf-8"))
        stored_at = dt.datetime.fromisoformat(raw["stored_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return stored_at if stored_at.tzinfo is not None else None


def sweep(
    store: Path, policy: RetentionPolicy, now: dt.datetime, *, dry_run: bool = False
) -> SweepReport:
    """Delete stored documents older than the policy allows.

    Args:
        store: The store directory.
        policy: The retention policy.
        now: The current time; timezone-aware.
        dry_run: Report what would be deleted without deleting it.

    Returns:
        What was deleted, kept, unreadable or foreign, each sorted.

    Raises:
        ValueError: If ``now`` is naive.
        FileNotFoundError: If ``store`` does not exist; a missing store is not an empty one.
    """
    _aware(now, "now")
    if not store.is_dir():
        raise FileNotFoundError(f"retention store {store} does not exist")
    report = SweepReport()
    for entry in sorted(store.iterdir()):
        # Junctions are links too, but is_symlink() is False for them on Python 3.12.
        linked = entry.is_symlink() or entry.is_junction()
        if linked or not entry.is_dir() or not is_document_id(entry.name):
            report.foreign.append(entry.name)
            continue
        stored_at = _stored_at(entry)
        if stored_at is None:
            report.unreadable.append(entry.name)
        elif is_expired(stored_at, now, policy):
            if not dry_run:
                shutil.rmtree(entry)
            report.deleted.append(entry.name)
        else:
            report.kept.append(entry.name)
    return report
