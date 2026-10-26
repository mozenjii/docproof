"""Provenance guard: no committed document without a record of where it came from.

The repository may only hold generated or public documents, never a real invoice or scan that was
committed by accident. Every file git stores that looks like a document must match a pattern in
the registry ``data/provenance.json``, which says whether it is synthetic (and which generator in
this repository produced it) or public (and its URL and licence). Run
``python -m docproof.provenance`` from the repository root.

What is judged is what git stores, never the working tree: a file deleted, edited or replaced on
disk after it was staged does not change the verdict. Three scopes:

* default: every entry of the index, the content the next commit would carry;
* ``--rev REV``: the tree of one commit;
* ``--history``: every path and blob in every commit reachable from any ref. One
  ``git log --all --raw`` over the commits, oldest first, lists for each commit the entries that
  differ from its first parent (all of them for a root commit). Every blob in a commit's tree is
  either in its first parent's tree or in that diff, and every reachable commit is walked, so the
  diffs together name every (path, blob) pair that ever existed, at the cost of one process and
  the size of the changes rather than one ``ls-tree`` per commit. The first commit that brings in
  a document-like blob at a path is reported with the path.

Blob contents come from one ``git cat-file --batch`` process; only the leading ``HEAD_SIZE`` bytes
of each are judged. A gitlink (a submodule) is document-like by definition, because its content
lives in another repository this guard never reads; it must be registered like any document.

Accepted limits, documented rather than detected: a document in a text encoding (base64, hex,
quoted-printable), encrypted or in a container other than ZIP and gzip (7z, xz, bzip2, rar, tar),
a plain-text invoice (txt, csv, html, json) outside ``data/``, XML in UTF-16, a document whose
signature sits beyond the first ``HEAD_SIZE`` bytes, the content of a submodule or of a Git LFS
object (the pointer is detected, the stored object is not read), and objects reachable from no ref
(reflog entries, dropped stashes, dangling blobs). The guard reports; it blocks only where it is
run, such as a hook or a required CI check.

Deliberate boundary: ``load_registry``, ``is_document_like``, ``check`` and the ``parse_*``
functions are pure and decide everything. ``list_index``, ``list_tree``, ``list_history``,
``read_registry`` and ``main`` only gather input and report, and they fail closed: git output that
cannot be produced or understood is an error, never an empty list.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path
from typing import IO, cast

REGISTRY_PATH = "data/provenance.json"
"""Where the registry lives, relative to the repository root. The registry itself is exempt."""

HEAD_SIZE = 1024
"""How many leading bytes of each blob are judged; PDF readers search this far for the header."""

GIT_TIMEOUT_SECONDS = 60
"""How long one git command may run before the listing is abandoned as failed."""

SHORT_SHA = 12
"""How many hex digits of a commit id the ``--history`` report prints."""

SOURCES = frozenset({"synthetic", "public"})
"""The only two places a committed document may come from."""

DOCUMENT_EXTENSIONS = frozenset(
    {
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
        "bmp",
        "gif",
        "webp",
        "heic",
        "heif",
        "docx",
        "doc",
        "xlsx",
        "xls",
        "odt",
        "ods",
        "eml",
        "msg",
        "rtf",
    }
)
"""Extensions, lower case and without the dot, that mark a file as a document.

``xml`` is deliberately absent: most XML in a repository is configuration. XML e-invoices are
caught by content instead (``E_INVOICE_MARKERS``).
"""

DOCUMENT_SIGNATURES = (
    b"%PDF",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"II*\x00",
    b"MM\x00*",
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    b"\x1f\x8b",
    b"{\\rtf",
    b"version https://git-lfs.github.com/spec",
)
"""Leading bytes that mark a blob as a document whatever its name.

PDF, PNG, JPEG, TIFF (both byte orders), GIF, ZIP (docx, xlsx and odt are ZIP containers),
OLE2 compound files (legacy doc, xls and Outlook msg), gzip, RTF, and Git LFS pointers, whose real
content is stored outside git and so cannot be inspected here.
"""

PDF_HEADER = b"%PDF-"
"""A PDF header counts anywhere in the first ``HEAD_SIZE`` bytes, because readers accept that."""

HEIF_BRANDS = frozenset({b"heic", b"heix", b"mif1", b"msf1", b"avif"})
"""ISO media brands, after ``ftyp`` at offset 4, of HEIC, HEIF and AVIF images."""

E_INVOICE_MARKERS = (
    b"urn:oasis:names:specification:ubl",
    b"<Invoice",
    b"<rsm:CrossIndustryInvoice",
)
"""Text that marks an XML blob as a structured invoice (UBL, Factur-X, ZUGFeRD, XRechnung)."""

SYNTHETIC_ORIGIN_PREFIXES = ("python -m docproof.", "scripts/", "bench/", "src/docproof/", "tests/")
"""A synthetic origin must start with one of these, so it names a generator in this repository.

``tests/`` covers test files whose fixtures look like documents, such as a PDF header in a byte
string near the top of a test module.
"""

_MODULE_ORIGIN = re.compile(r"python -m docproof(\.[A-Za-z_][A-Za-z0-9_]*)+( \S.*)?")
_PATH_ORIGIN = re.compile(r"(scripts|bench|src/docproof|tests)(/[A-Za-z0-9_.-]+)+( \S.*)?")
_WILDCARDS = frozenset("*?[")
_RECORD_KEYS = frozenset({"pattern", "source", "origin", "license"})
_GITLINK_MODE = "160000"
_UTF8_BOM = b"\xef\xbb\xbf"


class RegistryError(ValueError):
    """The provenance registry is missing, unreadable or does not follow the record contract."""


class ListingError(RuntimeError):
    """What git stores could not be listed or read, so nothing can be said about it."""


@dataclass(frozen=True)
class ProvenanceRecord:
    """Where a group of committed documents came from.

    Attributes:
        pattern: Glob relative to the repository root, with forward slashes. ``*`` and ``?`` stay
            within one path segment; a ``**`` segment spans any number of segments, including none.
            The first segment is always literal, so one record cannot cover the whole repository.
        source: ``"synthetic"`` or ``"public"``.
        origin: For synthetic files, the module (``python -m docproof....``) or repository path
            (under ``scripts/``, ``bench/`` or ``src/docproof/``) that produced them; for public
            files, the ``https://`` URL they were taken from.
        license: SPDX id or licence name. Required for public files, may be ``None`` for
            synthetic ones.
    """

    pattern: str
    source: str
    origin: str
    license: str | None


@dataclass(frozen=True)
class TrackedEntry:
    """One path and the content git stores for it.

    Attributes:
        path: Path relative to the repository root, with forward slashes.
        head: The blob's first ``HEAD_SIZE`` bytes; empty for a gitlink.
        gitlink: True for a submodule entry (mode 160000), whose content git does not hold here.
        commit: In ``--history`` mode, the first commit that carried this blob at this path;
            otherwise ``None``.
        oid: The git object id of the blob or, for a gitlink, of the submodule commit.
    """

    path: str
    head: bytes = b""
    gitlink: bool = False
    commit: str | None = None
    oid: str = ""


@dataclass(frozen=True)
class Violation:
    """A document-like path that no record covers.

    Attributes:
        path: The path, with forward slashes.
        first_commit: In ``--history`` mode, the first commit that carried a document-like blob at
            this path; otherwise ``None``.
    """

    path: str
    first_commit: str | None


@dataclass(frozen=True)
class Report:
    """The outcome of checking stored files against the registry.

    Attributes:
        violations: Document-like paths that no record covers, sorted by path. Any violation
            fails the check.
        stale_patterns: Patterns, in registry order, that match no listed path. These are
            warnings: the record is probably left over from a file that was removed.
    """

    violations: tuple[Violation, ...]
    stale_patterns: tuple[str, ...]


def _require_text(record: Mapping[object, object], key: str, index: int) -> str:
    """Return a record field that must be a non-empty string.

    Raises:
        RegistryError: if the field is not a string or is empty.
    """
    value = record[key]
    if not isinstance(value, str) or not value:
        raise RegistryError(f"record {index}: {key!r} must be a non-empty string, got {value!r}")
    return value


def _check_pattern(pattern: str, index: int) -> None:
    """Refuse a pattern that could cover the whole repository or reach outside it.

    A wildcard in the first segment (``**``, ``*``, ``*.pdf``, ``[a-z]*``) matches at the root, so
    one such record would switch the guard off for every directory.

    Raises:
        RegistryError: naming the record and the rule the pattern breaks.
    """
    if pattern.startswith("/"):
        raise RegistryError(f"record {index}: pattern {pattern!r} must be relative, not start /")
    segments = pattern.split("/")
    if ".." in segments:
        raise RegistryError(f"record {index}: pattern {pattern!r} must not contain '..'")
    if _WILDCARDS.intersection(segments[0]):
        raise RegistryError(
            f"record {index}: pattern {pattern!r} needs a literal first segment, not a wildcard"
        )
    # Wildcards only in the file name, and never alone: "tests/**" or "data/*" would vouch for a
    # whole directory, including documents added to it later.
    if "**" in segments or any(_WILDCARDS.intersection(s) for s in segments[:-1]):
        raise RegistryError(
            f"record {index}: pattern {pattern!r} may use wildcards only in its last segment"
        )
    if not segments[-1].strip("*?[]"):
        raise RegistryError(
            f"record {index}: pattern {pattern!r} needs a literal part in its file name"
        )


def _names_repo_generator(origin: str) -> bool:
    """Say whether a synthetic origin names a module or path inside this repository."""
    if _MODULE_ORIGIN.fullmatch(origin):
        return True
    if not _PATH_ORIGIN.fullmatch(origin):
        return False
    # A path that climbs out with '..' would name something outside the repository.
    segments = origin.split(" ", 1)[0].split("/")
    return not {".", ".."}.intersection(segments)


def _parse_record(item: object, index: int) -> ProvenanceRecord:
    """Validate one registry entry and turn it into a record.

    Raises:
        RegistryError: naming the entry's position and what is wrong with it.
    """
    if not isinstance(item, dict):
        raise RegistryError(f"record {index}: expected an object, got {type(item).__name__}")
    record = cast(dict[object, object], item)
    keys = set(record)
    unknown = keys - _RECORD_KEYS
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise RegistryError(f"record {index}: unknown keys {names}")
    missing = _RECORD_KEYS - keys
    if missing:
        raise RegistryError(f"record {index}: missing keys {', '.join(sorted(missing))}")
    pattern = _require_text(record, "pattern", index)
    source = _require_text(record, "source", index)
    origin = _require_text(record, "origin", index)
    _check_pattern(pattern, index)
    if source not in SOURCES:
        raise RegistryError(f"record {index}: source must be synthetic or public, got {source!r}")
    license_value = record["license"]
    if license_value is not None and not isinstance(license_value, str):
        raise RegistryError(f"record {index}: 'license' must be a string or null")
    if source == "public":
        if not origin.startswith("https://"):
            raise RegistryError(f"record {index}: a public origin must be an https:// URL")
        if not license_value:
            raise RegistryError(f"record {index}: a public record needs a licence")
    elif not _names_repo_generator(origin):
        raise RegistryError(
            f"record {index}: a synthetic origin must name a generator in this repository "
            f"(a module or path starting {', '.join(SYNTHETIC_ORIGIN_PREFIXES)}), got {origin!r}"
        )
    return ProvenanceRecord(pattern=pattern, source=source, origin=origin, license=license_value)


def load_registry(raw: object) -> list[ProvenanceRecord]:
    """Validate a decoded registry and return its records.

    Args:
        raw: The registry as decoded from JSON.

    Returns:
        The records in registry order.

    Raises:
        RegistryError: if ``raw`` is not a list or any entry breaks the record contract.
    """
    if not isinstance(raw, list):
        raise RegistryError(f"the registry must be a JSON array, got {type(raw).__name__}")
    items = cast(list[object], raw)
    return [_parse_record(item, index) for index, item in enumerate(items)]


def _is_e_invoice(head: bytes) -> bool:
    """Say whether a blob is XML that carries a structured invoice."""
    text = head.removeprefix(_UTF8_BOM).lstrip()
    return text.startswith(b"<") and any(marker in head for marker in E_INVOICE_MARKERS)


def _content_is_document(head: bytes) -> bool:
    """Say whether a blob's leading bytes mark it as a document, whatever its name."""
    if head.startswith(DOCUMENT_SIGNATURES) or PDF_HEADER in head[:HEAD_SIZE]:
        return True
    # BMP: "BM", a 4-byte size, then four reserved zero bytes. "BM" alone starts too much text.
    if head[:2] == b"BM" and head[6:10] == bytes(4):
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    if head[4:8] == b"ftyp" and head[8:12] in HEIF_BRANDS:
        return True
    return _is_e_invoice(head)


def is_document_like(path: str, head: bytes, *, gitlink: bool = False) -> bool:
    """Say whether a stored file must have a provenance record.

    Args:
        path: Path relative to the repository root, with forward slashes.
        head: The blob's first bytes; ``HEAD_SIZE`` is enough for every rule.
        gitlink: True for a submodule entry, whose content this guard cannot see.

    Returns:
        True for a gitlink; for a path under a top-level ``data/`` directory in any letter case
        (other than the registry), because data directories are where real documents end up
        whatever their format; for a document extension; or when the content marks it as one.
    """
    if gitlink:
        return True
    top, separator, _ = path.partition("/")
    # Case-insensitive: on a case-sensitive checkout Data/ and data/ are different directories.
    if separator and top.lower() == "data" and path != REGISTRY_PATH:
        return True
    name = path.rsplit("/", 1)[-1]
    if "." in name and name.rsplit(".", 1)[1].lower() in DOCUMENT_EXTENSIONS:
        return True
    return _content_is_document(head)


def matches(pattern: str, path: str) -> bool:
    """Match a forward-slash path against a registry glob.

    Each segment is matched with ``fnmatch`` rules (``*``, ``?``, ``[seq]``), case-sensitively as
    git stores paths, so ``*`` never crosses a ``/``. A segment that is exactly ``**`` matches zero
    or more whole segments.

    Args:
        pattern: The glob, for example ``data/samples/**/*.png``.
        path: The path to test, for example ``data/samples/a/b/page.png``.

    Returns:
        True if the whole path matches the whole pattern.
    """
    pattern_parts = tuple(pattern.split("/"))
    path_parts = tuple(path.split("/"))

    @cache
    def match_from(p: int, s: int) -> bool:
        if p == len(pattern_parts):
            return s == len(path_parts)
        if pattern_parts[p] == "**":
            return any(match_from(p + 1, k) for k in range(s, len(path_parts) + 1))
        return (
            s < len(path_parts)
            and fnmatchcase(path_parts[s], pattern_parts[p])
            and match_from(p + 1, s + 1)
        )

    return match_from(0, 0)


def check(entries: Iterable[TrackedEntry], registry: Sequence[ProvenanceRecord]) -> Report:
    """Find document-like paths that no record covers, and records that cover nothing.

    Args:
        entries: The stored files. A path may appear more than once (several blobs in history,
            or conflict stages in the index); it is a document if any of its blobs is one. In
            history order, the first document-like entry of a path names its first commit.
        registry: The validated records.

    Returns:
        The violations and the stale patterns.
    """
    first_document: dict[str, str | None] = {}
    paths: set[str] = set()
    for entry in entries:
        paths.add(entry.path)
        if entry.path not in first_document and is_document_like(
            entry.path, entry.head, gitlink=entry.gitlink
        ):
            first_document[entry.path] = entry.commit
    violations = tuple(
        Violation(path=path, first_commit=first_document[path])
        for path in sorted(first_document)
        if not any(matches(record.pattern, path) for record in registry)
    )
    stale = tuple(
        record.pattern
        for record in registry
        if not any(matches(record.pattern, path) for path in paths)
    )
    return Report(violations=violations, stale_patterns=stale)


def _decode_path(raw: bytes) -> str:
    """Decode a path as git stores it; undecodable bytes survive as surrogates."""
    return raw.decode("utf-8", errors="surrogateescape")


def _split_records(listing: bytes) -> list[bytes]:
    """Split NUL-terminated git output into its non-empty records."""
    return [record for record in listing.split(b"\0") if record]


def parse_ls_files(listing: bytes) -> list[TrackedEntry]:
    """Parse ``git ls-files -s -z``: ``<mode> <oid> <stage>``, a tab, ``<path>``.

    Args:
        listing: The command's standard output.

    Returns:
        One entry per distinct (path, object), without contents.

    Raises:
        ListingError: if a record does not have that shape.
    """
    entries: dict[tuple[str, str], TrackedEntry] = {}
    for record in _split_records(listing):
        meta, tab, raw_path = record.partition(b"\t")
        fields = meta.split(b" ")
        if not tab or len(fields) != 3:
            raise ListingError(f"unexpected git ls-files record {record[:80]!r}")
        mode, oid = fields[0].decode("latin-1"), fields[1].decode("latin-1")
        path = _decode_path(raw_path)
        entries.setdefault(
            (path, oid), TrackedEntry(path=path, gitlink=mode == _GITLINK_MODE, oid=oid)
        )
    return list(entries.values())


def parse_ls_tree(listing: bytes) -> list[TrackedEntry]:
    """Parse ``git ls-tree -r -z``: ``<mode> <type> <oid>``, a tab, ``<path>``.

    Args:
        listing: The command's standard output.

    Returns:
        One entry per path, without contents.

    Raises:
        ListingError: if a record does not have that shape.
    """
    entries: list[TrackedEntry] = []
    for record in _split_records(listing):
        meta, tab, raw_path = record.partition(b"\t")
        fields = meta.split(b" ")
        if not tab or len(fields) != 3:
            raise ListingError(f"unexpected git ls-tree record {record[:80]!r}")
        entries.append(
            TrackedEntry(
                path=_decode_path(raw_path),
                gitlink=fields[0] == _GITLINK_MODE.encode(),
                oid=fields[2].decode("latin-1"),
            )
        )
    return entries


def parse_log(listing: bytes) -> list[TrackedEntry]:
    """Parse ``git log --raw -z --format=%x01%H`` into the (path, blob) pairs history introduced.

    Each commit starts with a record of byte 0x01 and the commit id; each change is a
    ``:<old mode> <new mode> <old oid> <new oid> <status>`` record followed by a path record.
    Deletions bring in nothing and are skipped.

    Args:
        listing: The command's standard output, commits oldest first.

    Returns:
        One entry per distinct (path, object), in order of first appearance, each naming the
        first commit that carried it.

    Raises:
        ListingError: if the output does not have that shape.
    """
    entries: dict[tuple[str, str], TrackedEntry] = {}
    commit: str | None = None
    records = iter(listing.split(b"\0"))
    for raw in records:
        # git separates a commit header from its changes with a newline.
        record = raw.lstrip(b"\n")
        if not record:
            continue
        if record.startswith(b"\x01"):
            commit = record[1:].decode("latin-1")
            continue
        fields = record[1:].split(b" ")
        raw_path = next(records, None)
        if commit is None or not record.startswith(b":") or len(fields) != 5 or not raw_path:
            raise ListingError(f"unexpected git log record {record[:80]!r}")
        if fields[4] == b"D":
            continue
        path, oid = _decode_path(raw_path), fields[3].decode("latin-1")
        entries.setdefault(
            (path, oid),
            TrackedEntry(
                path=path, gitlink=fields[1] == _GITLINK_MODE.encode(), commit=commit, oid=oid
            ),
        )
    return list(entries.values())


def parse_batch(stream: IO[bytes], oids: Sequence[str]) -> dict[str, bytes]:
    """Read ``git cat-file --batch`` output, keeping the first ``HEAD_SIZE`` bytes of each blob.

    Args:
        stream: A seekable stream holding the command's output, positioned at its start.
        oids: The object ids that were requested, in request order.

    Returns:
        Each object id mapped to its blob's leading bytes.

    Raises:
        ListingError: if an object is missing, is not a blob, or the output is cut short.
    """
    heads: dict[str, bytes] = {}
    for oid in oids:
        header = stream.readline()
        fields = header.rstrip(b"\n").split(b" ")
        if len(fields) != 3 or fields[0].decode("latin-1") != oid or not fields[2].isdigit():
            raise ListingError(f"cannot read object {oid}: git answered {header[:120]!r}")
        if fields[1] != b"blob":
            raise ListingError(f"object {oid} is a {fields[1].decode('latin-1')}, not a blob")
        size = int(fields[2])
        head = stream.read(min(size, HEAD_SIZE))
        # Skip the rest of the blob; only its head is judged.
        stream.seek(size - len(head), os.SEEK_CUR)
        if stream.read(1) != b"\n":
            raise ListingError(f"git cat-file output for {oid} is cut short")
        heads[oid] = head
    return heads


def _git(
    git: str,
    repo: Path,
    *args: str,
    stdin: IO[bytes] | None = None,
    stdout: IO[bytes] | None = None,
) -> bytes:
    """Run one git command in ``repo`` and return its standard output.

    Replace refs are ignored so a ``refs/replace`` entry cannot show the guard a different
    history from the one that is pushed.

    Args:
        git: The git executable.
        repo: The working directory for the command.
        *args: The git arguments.
        stdin: A file to feed as standard input; nothing is fed when ``None``.
        stdout: A file to receive standard output; when given, the return value is empty.

    Returns:
        The standard output, unless it went to ``stdout``.

    Raises:
        ListingError: if git cannot start, times out or exits non-zero.
    """
    try:
        result = subprocess.run(
            [git, "--no-replace-objects", *args],
            cwd=repo,
            check=True,
            stdin=stdin if stdin is not None else subprocess.DEVNULL,
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        detail = cast(bytes, exc.stderr).decode("utf-8", errors="replace").strip()
        raise ListingError(f"git {' '.join(args)} failed in {repo}: {detail}") from exc
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise ListingError(f"git {' '.join(args)} failed in {repo}: {exc}") from exc
    return result.stdout or b""


def _repository_git(repo: Path) -> str:
    """Find git and confirm ``repo`` is the root of a working tree.

    Raises:
        ListingError: if git is not installed, ``repo`` is not a repository, or it is a
            subdirectory rather than the root (paths would then be relative to the wrong place).
    """
    git = shutil.which("git")
    if git is None:
        raise ListingError("git was not found on PATH")
    prefix = _git(git, repo, "rev-parse", "--show-prefix").decode("utf-8").strip()
    if prefix:
        raise ListingError(f"{repo} is inside a repository but is not its root ({prefix})")
    return git


def _with_heads(git: str, repo: Path, entries: list[TrackedEntry]) -> list[TrackedEntry]:
    """Attach each blob's leading bytes, read through one ``git cat-file --batch`` process.

    Both the request and the answer go through temporary files: neither pipe can then fill up and
    stall git, and the answer, which holds every blob in full, never has to sit in memory.

    Raises:
        ListingError: if any blob cannot be read.
    """
    oids = list(dict.fromkeys(entry.oid for entry in entries if not entry.gitlink))
    if not oids:
        return entries
    with tempfile.TemporaryFile() as request, tempfile.TemporaryFile() as answer:
        request.write("".join(f"{oid}\n" for oid in oids).encode("latin-1"))
        request.seek(0)
        _git(git, repo, "cat-file", "--batch", stdin=request, stdout=answer)
        answer.seek(0)
        heads = parse_batch(answer, oids)
    return [entry if entry.gitlink else replace(entry, head=heads[entry.oid]) for entry in entries]


def list_index(repo: Path) -> list[TrackedEntry]:
    """List every index entry of ``repo`` with the leading bytes of its staged blob.

    Args:
        repo: The root of a git working tree.

    Returns:
        One entry per distinct (path, blob); conflict stages may give a path several.

    Raises:
        ListingError: if git is missing, fails, or ``repo`` is not the root of a working tree.
    """
    git = _repository_git(repo)
    return _with_heads(git, repo, parse_ls_files(_git(git, repo, "ls-files", "-s", "-z")))


def list_tree(repo: Path, rev: str) -> list[TrackedEntry]:
    """List every file in the tree of one commit, with the leading bytes of each blob.

    Args:
        repo: The root of a git working tree.
        rev: Anything git resolves to a commit, such as ``HEAD`` or a branch name.

    Returns:
        One entry per path in that commit.

    Raises:
        ListingError: if ``rev`` names no commit, or git is missing or fails.
    """
    git = _repository_git(repo)
    # --end-of-options stops a revision such as '--output=x' being read as an option.
    resolved = _git(git, repo, "rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}")
    commit = resolved.decode("latin-1").strip()
    listing = _git(git, repo, "ls-tree", "-r", "-z", "--full-tree", commit)
    return _with_heads(git, repo, parse_ls_tree(listing))


def list_history(repo: Path) -> list[TrackedEntry]:
    """List every (path, blob) pair in every commit reachable from any ref, oldest first.

    The method, and why it misses nothing reachable, is in the module docstring.

    Args:
        repo: The root of a git working tree.

    Returns:
        One entry per distinct (path, blob), each naming the first commit that carried it.

    Raises:
        ListingError: if git is missing or fails.
    """
    git = _repository_git(repo)
    # A shallow clone (CI's default) would report "ok" for the commits it never fetched.
    if _git(git, repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise ListingError(f"{repo} is a shallow clone; fetch full history before --history")
    # git log walks commits only. A tag can point straight at a blob or a tree, which a push
    # carries but the walk below never sees, so such a ref is refused rather than ignored.
    refs = _git(git, repo, "for-each-ref", "--format=%(objecttype) %(*objecttype) %(refname)")
    for line in refs.decode("utf-8", errors="replace").splitlines():
        own, _, rest = line.partition(" ")
        peeled, _, name = rest.partition(" ")
        if (peeled or own) in {"blob", "tree"}:
            raise ListingError(f"ref {name} points at a {peeled or own}, not a commit")
    listing = _git(
        git,
        repo,
        "log",
        "--all",
        "--reverse",
        "--topo-order",
        "--root",
        "--raw",
        "--no-abbrev",
        "-z",
        "--format=%x01%H",
        "--diff-merges=first-parent",
        # Each of these overrides a user setting that would hide or reshape entries.
        "--no-renames",
        "--no-relative",
        "--ignore-submodules=none",
        "--no-color",
        "--no-show-signature",
    )
    return _with_heads(git, repo, parse_log(listing))


def read_registry(path: Path) -> list[ProvenanceRecord]:
    """Read and validate the registry file.

    Args:
        path: The registry's location on disk.

    Returns:
        The records in registry order.

    Raises:
        RegistryError: if the file is missing or unreadable, is not valid JSON, or breaks the
            record contract. A missing registry is an error, not an empty one, so deleting the
            file cannot switch the guard off.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"cannot read the registry {path}: {exc}") from exc
    try:
        raw: object = json.loads(text)
    except ValueError as exc:
        raise RegistryError(f"the registry {path} is not valid JSON: {exc}") from exc
    return load_registry(raw)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        0 when every document has a record, 1 when some do not, 2 when the registry is invalid
        or what git stores cannot be listed.
    """
    parser = argparse.ArgumentParser(
        description="Fail when a document stored in git has no provenance record."
    )
    parser.add_argument("--repo", type=Path, default=Path(), help="repository root")
    parser.add_argument(
        "--registry", type=Path, help=f"registry file; default <repo>/{REGISTRY_PATH}"
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--rev", help="check the tree of this commit instead of the index")
    scope.add_argument(
        "--history", action="store_true", help="check every commit reachable from any ref"
    )
    args = parser.parse_args(argv)
    repo = cast(Path, args.repo)
    registry_path = cast(Path | None, args.registry) or repo / REGISTRY_PATH
    rev = cast(str | None, args.rev)
    history = cast(bool, args.history)
    try:
        registry = read_registry(registry_path)
        if history:
            entries, where = list_history(repo), "across history"
        elif rev is not None:
            entries, where = list_tree(repo, rev), f"in {rev}"
        else:
            entries, where = list_index(repo), "in the index"
    except (RegistryError, ListingError) as exc:
        print(f"provenance: {exc}", file=sys.stderr)
        return 2
    report = check(entries, registry)
    for pattern in report.stale_patterns:
        print(f"provenance: warning: no stored file matches {pattern!r}", file=sys.stderr)
    if report.violations:
        print(
            f"provenance: {len(report.violations)} stored document(s) {where} without a record "
            f"in {registry_path}:",
            file=sys.stderr,
        )
        for violation in report.violations:
            if violation.first_commit is None:
                print(violation.path)
            else:
                print(f"{violation.path} (first in {violation.first_commit[:SHORT_SHA]})")
        return 1
    paths = len({entry.path for entry in entries})
    print(f"provenance: ok, {paths} paths {where}, every document has a record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
