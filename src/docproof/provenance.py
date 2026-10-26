"""Provenance guard: no tracked document without a record of where it came from.

The repository may only hold generated or public documents, never a real invoice or scan that was
committed by accident. Every tracked file that looks like a document must match a pattern in the
registry ``data/provenance.json``, which says whether it is synthetic (and what produced it) or
public (and its URL and licence). Run ``python -m docproof.provenance`` from the repository root.

Deliberate boundary: ``load_registry``, ``is_document_like`` and ``check`` are pure and decide
everything. ``tracked_files``, ``read_registry`` and ``main`` only gather input and report, and
they fail closed: a file list that cannot be produced is an error, never an empty list.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path
from typing import cast

REGISTRY_PATH = "data/provenance.json"
"""Where the registry lives, relative to the repository root. The registry itself is exempt."""

HEAD_SIZE = 16
"""How many leading bytes of each file the signature check needs."""

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
    }
)
"""Extensions, lower case and without the dot, that mark a file as a document."""

DOCUMENT_SIGNATURES = (
    b"%PDF",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"II*\x00",
    b"MM\x00*",
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",
)
"""Leading bytes of PDF, PNG, JPEG, TIFF (both byte orders), GIF and ZIP-based office files.

The signature check catches a document whose extension was changed, such as a PDF saved as
``invoice.txt``. ZIP is included because docx, xlsx and odt are ZIP containers.
"""

_RECORD_KEYS = frozenset({"pattern", "source", "origin", "license"})


class RegistryError(ValueError):
    """The provenance registry is missing, unreadable or does not follow the record contract."""


class ListingError(RuntimeError):
    """The tracked files could not be listed, so nothing can be said about them."""


@dataclass(frozen=True)
class ProvenanceRecord:
    """Where a group of tracked documents came from.

    Attributes:
        pattern: Glob relative to the repository root, with forward slashes. ``*`` and ``?`` stay
            within one path segment; a ``**`` segment spans any number of segments, including none.
        source: ``"synthetic"`` or ``"public"``.
        origin: For synthetic files, the command or module that produced them; for public files,
            the ``https://`` URL they were taken from.
        license: SPDX id or licence name. Required for public files, may be ``None`` for
            synthetic ones.
    """

    pattern: str
    source: str
    origin: str
    license: str | None


@dataclass(frozen=True)
class Report:
    """The outcome of checking tracked files against the registry.

    Attributes:
        violations: Sorted paths of document-like files that no record covers. Any violation fails
            the check.
        stale_patterns: Patterns, in registry order, that match no tracked file. These are
            warnings: the record is probably left over from a file that was removed.
    """

    violations: tuple[str, ...]
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


def is_document_like(path: str, head: bytes) -> bool:
    """Say whether a tracked file must have a provenance record.

    Args:
        path: Path relative to the repository root, with forward slashes.
        head: The file's first bytes; ``HEAD_SIZE`` is enough for every signature.

    Returns:
        True if the extension or the leading bytes mark it as a document, or if it lives under
        ``data/`` (other than the registry), because data directories are where real documents
        end up whatever their format.
    """
    if path.startswith("data/") and path != REGISTRY_PATH:
        return True
    name = path.rsplit("/", 1)[-1]
    if "." in name and name.rsplit(".", 1)[1].lower() in DOCUMENT_EXTENSIONS:
        return True
    return head.startswith(DOCUMENT_SIGNATURES)


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


def check(files: Mapping[str, bytes], registry: Sequence[ProvenanceRecord]) -> Report:
    """Find document-like tracked files that no record covers, and records that cover nothing.

    Args:
        files: Each tracked path, forward slashes, mapped to its first ``HEAD_SIZE`` bytes.
        registry: The validated records.

    Returns:
        The violations and the stale patterns.
    """
    violations = sorted(
        path
        for path, head in files.items()
        if is_document_like(path, head)
        and not any(matches(record.pattern, path) for record in registry)
    )
    stale = tuple(
        record.pattern
        for record in registry
        if not any(matches(record.pattern, path) for path in files)
    )
    return Report(violations=tuple(violations), stale_patterns=stale)


def _git(git: str, repo: Path, *args: str) -> bytes:
    """Run one git command in ``repo`` and return its standard output.

    Raises:
        ListingError: if git exits non-zero, for example because ``repo`` is not a repository.
    """
    try:
        result = subprocess.run([git, *args], cwd=repo, check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise ListingError(f"git {' '.join(args)} failed in {repo}: {exc}") from exc
    return result.stdout


def tracked_files(repo: Path) -> dict[str, bytes]:
    """List the files git tracks in ``repo`` with the first bytes of each.

    A tracked file that is missing from the working tree is kept with empty contents, so its
    extension and location are still checked.

    Args:
        repo: The root of a git working tree.

    Returns:
        Each tracked path, with forward slashes, mapped to its first ``HEAD_SIZE`` bytes.

    Raises:
        ListingError: if git is not installed, the command fails, or ``repo`` is a subdirectory
            rather than the root (paths would then be relative to the wrong place).
    """
    git = shutil.which("git")
    if git is None:
        raise ListingError("git was not found on PATH")
    prefix = _git(git, repo, "rev-parse", "--show-prefix").decode("utf-8").strip()
    if prefix:
        raise ListingError(f"{repo} is inside a repository but is not its root ({prefix})")
    listing = _git(git, repo, "ls-files", "-z")
    files: dict[str, bytes] = {}
    for raw_path in listing.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        full = repo / path
        if full.is_file():
            with full.open("rb") as handle:
                files[path] = handle.read(HEAD_SIZE)
        else:
            files[path] = b""
    return files


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
        or the tracked files cannot be listed.
    """
    parser = argparse.ArgumentParser(
        description="Fail when a tracked document has no provenance record."
    )
    parser.add_argument("--repo", type=Path, default=Path(), help="repository root")
    parser.add_argument(
        "--registry", type=Path, help=f"registry file; default <repo>/{REGISTRY_PATH}"
    )
    args = parser.parse_args(argv)
    repo = cast(Path, args.repo)
    registry_path = cast(Path | None, args.registry) or repo / REGISTRY_PATH
    try:
        registry = read_registry(registry_path)
        files = tracked_files(repo)
    except (RegistryError, ListingError) as exc:
        print(f"provenance: {exc}", file=sys.stderr)
        return 2
    report = check(files, registry)
    for pattern in report.stale_patterns:
        print(f"provenance: warning: no tracked file matches {pattern!r}", file=sys.stderr)
    if report.violations:
        print(
            f"provenance: {len(report.violations)} tracked document(s) without a record "
            f"in {registry_path}:",
            file=sys.stderr,
        )
        for path in report.violations:
            print(path)
        return 1
    print(f"provenance: ok, {len(files)} tracked files, every document has a record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
