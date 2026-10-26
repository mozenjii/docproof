"""Provenance guard tests: what counts as a document, glob matching, the registry and the CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from docproof.provenance import (
    DOCUMENT_EXTENSIONS,
    DOCUMENT_SIGNATURES,
    REGISTRY_PATH,
    ListingError,
    ProvenanceRecord,
    RegistryError,
    check,
    is_document_like,
    load_registry,
    main,
    matches,
    read_registry,
    tracked_files,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3"

EXTENSIONS = [
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
]

SIGNATURES = [
    b"%PDF",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"II*\x00",
    b"MM\x00*",
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",
]


def synthetic(pattern: str) -> ProvenanceRecord:
    return ProvenanceRecord(pattern=pattern, source="synthetic", origin="make_pages", license=None)


def valid_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "pattern": "assets/*.png",
        "source": "public",
        "origin": "https://example.org/sample.png",
        "license": "CC-BY-4.0",
    }
    entry.update(overrides)
    return entry


# --- what counts as a document -------------------------------------------------------------


def test_extension_list_is_the_contract() -> None:
    assert set(EXTENSIONS) == DOCUMENT_EXTENSIONS
    assert set(SIGNATURES) == set(DOCUMENT_SIGNATURES)


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_every_document_extension_is_caught(extension: str) -> None:
    assert is_document_like(f"docs/scan.{extension}", b"plain text")


@pytest.mark.parametrize("extension", ["PDF", "Png", "JPEG", "TiFf", "DOCX"])
def test_extensions_are_case_insensitive(extension: str) -> None:
    assert is_document_like(f"scan.{extension}", b"")


@pytest.mark.parametrize("signature", SIGNATURES)
def test_every_signature_is_caught_whatever_the_name(signature: bytes) -> None:
    assert is_document_like("notes/readme.txt", signature + b"\x00\x01\x02")


def test_renamed_pdf_is_caught() -> None:
    assert is_document_like("invoice.txt", PDF)


@pytest.mark.parametrize(
    ("path", "head"),
    [
        ("src/docproof/provenance.py", b'"""Provenance'),
        ("README.md", b"# docproof\n"),
        ("pdf", b""),
        ("notes.pdf.txt", b"just text"),
        ("archive.tar", b"ustar"),
        ("short", b"%PD"),
    ],
)
def test_ordinary_files_are_not_documents(path: str, head: bytes) -> None:
    assert not is_document_like(path, head)


@pytest.mark.parametrize(
    "path", ["data/notes.txt", "data/a/b/c.json", "data/.gitkeep", "data/provenance.json.bak"]
)
def test_anything_under_data_is_a_document(path: str) -> None:
    assert is_document_like(path, b"")


def test_the_registry_itself_is_exempt() -> None:
    assert not is_document_like(REGISTRY_PATH, b"[]\n")


def test_data_must_be_the_top_directory() -> None:
    assert not is_document_like("src/data/notes.txt", b"")
    assert not is_document_like("database/notes.txt", b"")


# --- glob semantics ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("logo.png", "logo.png", True),
        ("logo.png", "assets/logo.png", False),
        ("assets/logo.png", "assets/logo.png", True),
        ("assets/logo.png", "assets/logo.PNG", False),
        ("*.png", "logo.png", True),
        ("*.png", "assets/logo.png", False),
        ("assets/*.png", "assets/logo.png", True),
        ("assets/*.png", "assets/icons/logo.png", False),
        ("assets/*", "assets/icons/logo.png", False),
        ("assets/*/*.png", "assets/icons/logo.png", True),
        ("**/*.png", "logo.png", True),
        ("**/*.png", "a/b/c/logo.png", True),
        ("**/*.png", "a/b/c/logo.jpg", False),
        ("assets/**", "assets/logo.png", True),
        ("assets/**", "assets/a/b/logo.png", True),
        ("assets/**", "other/logo.png", False),
        ("assets/**/logo.png", "assets/logo.png", True),
        ("assets/**/logo.png", "assets/a/b/logo.png", True),
        ("assets/**/logo.png", "assets/a/b/other.png", False),
        ("a/**/b/**/c.pdf", "a/x/b/y/z/c.pdf", True),
        ("a/**/b/**/c.pdf", "a/x/y/c.pdf", False),
        ("**", "anything/at/all.pdf", True),
        ("page-?.png", "page-1.png", True),
        ("page-?.png", "page-10.png", False),
        ("page-[0-9].png", "page-7.png", True),
        ("page-[!0-9].png", "page-7.png", False),
        ("assets", "assets/logo.png", False),
    ],
)
def test_glob_matching(pattern: str, path: str, expected: bool) -> None:
    assert matches(pattern, path) is expected


# --- registry validation -------------------------------------------------------------------


def test_valid_registry_loads_in_order() -> None:
    raw = [
        valid_entry(),
        {"pattern": "bench/**", "source": "synthetic", "origin": "bench.make", "license": None},
    ]
    records = load_registry(raw)
    assert records == [
        ProvenanceRecord("assets/*.png", "public", "https://example.org/sample.png", "CC-BY-4.0"),
        ProvenanceRecord("bench/**", "synthetic", "bench.make", None),
    ]


def test_synthetic_record_may_carry_a_licence() -> None:
    entry = valid_entry(source="synthetic", origin="docproof.synth.dataset", license="MIT")
    assert load_registry([entry])[0].license == "MIT"


def test_empty_registry_is_valid() -> None:
    assert load_registry([]) == []


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"pattern": "x"}, "JSON array"),
        ("[]", "JSON array"),
        (["not an object"], "expected an object"),
        ([valid_entry(extra=1)], "unknown keys"),
        ([{k: v for k, v in valid_entry().items() if k != "license"}], "missing keys license"),
        ([{k: v for k, v in valid_entry().items() if k != "pattern"}], "missing keys pattern"),
        ([valid_entry(pattern="")], "'pattern' must be a non-empty string"),
        ([valid_entry(pattern=3)], "'pattern' must be a non-empty string"),
        ([valid_entry(source="scanned")], "source must be synthetic or public"),
        ([valid_entry(source=None)], "'source' must be a non-empty string"),
        ([valid_entry(origin="")], "'origin' must be a non-empty string"),
        ([valid_entry(origin="http://example.org/a.png")], "https:// URL"),
        ([valid_entry(origin="example.org/a.png")], "https:// URL"),
        ([valid_entry(license=None)], "needs a licence"),
        ([valid_entry(license="")], "needs a licence"),
        ([valid_entry(license=7)], "'license' must be a string or null"),
    ],
)
def test_registry_errors(raw: object, message: str) -> None:
    with pytest.raises(RegistryError, match=message):
        load_registry(raw)


def test_registry_error_names_the_record() -> None:
    with pytest.raises(RegistryError, match="record 1"):
        load_registry([valid_entry(), valid_entry(source="leaked")])


def test_registry_error_is_a_value_error() -> None:
    assert issubclass(RegistryError, ValueError)


def test_read_registry_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps([valid_entry()]), encoding="utf-8")
    assert read_registry(path)[0].pattern == "assets/*.png"


def test_missing_registry_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="cannot read"):
        read_registry(tmp_path / "absent.json")


def test_malformed_registry_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    path.write_text("[{", encoding="utf-8")
    with pytest.raises(RegistryError, match="not valid JSON"):
        read_registry(path)


def test_committed_registry_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / REGISTRY_PATH).read_text(encoding="utf-8")
    assert text.endswith("\n")
    load_registry(json.loads(text))


# --- check ---------------------------------------------------------------------------------


def test_uncovered_documents_are_violations_sorted() -> None:
    files = {"z.pdf": PDF, "a/scan.png": PNG, "src/x.py": b"import os", "invoice.txt": PDF}
    report = check(files, [])
    assert report.violations == ("a/scan.png", "invoice.txt", "z.pdf")
    assert report.stale_patterns == ()


def test_covered_documents_pass() -> None:
    files = {"assets/logo.png": PNG, "README.md": b"# hi"}
    report = check(files, [synthetic("assets/*.png")])
    assert report.violations == ()
    assert report.stale_patterns == ()


def test_a_record_for_one_directory_does_not_cover_a_nested_one() -> None:
    files = {"assets/logo.png": PNG, "assets/raw/scan.png": PNG}
    report = check(files, [synthetic("assets/*.png")])
    assert report.violations == ("assets/raw/scan.png",)


def test_records_that_match_nothing_are_stale_in_registry_order() -> None:
    files = {"assets/logo.png": PNG}
    registry = [synthetic("old/*.pdf"), synthetic("assets/*.png"), synthetic("gone.tif")]
    report = check(files, registry)
    assert report.violations == ()
    assert report.stale_patterns == ("old/*.pdf", "gone.tif")


def test_registry_file_needs_no_record() -> None:
    assert check({REGISTRY_PATH: b"[]\n"}, []).violations == ()


def test_data_files_need_a_record() -> None:
    files = {"data/notes.txt": b"hello", REGISTRY_PATH: b"[]\n"}
    assert check(files, []).violations == ("data/notes.txt",)
    assert check(files, [synthetic("data/*.txt")]).violations == ()


# --- the git edge and the CLI --------------------------------------------------------------


def git(repo: Path, *args: str) -> None:
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run([executable, *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.fail("git is required for the provenance integration tests")
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "--quiet")
    git(root, "config", "user.name", "Provenance Test")
    git(root, "config", "user.email", "provenance@example.invalid")
    (root / "assets").mkdir()
    (root / "assets" / "logo.png").write_bytes(PNG + b"rest of the image")
    (root / "README.md").write_bytes(b"# test\n")
    git(root, "add", "assets/logo.png", "README.md")
    return root


def synthetic_entry(pattern: str) -> dict[str, object]:
    return {"pattern": pattern, "source": "synthetic", "origin": "x", "license": None}


def write_registry(repo: Path, entries: list[dict[str, object]]) -> Path:
    path = repo / REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return path


def test_tracked_files_reads_heads(repo: Path) -> None:
    files = tracked_files(repo)
    assert files == {"README.md": b"# test\n", "assets/logo.png": PNG}


def test_tracked_file_missing_from_disk_is_still_listed(repo: Path) -> None:
    (repo / "assets" / "logo.png").unlink()
    assert tracked_files(repo)["assets/logo.png"] == b""


def test_untracked_documents_are_not_listed(repo: Path) -> None:
    (repo / "scratch.pdf").write_bytes(PDF)
    assert "scratch.pdf" not in tracked_files(repo)


def test_main_fails_then_passes_once_a_record_is_added(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_registry(repo, [])
    assert main(["--repo", str(repo)]) == 1
    out, err = capsys.readouterr()
    assert out.splitlines() == ["assets/logo.png"]
    assert "without a record" in err

    record: dict[str, object] = {
        "pattern": "assets/*.png",
        "source": "synthetic",
        "origin": "scripts/make_logo.py",
        "license": None,
    }
    write_registry(repo, [record])
    assert main(["--repo", str(repo)]) == 0
    out, err = capsys.readouterr()
    assert "ok" in out
    assert err == ""


def test_main_warns_about_stale_records_but_passes(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = write_registry(
        repo,
        [synthetic_entry("assets/*.png"), synthetic_entry("old/*.pdf")],
    )
    assert main(["--repo", str(repo), "--registry", str(registry)]) == 0
    _, err = capsys.readouterr()
    assert "old/*.pdf" in err
    assert "warning" in err


def test_main_catches_a_renamed_pdf(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_registry(repo, [synthetic_entry("assets/*.png")])
    (repo / "invoice.txt").write_bytes(PDF)
    git(repo, "add", "invoice.txt")
    assert main(["--repo", str(repo)]) == 1
    out, _ = capsys.readouterr()
    assert out.splitlines() == ["invoice.txt"]


def test_missing_registry_is_exit_2(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--repo", str(repo)]) == 2
    _, err = capsys.readouterr()
    assert "cannot read the registry" in err


def test_invalid_registry_is_exit_2(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_registry(repo, [valid_entry(source="scanned")])
    assert main(["--repo", str(repo)]) == 2
    _, err = capsys.readouterr()
    assert "source must be" in err


def test_not_a_repository_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    write_registry(plain, [])
    assert main(["--repo", str(plain)]) == 2
    _, err = capsys.readouterr()
    assert "failed" in err


def test_not_a_repository_raises(tmp_path: Path) -> None:
    with pytest.raises(ListingError):
        tracked_files(tmp_path)


def test_subdirectory_of_a_repository_is_refused(repo: Path) -> None:
    with pytest.raises(ListingError, match="not its root"):
        tracked_files(repo / "assets")


def test_missing_git_fails_closed(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(ListingError, match="not found"):
        tracked_files(repo)
    write_registry(repo, [])
    assert main(["--repo", str(repo)]) == 2
