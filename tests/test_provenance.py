"""Provenance guard tests: document detection, globs, the registry, git listing and the CLI."""

from __future__ import annotations

import base64
import gzip
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from docproof.provenance import (
    DOCUMENT_EXTENSIONS,
    DOCUMENT_SIGNATURES,
    HEAD_SIZE,
    REGISTRY_PATH,
    SHORT_SHA,
    ListingError,
    ProvenanceRecord,
    RegistryError,
    TrackedEntry,
    Violation,
    check,
    is_document_like,
    list_history,
    list_index,
    list_tree,
    load_registry,
    main,
    matches,
    parse_batch,
    parse_log,
    parse_ls_files,
    parse_ls_tree,
    read_registry,
)

# Built in two pieces so this source file does not itself carry a PDF header in its first bytes.
PDF_MAGIC = b"%PDF" + b"-"
PDF = PDF_MAGIC + b"1.7\n1 0 obj<<>>endobj\n% synthetic invoice\n%%EOF\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
SHA = "a" * 40

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
    "rtf",
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
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    b"\x1f\x8b",
    b"{\\rtf",
    b"version https://git-lfs.github.com/spec",
]

UBL_INVOICE = (
    b'<?xml version="1.0"?><Invoice xmlns="urn:oasis:names:specification:ubl:'
    b'schema:xsd:Invoice-2"><cbc:ID>INV-1</cbc:ID></Invoice>'
)


def synthetic(pattern: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        pattern=pattern, source="synthetic", origin="scripts/make_pages.py", license=None
    )


def entries(files: dict[str, bytes]) -> list[TrackedEntry]:
    return [TrackedEntry(path=path, head=head) for path, head in files.items()]


def valid_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "pattern": "assets/*.png",
        "source": "public",
        "origin": "https://example.org/sample.png",
        "license": "CC-BY-4.0",
    }
    entry.update(overrides)
    return entry


def synthetic_entry(pattern: str, origin: str = "scripts/make_logo.py") -> dict[str, object]:
    return {"pattern": pattern, "source": "synthetic", "origin": origin, "license": None}


# --- what counts as a document -------------------------------------------------------------


def test_extension_list_is_the_contract() -> None:
    assert set(EXTENSIONS) == DOCUMENT_EXTENSIONS
    assert set(SIGNATURES) == set(DOCUMENT_SIGNATURES)


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_every_document_extension_is_caught(extension: str) -> None:
    assert is_document_like(f"docs/scan.{extension}", b"plain text")


@pytest.mark.parametrize("extension", ["PDF", "Png", "JPEG", "TiFf", "DOCX", "RTF"])
def test_extensions_are_case_insensitive(extension: str) -> None:
    assert is_document_like(f"scan.{extension}", b"")


@pytest.mark.parametrize("signature", SIGNATURES)
def test_every_signature_is_caught_whatever_the_name(signature: bytes) -> None:
    assert is_document_like("notes/readme.txt", signature + b"\x00\x01\x02")


def test_bmp_needs_its_reserved_zero_bytes() -> None:
    assert is_document_like("notes/scan.dat", b"BM" + bytes(8) + b"\x01\x02")
    assert not is_document_like("notes/cars.txt", b"BMW service report, March")


def test_a_test_file_can_be_registered_as_synthetic() -> None:
    record = {
        "pattern": "tests/test_provenance.py",
        "source": "synthetic",
        "origin": "tests/test_provenance.py",
        "license": None,
    }
    assert load_registry([record])[0].origin == "tests/test_provenance.py"


def test_renamed_pdf_is_caught() -> None:
    assert is_document_like("invoice.txt", PDF)


def test_pdf_header_after_leading_bytes_is_caught() -> None:
    assert is_document_like("invoice.dat", b"\n" + PDF)
    assert is_document_like("invoice.dat", b"x" * (HEAD_SIZE - len(PDF_MAGIC)) + PDF_MAGIC)


def test_pdf_header_beyond_the_head_is_not_seen() -> None:
    assert not is_document_like("invoice.dat", b"x" * HEAD_SIZE + PDF)


@pytest.mark.parametrize(
    "head",
    [
        b"RIFF\x24\x00\x00\x00WEBPVP8 ",
        b"\x00\x00\x00\x18ftypheic" + b"\x00" * 8,
        b"\x00\x00\x00\x18ftypheix" + b"\x00" * 8,
        b"\x00\x00\x00\x18ftypmif1" + b"\x00" * 8,
        b"\x00\x00\x00\x18ftypmsf1" + b"\x00" * 8,
        b"\x00\x00\x00\x1cftypavif" + b"\x00" * 8,
    ],
    ids=["webp", "heic", "heix", "mif1", "msf1", "avif"],
)
def test_container_images_are_caught_by_content(head: bytes) -> None:
    assert is_document_like("photo.dat", head)


@pytest.mark.parametrize(
    "head",
    [
        UBL_INVOICE,
        b"\xef\xbb\xbf\n  <Invoice><ID>1</ID></Invoice>",
        b'<rsm:CrossIndustryInvoice xmlns:rsm="urn:un:unece:uncefact">',
        b"<!-- e-invoice --><Root>urn:oasis:names:specification:ubl:schema</Root>",
    ],
)
def test_xml_e_invoices_are_caught_by_content(head: bytes) -> None:
    assert is_document_like("export/invoice.xml", head)


@pytest.mark.parametrize(
    ("path", "head"),
    [
        ("src/docproof/provenance.py", b'"""Provenance'),
        ("README.md", b"# docproof\n"),
        ("pdf", b""),
        ("notes.pdf.txt", b"just text"),
        ("archive.tar", b"ustar"),
        ("short", b"%PD"),
        ("config.xml", b'<?xml version="1.0"?><project><name>docproof</name></project>'),
        ("doc.md", b"Write <Invoice> elements as described in the UBL guide."),
        ("riff.wav", b"RIFF\x24\x00\x00\x00WAVEfmt "),
        ("movie.mp4", b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8),
        ("invoice.xml", b""),
    ],
)
def test_ordinary_files_are_not_documents(path: str, head: bytes) -> None:
    assert not is_document_like(path, head)


def test_base64_documents_are_an_accepted_limit() -> None:
    assert not is_document_like("invoice.b64", base64.b64encode(PDF))


def test_gzip_compressed_document_is_caught() -> None:
    assert is_document_like("invoice.pdf.gz", gzip.compress(PDF, mtime=0))


def test_gitlink_is_always_a_document() -> None:
    assert is_document_like("vendor/lib", b"", gitlink=True)


@pytest.mark.parametrize(
    "path",
    [
        "data/notes.txt",
        "data/a/b/c.json",
        "data/.gitkeep",
        "data/provenance.json.bak",
        "Data/invoice.txt",
        "DATA/notes.txt",
        "DATA/provenance.json",
    ],
)
def test_anything_under_data_is_a_document(path: str) -> None:
    assert is_document_like(path, b"")


def test_the_registry_itself_is_exempt() -> None:
    assert not is_document_like(REGISTRY_PATH, b"[]\n")


def test_data_must_be_the_top_directory() -> None:
    assert not is_document_like("src/data/notes.txt", b"")
    assert not is_document_like("database/notes.txt", b"")
    assert not is_document_like("data", b"")


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
    raw = [valid_entry(), synthetic_entry("bench/pages/*.png", "python -m docproof.synth.dataset")]
    records = load_registry(raw)
    assert records == [
        ProvenanceRecord("assets/*.png", "public", "https://example.org/sample.png", "CC-BY-4.0"),
        ProvenanceRecord(
            "bench/pages/*.png", "synthetic", "python -m docproof.synth.dataset", None
        ),
    ]


def test_synthetic_record_may_carry_a_licence() -> None:
    entry = valid_entry(
        source="synthetic", origin="python -m docproof.synth.dataset", license="MIT"
    )
    assert load_registry([entry])[0].license == "MIT"


def test_empty_registry_is_valid() -> None:
    assert load_registry([]) == []


@pytest.mark.parametrize(
    "origin",
    [
        "python -m docproof.synth.dataset",
        "python -m docproof.synth.dataset --seed 7 --out data/synth",
        "scripts/make_pages.py",
        "scripts/make_pages.py --count 3",
        "bench/ocr_shortlist/pages.py",
        "src/docproof/synth/render.py",
    ],
)
def test_synthetic_origins_naming_this_repository_are_accepted(origin: str) -> None:
    assert load_registry([synthetic_entry("data/samples/*.pdf", origin)])[0].origin == origin


@pytest.mark.parametrize(
    "origin",
    [
        "x",
        "make_pages",
        "bench.make",
        "docproof.synth.dataset",
        "python -m docproof.",
        "python -m docproof",
        "python -m other.module",
        "python -m docproof.synth.dataset; curl example.org",
        "scripts/",
        "scripts",
        "scripts/../../elsewhere.py",
        "bench/./x.py",
        "src/other/x.py",
        "/scripts/x.py",
        "https://example.org/generator",
    ],
)
def test_synthetic_origins_outside_this_repository_are_refused(origin: str) -> None:
    with pytest.raises(RegistryError, match="must name a generator in this repository"):
        load_registry([synthetic_entry("data/samples/*.pdf", origin)])


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        ("**", "literal first segment"),
        ("**/*.pdf", "literal first segment"),
        ("*", "literal first segment"),
        ("*.pdf", "literal first segment"),
        ("*/x.pdf", "literal first segment"),
        ("?ata/**", "literal first segment"),
        ("[a-z]*/**", "literal first segment"),
        ("data*/x.pdf", "literal first segment"),
        ("/data/x.pdf", "must be relative"),
        ("data/../x.pdf", r"must not contain '\.\.'"),
        ("../x.pdf", r"must not contain '\.\.'"),
    ],
)
def test_patterns_that_could_cover_everything_are_refused(pattern: str, message: str) -> None:
    with pytest.raises(RegistryError, match=message):
        load_registry([synthetic_entry(pattern)])


@pytest.mark.parametrize("pattern", ["data/*.pdf", "data/samples/scan-?.png", "logo.png"])
def test_patterns_with_a_literal_first_segment_are_accepted(pattern: str) -> None:
    assert load_registry([synthetic_entry(pattern)])[0].pattern == pattern


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
    report = check(entries(files), [])
    assert [violation.path for violation in report.violations] == [
        "a/scan.png",
        "invoice.txt",
        "z.pdf",
    ]
    assert report.stale_patterns == ()


def test_covered_documents_pass() -> None:
    report = check(
        entries({"assets/logo.png": PNG, "README.md": b"# hi"}), [synthetic("assets/*.png")]
    )
    assert report.violations == ()
    assert report.stale_patterns == ()


def test_a_record_for_one_directory_does_not_cover_a_nested_one() -> None:
    files = {"assets/logo.png": PNG, "assets/raw/scan.png": PNG}
    report = check(entries(files), [synthetic("assets/*.png")])
    assert report.violations == (Violation("assets/raw/scan.png", None),)


def test_records_that_match_nothing_are_stale_in_registry_order() -> None:
    registry = [synthetic("old/*.pdf"), synthetic("assets/*.png"), synthetic("gone.tif")]
    report = check(entries({"assets/logo.png": PNG}), registry)
    assert report.violations == ()
    assert report.stale_patterns == ("old/*.pdf", "gone.tif")


def test_registry_file_needs_no_record() -> None:
    assert check(entries({REGISTRY_PATH: b"[]\n"}), []).violations == ()


def test_data_files_need_a_record() -> None:
    files = entries({"data/notes.txt": b"hello", REGISTRY_PATH: b"[]\n"})
    assert check(files, []).violations == (Violation("data/notes.txt", None),)
    assert check(files, [synthetic("data/*.txt")]).violations == ()


def test_a_path_is_a_document_if_any_of_its_blobs_is_and_names_the_first() -> None:
    history = [
        TrackedEntry(path="notes.txt", head=b"meeting notes", commit="c1"),
        TrackedEntry(path="notes.txt", head=PDF, commit="c2"),
        TrackedEntry(path="notes.txt", head=PNG, commit="c3"),
        TrackedEntry(path="README.md", head=b"# hi", commit="c1"),
    ]
    assert check(history, []).violations == (Violation("notes.txt", "c2"),)


def test_gitlinks_need_a_record() -> None:
    gitlink = [TrackedEntry(path="vendor", gitlink=True, oid=SHA)]
    assert check(gitlink, []).violations == (Violation("vendor", None),)
    assert check(gitlink, [synthetic("vendor")]).violations == ()


# --- parsing git output --------------------------------------------------------------------


def test_parse_ls_files_keeps_modes_and_drops_repeated_blobs() -> None:
    listing = (
        f"100644 {SHA} 0\tREADME.md\0".encode()
        + f"160000 {'b' * 40} 0\tvendor\0".encode()
        + f"100644 {'c' * 40} 1\tconflict.txt\0".encode()
        + f"100644 {'c' * 40} 2\tconflict.txt\0".encode()
        + f"100644 {'d' * 40} 3\tconflict.txt\0".encode()
    )
    assert parse_ls_files(listing) == [
        TrackedEntry(path="README.md", oid=SHA),
        TrackedEntry(path="vendor", gitlink=True, oid="b" * 40),
        TrackedEntry(path="conflict.txt", oid="c" * 40),
        TrackedEntry(path="conflict.txt", oid="d" * 40),
    ]


def test_parse_ls_files_keeps_undecodable_path_bytes() -> None:
    (entry,) = parse_ls_files(f"100644 {SHA} 0\t".encode() + b"caf\xe9.pdf\0")
    assert entry.path.encode("utf-8", errors="surrogateescape") == b"caf\xe9.pdf"


@pytest.mark.parametrize("listing", [b"100644 abc\tx\0", b"100644 abc 0 x\0"])
def test_parse_ls_files_refuses_unexpected_records(listing: bytes) -> None:
    with pytest.raises(ListingError, match="ls-files"):
        parse_ls_files(listing)


def test_parse_ls_tree() -> None:
    listing = f"100644 blob {SHA}\ta b/c.pdf\x00160000 commit {'b' * 40}\tvendor\0".encode()
    assert parse_ls_tree(listing) == [
        TrackedEntry(path="a b/c.pdf", oid=SHA),
        TrackedEntry(path="vendor", gitlink=True, oid="b" * 40),
    ]


def test_parse_ls_tree_refuses_unexpected_records() -> None:
    with pytest.raises(ListingError, match="ls-tree"):
        parse_ls_tree(b"100644 blob\0")


def log_record(old_mode: str, new_mode: str, old: str, new: str, status: str, path: str) -> bytes:
    return f":{old_mode} {new_mode} {old} {new} {status}\0{path}\0".encode()


def test_parse_log_names_the_first_commit_of_each_blob() -> None:
    zero = "0" * 40
    listing = (
        b"\x01c1\0\n"
        + log_record("000000", "100644", zero, SHA, "A", "invoice.pdf")
        + log_record("000000", "160000", zero, "b" * 40, "A", "vendor")
        + b"\x01c2\0\n"
        + log_record("100644", "000000", SHA, zero, "D", "invoice.pdf")
        + b"\x01c3\0\x01c4\0\n"
        + log_record("000000", "100644", zero, SHA, "A", "invoice.pdf")
        + log_record("000000", "100644", zero, SHA, "A", "copy.pdf")
    )
    assert parse_log(listing) == [
        TrackedEntry(path="invoice.pdf", commit="c1", oid=SHA),
        TrackedEntry(path="vendor", gitlink=True, commit="c1", oid="b" * 40),
        TrackedEntry(path="copy.pdf", commit="c4", oid=SHA),
    ]


@pytest.mark.parametrize(
    "listing",
    [
        log_record("000000", "100644", "0" * 40, SHA, "A", "before-any-commit.pdf"),
        b"\x01c1\0\nnot a change\0path\0",
        b"\x01c1\0\n:000000 100644 short A\0path\0",
        b"\x01c1\0\n:000000 100644 " + f"{'0' * 40} {SHA} A".encode(),
    ],
    ids=["no-commit", "not-a-change", "short-change", "no-path"],
)
def test_parse_log_refuses_unexpected_output(listing: bytes) -> None:
    with pytest.raises(ListingError, match="git log"):
        parse_log(listing)


def batch(*objects: tuple[str, str, bytes]) -> io.BytesIO:
    out = b"".join(
        f"{oid} {kind} {len(body)}\n".encode() + body + b"\n" for oid, kind, body in objects
    )
    return io.BytesIO(out)


def test_parse_batch_keeps_only_the_head() -> None:
    big = PDF + b"x" * (3 * HEAD_SIZE)
    stream = batch((SHA, "blob", big), ("b" * 40, "blob", b""))
    heads = parse_batch(stream, [SHA, "b" * 40])
    assert heads == {SHA: big[:HEAD_SIZE], "b" * 40: b""}


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (io.BytesIO(f"{SHA} missing\n".encode()), "cannot read object"),
        (io.BytesIO(b""), "cannot read object"),
        (batch(("b" * 40, "blob", b"x")), "cannot read object"),
        (io.BytesIO(f"{SHA} blob many\n".encode()), "cannot read object"),
        (batch((SHA, "tree", b"x")), "is a tree, not a blob"),
        (io.BytesIO(f"{SHA} blob 10\nshort".encode()), "cut short"),
        (io.BytesIO(f"{SHA} blob 2\nabX".encode()), "cut short"),
    ],
    ids=["missing", "empty", "wrong-object", "bad-size", "not-a-blob", "truncated", "no-newline"],
)
def test_parse_batch_refuses_what_it_cannot_read(stream: io.BytesIO, message: str) -> None:
    with pytest.raises(ListingError, match=message):
        parse_batch(stream, [SHA])


# --- the git edge and the CLI --------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    executable = shutil.which("git")
    assert executable is not None
    # Hooks and line-ending conversion from the machine's global configuration stay out of it.
    hooks = repo.parent / "no-hooks"
    hooks.mkdir(exist_ok=True)
    result = subprocess.run(
        [
            executable,
            "-c",
            f"core.hooksPath={hooks}",
            "-c",
            "core.autocrlf=false",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "protocol.file.allow=always",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


def new_repo(root: Path) -> Path:
    root.mkdir()
    git(root, "init", "--quiet")
    git(root, "config", "user.name", "Provenance Test")
    git(root, "config", "user.email", "provenance@example.invalid")
    return root


def add(repo: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(content)
        git(repo, "add", "--", name)


def commit(repo: Path, files: dict[str, bytes], message: str = "add") -> str:
    add(repo, files)
    git(repo, "commit", "--quiet", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.fail("git is required for the provenance integration tests")
    root = new_repo(tmp_path / "repo")
    add(root, {"assets/logo.png": PNG + b"rest of the image", "README.md": b"# test\n"})
    return root


def write_registry(repo: Path, entries: list[dict[str, object]]) -> Path:
    path = repo / REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return path


def heads(listed: list[TrackedEntry]) -> dict[str, bytes]:
    return {entry.path: entry.head for entry in listed}


def run(repo: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--repo", str(repo), *args])
    out, err = capsys.readouterr()
    return code, out, err


def test_list_index_reads_heads_from_git(repo: Path) -> None:
    assert heads(list_index(repo)) == {
        "README.md": b"# test\n",
        "assets/logo.png": PNG + b"rest of the image",
    }


def test_list_index_of_an_empty_repository(tmp_path: Path) -> None:
    assert list_index(new_repo(tmp_path / "empty")) == []


def test_untracked_documents_are_not_listed(repo: Path) -> None:
    (repo / "scratch.pdf").write_bytes(PDF)
    assert "scratch.pdf" not in heads(list_index(repo))


def test_staged_blob_is_judged_not_the_edited_working_copy(repo: Path) -> None:
    add(repo, {"notes.txt": PDF})
    (repo / "notes.txt").write_bytes(b"meeting notes\n")
    assert heads(list_index(repo))["notes.txt"] == PDF


def test_staged_blob_is_judged_when_the_working_copy_is_deleted(repo: Path) -> None:
    add(repo, {"notes.txt": PDF})
    (repo / "notes.txt").unlink()
    assert heads(list_index(repo))["notes.txt"] == PDF


def test_skip_worktree_does_not_hide_the_staged_blob(repo: Path) -> None:
    add(repo, {"notes.txt": PDF})
    git(repo, "update-index", "--skip-worktree", "notes.txt")
    (repo / "notes.txt").write_bytes(b"harmless\n")
    assert heads(list_index(repo))["notes.txt"] == PDF


def test_index_entry_whose_blob_is_missing_fails_closed(repo: Path) -> None:
    git(repo, "update-index", "--add", "--cacheinfo", f"100644,{'1' * 40},ghost.txt")
    with pytest.raises(ListingError, match="cannot read object"):
        list_index(repo)


def test_list_tree_reads_one_commit(repo: Path) -> None:
    first = commit(repo, {"invoice.txt": PDF})
    commit(repo, {"invoice.txt": b"text now\n"})
    assert heads(list_tree(repo, first))["invoice.txt"] == PDF
    assert heads(list_tree(repo, "HEAD"))["invoice.txt"] == b"text now\n"


@pytest.mark.parametrize("rev", ["no-such-branch", "--all", "HEAD:README.md"])
def test_list_tree_refuses_what_is_not_a_commit(repo: Path, rev: str) -> None:
    commit(repo, {})
    with pytest.raises(ListingError, match="rev-parse"):
        list_tree(repo, rev)


def test_list_history_names_the_first_commit(repo: Path) -> None:
    first = commit(repo, {"invoice.pdf": PDF})
    git(repo, "rm", "--quiet", "invoice.pdf")
    commit(repo, {})
    listed = list_history(repo)
    invoice = [entry for entry in listed if entry.path == "invoice.pdf"]
    assert invoice == [TrackedEntry(path="invoice.pdf", head=PDF, commit=first, oid=invoice[0].oid)]


def test_list_history_refuses_a_shallow_clone(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"a.txt": b"one"})
    commit(repo, {"b.txt": b"two"})
    shallow = tmp_path / "shallow"
    git(tmp_path, "clone", "--quiet", "--depth", "1", repo.as_uri(), str(shallow))
    with pytest.raises(ListingError, match="shallow clone"):
        list_history(shallow)


def test_list_history_sees_every_branch_and_merge(repo: Path) -> None:
    base = commit(repo, {})
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    git(repo, "checkout", "--quiet", "-b", "side")
    side = commit(repo, {"side.txt": PDF})
    git(repo, "checkout", "--quiet", branch)
    commit(repo, {"main.txt": b"main\n"})
    git(repo, "merge", "--quiet", "--no-edit", "side")
    git(repo, "branch", "--quiet", "-D", "side")
    found = {entry.path: entry.commit for entry in list_history(repo)}
    assert found["side.txt"] == side
    assert found["README.md"] == base


def test_list_history_of_a_repository_without_commits(tmp_path: Path) -> None:
    assert list_history(new_repo(tmp_path / "empty")) == []


def test_main_fails_then_passes_once_a_record_is_added(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_registry(repo, [])
    code, out, err = run(repo, capsys)
    assert code == 1
    assert out.splitlines() == ["assets/logo.png"]
    assert "without a record" in err

    write_registry(repo, [synthetic_entry("assets/*.png")])
    code, out, err = run(repo, capsys)
    assert code == 0
    assert "ok" in out
    assert err == ""


def test_main_warns_about_stale_records_but_passes(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = write_registry(repo, [synthetic_entry("assets/*.png"), synthetic_entry("old/*.pdf")])
    code, _, err = run(repo, capsys, "--registry", str(registry))
    assert code == 0
    assert "old/*.pdf" in err
    assert "warning" in err


def test_main_catches_a_renamed_pdf(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_registry(repo, [synthetic_entry("assets/*.png")])
    add(repo, {"invoice.txt": PDF})
    code, out, _ = run(repo, capsys)
    assert code == 1
    assert out.splitlines() == ["invoice.txt"]


# The scenarios of the review's proof of concept: each one once exited 0.


def test_poc_document_removed_at_head_is_caught_by_history(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_registry(repo, [synthetic_entry("assets/*.png")])
    first = commit(repo, {"invoice.pdf": PDF})
    git(repo, "rm", "--quiet", "invoice.pdf")
    commit(repo, {}, "remove")
    assert run(repo, capsys)[0] == 0
    code, out, _ = run(repo, capsys, "--history")
    assert code == 1
    assert out.splitlines() == [f"invoice.pdf (first in {first[:SHORT_SHA]})"]


def test_poc_committed_pdf_with_edited_working_copy_is_caught(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_registry(repo, [synthetic_entry("assets/*.png")])
    commit(repo, {"notes.txt": PDF})
    (repo / "notes.txt").write_bytes(b"meeting notes\n")
    for scope in ([], ["--rev", "HEAD"], ["--history"]):
        code, out, _ = run(repo, capsys, *scope)
        assert code == 1
        assert out.split()[0] == "notes.txt"


def test_poc_committed_pdf_missing_on_disk_is_caught(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_registry(repo, [synthetic_entry("assets/*.png")])
    commit(repo, {"notes.txt": PDF})
    (repo / "notes.txt").unlink()
    code, out, _ = run(repo, capsys)
    assert code == 1
    assert out.splitlines() == ["notes.txt"]


def test_poc_submodule_must_be_registered(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inner = new_repo(tmp_path / "inner")
    commit(inner, {"invoice.pdf": PDF})
    write_registry(repo, [synthetic_entry("assets/*.png")])
    git(repo, "submodule", "add", "--quiet", str(inner), "vendor")
    commit(repo, {})
    for scope in ([], ["--rev", "HEAD"], ["--history"]):
        code, out, _ = run(repo, capsys, *scope)
        assert code == 1
        assert out.split()[0] == "vendor"
    write_registry(repo, [synthetic_entry("assets/*.png"), synthetic_entry("vendor")])
    assert run(repo, capsys)[0] == 0


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("scan.dat", b"BM" + b"\x00" * 52),
        ("scan.bin", b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 16),
        ("letter.bin", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 504),
        ("invoice.pdf.gz", gzip.compress(PDF, mtime=0)),
        ("invoice.rtf", b"{\\rtf1\\ansi Invoice for Jane Roe}"),
        ("invoice.xml", UBL_INVOICE),
        ("invoice.dat", b"\n" + PDF),
        ("photo.dat", b"\x00\x00\x00\x18ftypheic" + b"\x00" * 16),
        ("large.bin", b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 9\n"),
    ],
)
def test_poc_document_formats_are_caught(
    repo: Path, capsys: pytest.CaptureFixture[str], name: str, content: bytes
) -> None:
    write_registry(repo, [synthetic_entry("assets/*.png")])
    commit(repo, {name: content})
    code, out, _ = run(repo, capsys)
    assert code == 1
    assert out.splitlines() == [name]


def test_poc_case_variant_data_directory_is_caught(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_registry(repo, [synthetic_entry("assets/*.png")])
    blob = git(repo, "hash-object", "-w", "README.md").strip()
    # Recorded straight into the index so the test holds on case-insensitive file systems too.
    git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},Data/invoice.txt")
    code, out, _ = run(repo, capsys)
    assert code == 1
    assert out.splitlines() == ["Data/invoice.txt"]


def test_poc_one_wildcard_record_is_refused(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_registry(repo, [synthetic_entry("**", "x")])
    commit(repo, {"invoice.pdf": PDF})
    code, out, err = run(repo, capsys)
    assert code == 2
    assert out == ""
    assert "literal first segment" in err


def test_history_ok_counts_paths(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_registry(repo, [synthetic_entry("assets/*.png")])
    commit(repo, {})
    code, out, err = run(repo, capsys, "--history")
    assert code == 0
    assert out == "provenance: ok, 2 paths across history, every document has a record\n"
    assert err == ""


def test_rev_and_history_are_exclusive(repo: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--repo", str(repo), "--rev", "HEAD", "--history"])


def test_bad_rev_is_exit_2(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_registry(repo, [])
    code, out, err = run(repo, capsys, "--rev", "no-such-branch")
    assert code == 2
    assert out == ""
    assert "rev-parse" in err


def test_missing_registry_is_exit_2(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(repo, capsys)
    assert code == 2
    assert "cannot read the registry" in err


def test_invalid_registry_is_exit_2(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_registry(repo, [valid_entry(source="scanned")])
    code, _, err = run(repo, capsys)
    assert code == 2
    assert "source must be" in err


def test_not_a_repository_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    write_registry(plain, [])
    for scope in ([], ["--rev", "HEAD"], ["--history"]):
        code, _, err = run(plain, capsys, *scope)
        assert code == 2
        assert "failed" in err


def test_not_a_repository_raises(tmp_path: Path) -> None:
    with pytest.raises(ListingError):
        list_index(tmp_path)


def test_subdirectory_of_a_repository_is_refused(repo: Path) -> None:
    with pytest.raises(ListingError, match="not its root"):
        list_index(repo / "assets")


def test_missing_git_fails_closed(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(ListingError, match="not found"):
        list_index(repo)
    write_registry(repo, [])
    assert main(["--repo", str(repo)]) == 2


def test_git_that_cannot_start_fails_closed(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise OSError("cannot execute")

    monkeypatch.setattr(subprocess, "run", refuse)
    with pytest.raises(ListingError, match="cannot execute"):
        list_index(repo)


@pytest.mark.parametrize("pattern", ["tests/**", "data/*", "data/**/x.png", "data/*/x.png", "a/?"])
def test_patterns_that_vouch_for_a_whole_directory_are_refused(pattern: str) -> None:
    record = {"pattern": pattern, "source": "synthetic", "origin": "scripts/x.py", "license": None}
    with pytest.raises(RegistryError, match=r"wildcards only in its last segment|literal part"):
        load_registry([record])


def test_a_file_name_wildcard_is_still_allowed() -> None:
    record = {
        "pattern": "data/synthetic/*.png",
        "source": "synthetic",
        "origin": "python -m docproof.synth.dataset",
        "license": None,
    }
    assert load_registry([record])[0].pattern == "data/synthetic/*.png"


def test_list_history_refuses_a_tag_that_points_at_a_blob(repo: Path) -> None:
    commit(repo, {"readme.txt": b"hello"})
    loose = repo.parent / "loose.pdf"
    loose.write_bytes(PDF)
    blob = git(repo, "hash-object", "-w", str(loose)).strip()
    git(repo, "tag", "loose-blob", blob)
    with pytest.raises(ListingError, match="points at a blob"):
        list_history(repo)
