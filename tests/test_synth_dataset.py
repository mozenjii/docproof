"""Dataset writer tests: the files exist, decode, and are identical across processes."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from docproof.labels import decode_labels
from docproof.synth.dataset import main, write_dataset
from docproof.synth.templates import TEMPLATES

FEW = ["inv-b", "bol-a"]


def test_writes_labels_images_words_and_manifest(tmp_path: Path) -> None:
    count = write_dataset(tmp_path, per_template=1, seed=5, levels=[0, 3], template_ids=FEW)
    assert count == len(FEW)
    entries = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert len(entries) == count
    for entry in entries:
        folder = tmp_path / entry["split"]
        doc_type, _ = decode_labels(
            json.loads((folder / f"{entry['doc_id']}.labels.json").read_text())
        )
        assert doc_type.value == entry["doc_type"]
        for level in (0, 3):
            assert (folder / f"{entry['doc_id']}.L{level}.png").stat().st_size > 0
            words = json.loads((folder / f"{entry['doc_id']}.L{level}.words.json").read_text())
            assert words
            assert {"text", "box", "field", "row"} <= set(words[0])


def _digest(folder: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(folder).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def test_output_is_identical_across_processes(tmp_path: Path) -> None:
    # Separate interpreters catch anything seeded from per-process state such as str hashing.
    for name in ("a", "b"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "docproof.synth.dataset",
                "--out",
                str(tmp_path / name),
                "--per-template",
                "1",
                "--levels",
                "2",
                "--templates",
                *FEW,
            ],
            check=True,
            capture_output=True,
        )
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")


def test_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="per_template"):
        write_dataset(tmp_path, 0, 1, [0])
    with pytest.raises(ValueError, match="levels"):
        write_dataset(tmp_path, 1, 1, [9])
    with pytest.raises(KeyError, match="no template"):
        write_dataset(tmp_path, 1, 1, [0], ["inv-z"])
    assert len(TEMPLATES) > len(FEW)


def test_cli_entry_point(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--out", str(tmp_path), "--per-template", "1", "--levels", "0", "--templates", "rc-a"]
    assert main(args) == 0
    assert "wrote" in capsys.readouterr().out
