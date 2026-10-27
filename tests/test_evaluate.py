"""Tests for the evaluation command on a small generated dataset."""

import json
from pathlib import Path

import pytest

from docproof.evaluate import evaluate, main, table
from docproof.synth.dataset import write_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("eval")
    write_dataset(out, per_template=1, seed=3, levels=[0, 2], template_ids=["inv-a", "inv-h"])
    return out


def test_perfect_ocr_scores_level_zero_only(dataset: Path) -> None:
    results = evaluate(dataset, "perfect", tolerance=0, thresholds=[0.9])
    assert set(results["groups"]) == {"train L0 @0.9", "test L0 @0.9"}
    train = results["groups"]["train L0 @0.9"]
    assert train["accepted_precision"]["estimate"] == 1.0
    assert train["documents"] == 1


def test_missing_recorded_ocr_is_an_error_not_a_skip(dataset: Path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluate(dataset, "no-such-engine", tolerance=0, thresholds=[0.9])


def test_cli_writes_json_and_a_table(dataset: Path, tmp_path: Path) -> None:
    args = ["--dataset", str(dataset), "--ocr", "perfect", "--tolerance", "0"]
    assert main([*args, "--thresholds", "0.5", "0.9", "--out", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["settings"]["thresholds"] == [0.5, 0.9]
    rendered = table(saved)
    assert rendered.startswith("| group |")
    assert "train L0 @0.5" in rendered
