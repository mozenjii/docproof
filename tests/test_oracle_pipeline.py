"""Differential test: OCR loading and grounding against independently authored expected results.

The cases in ``tests/oracle/pipeline`` were written from ``03_PIPELINE.md`` alone, without seeing
the implementation. This module only converts between their JSON and the API. A failing case is
resolved by finding which side misread the contract, never by editing the case to agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from docproof.errors import ContractError
from docproof.grounding import ground
from docproof.ocr import load_ocr_page

CASES = Path(__file__).parent / "oracle" / "pipeline"


def _load(name: str) -> list[Any]:
    return json.loads((CASES / name).read_text(encoding="utf-8"))


OCR_CASES = _load("ocr.json")
GROUNDING_CASES = _load("grounding.json")


def _ground(args: dict[str, Any]) -> list[dict[str, Any]]:
    page = load_ocr_page(args["page"])
    return [
        {
            "tokens": list(m.tokens),
            "box": list(m.box),
            "confidence": m.confidence,
            "whole_tokens": m.whole_tokens,
        }
        for m in ground(args["value"], page)
    ]


@pytest.mark.parametrize("case", OCR_CASES, ids=[c["id"] for c in OCR_CASES])
def test_ocr_case(case: dict[str, Any]) -> None:
    if "expected_error" in case:
        with pytest.raises(ContractError) as info:
            load_ocr_page(case["args"]["raw"])
        assert info.value.code == case["expected_error"], case["why"]
    else:
        load_ocr_page(case["args"]["raw"])


@pytest.mark.parametrize("case", GROUNDING_CASES, ids=[c["id"] for c in GROUNDING_CASES])
def test_grounding_case(case: dict[str, Any]) -> None:
    if "expected_error" in case:
        with pytest.raises(ContractError) as info:
            _ground(case["args"])
        assert info.value.code == case["expected_error"], case["why"]
    else:
        assert _ground(case["args"]) == case["expected"], case["why"]


def test_case_ids_are_unique() -> None:
    ids = [c["id"] for c in OCR_CASES + GROUNDING_CASES]
    assert len(ids) == len(set(ids))
