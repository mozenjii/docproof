"""Tests for the OCR page boundary and for grounding values on a page."""

import copy
import json
from typing import Any

import pytest

from docproof.errors import ContractError
from docproof.grounding import Match, ground, key, same_line
from docproof.ocr import OcrPage, dump_ocr_page, load_ocr_page, page_from_words

GOOD: dict[str, Any] = {
    "width": 100,
    "height": 50,
    "tokens": [
        {"text": "Total 12.50", "box": [[0, 0], [60, 0], [60, 10], [0, 10]], "confidence": 0.9}
    ],
}


def broken(path: tuple[Any, ...], value: object) -> dict[str, Any]:
    raw = copy.deepcopy(GOOD)
    target: Any = raw
    for step in path[:-1]:
        target = target[step]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return raw


def test_a_valid_page_round_trips_through_json() -> None:
    page = load_ocr_page(GOOD)
    assert page.tokens[0].rect == (0, 0, 60, 10)
    assert load_ocr_page(json.loads(json.dumps(dump_ocr_page(page)))) == page


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("width",), 0),
        (("height",), True),
        (("width",), 10.5),
        (("tokens",), {}),
        (("extra",), 1),
        (("tokens", 0, "text"), "   "),
        (("tokens", 0, "text"), 7),
        (("tokens", 0, "confidence"), 1.5),
        (("tokens", 0, "confidence"), True),
        (("tokens", 0, "box"), [[0, 0], [1, 0], [1, 1]]),
        (("tokens", 0, "box"), [[0, 0], [0, 0], [0, 0], [0, 0]]),
        (("tokens", 0, "box", 0), [0, "0"]),
        (("tokens", 0, "box", 0), [0]),
        (("tokens", 0, "confidence"), KeyError),
    ],
)
def test_broken_pages_are_refused(path: tuple[Any, ...], value: object) -> None:
    with pytest.raises(ContractError) as info:
        load_ocr_page(broken(path, value))
    assert info.value.code == "invalid_ocr"


def test_not_an_object_is_refused() -> None:
    with pytest.raises(ContractError):
        load_ocr_page([GOOD])
    with pytest.raises(ContractError):
        load_ocr_page({**GOOD, "tokens": ["x"]})


def line_page(*words: tuple[str, tuple[int, int, int, int]]) -> OcrPage:
    return page_from_words(1000, 1000, words)


def test_key_removes_whitespace_and_folds() -> None:
    assert key(" NJ  07601\t") == "nj07601"
    full_width_inv = "".join(map(chr, (0xFF29, 0xFF2E, 0xFF36)))
    assert key(full_width_inv) == "inv"  # full-width letters fold under NFKC


def test_merged_token_still_grounds() -> None:
    page = line_page(("NJ07601", (0, 0, 70, 20)))
    assert ground("NJ 07601", page) == (Match((0,), (0, 0, 70, 20), 1.0, True),)


def test_substring_box_is_interpolated() -> None:
    page = line_page(("Total12.50", (0, 0, 100, 10)))
    (match,) = ground("12.50", page)
    assert match.box == (50, 0, 100, 10)
    assert not match.whole_tokens


def test_runs_stay_on_one_line_and_in_reading_order() -> None:
    page = line_page(
        ("Riverton,", (0, 0, 80, 20)), ("NJ", (0, 40, 20, 60)), ("07601", (30, 40, 80, 60))
    )
    assert ground("Riverton, NJ", page) == ()
    assert ground("NJ 07601", page)[0].tokens == (1, 2)


def test_half_height_overlap_is_the_line_rule() -> None:
    a = line_page(("x", (0, 0, 10, 20))).tokens[0]
    b_same = line_page(("y", (0, 10, 10, 30))).tokens[0]  # overlap 10 = half of 20
    b_other = line_page(("y", (0, 11, 10, 31))).tokens[0]  # overlap 9
    assert same_line(a, b_same)
    assert not same_line(a, b_other)


def test_repeated_values_are_all_reported_in_reading_order() -> None:
    page = line_page(("12.50", (0, 0, 50, 10)), ("x", (0, 30, 5, 40)), ("12.50", (0, 60, 50, 70)))
    assert [m.tokens for m in ground("12.50", page)] == [(0,), (2,)]


def test_no_idle_tokens_and_confidence_is_the_minimum() -> None:
    words = page_from_words(1000, 1000, [("A", (0, 0, 10, 10)), ("B", (12, 0, 22, 10))])
    page = OcrPage(
        words.width,
        words.height,
        (words.tokens[0], type(words.tokens[1])(words.tokens[1].text, words.tokens[1].box, 0.4)),
    )
    assert ground("b", page) == (Match((1,), (12, 0, 22, 10), 0.4, True),)
    assert ground("ab", page)[0].confidence == 0.4


def test_runs_are_capped_at_eight_tokens() -> None:
    nine = [(c, (i * 20, 0, i * 20 + 10, 10)) for i, c in enumerate("abcdefghi")]
    page = line_page(*nine)
    assert ground("abcdefgh", page)
    assert ground("abcdefghi", page) == ()


def test_empty_value_is_an_error_and_absence_is_empty() -> None:
    with pytest.raises(ContractError) as info:
        ground("  ", line_page(("x", (0, 0, 10, 10))))
    assert info.value.code == "empty_value"
    assert ground("y", line_page(("x", (0, 0, 10, 10)))) == ()


def test_a_token_with_no_characters_never_starts_a_match() -> None:
    blank = "".join(map(chr, (0x3000,)))  # an ideographic space: whitespace only
    page = line_page((blank, (0, 0, 10, 10)), ("x", (12, 0, 22, 10)))
    assert [m.tokens for m in ground("x", page)] == [(1,)]
