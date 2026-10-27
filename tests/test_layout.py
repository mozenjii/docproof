"""Tests for line grouping on straight and skewed pages."""

import math

import pytest

from docproof.extract.layout import lines, page_angle
from docproof.ocr import OcrPage, OcrToken, page_from_words


def rotated_token(
    text: str, cx: float, cy: float, width: float, height: float, degrees: float
) -> OcrToken:
    """A rectangle token rotated about its centre, as a skewed scan gives."""
    a = math.radians(degrees)
    cos, sin = math.cos(a), math.sin(a)
    corners = [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]
    points = [(cx + x * cos - y * sin, cy + x * sin + y * cos) for x, y in corners]
    return OcrToken(text, (points[0], points[1], points[2], points[3]), 0.9)


def test_straight_word_page_groups_rows_left_to_right() -> None:
    page = page_from_words(
        1000,
        1000,
        [
            ("Total:", (500, 100, 560, 120)),
            ("Invoice", (100, 102, 170, 122)),
            ("No.:", (175, 101, 210, 121)),
            ("A-1", (100, 140, 140, 160)),
        ],
    )
    assert [line.text for line in lines(page)] == ["Invoice No.: Total:", "A-1"]
    assert page_angle(page) == 0.0


def test_skewed_rows_do_not_chain_together() -> None:
    # Two columns, two rows, rotated 2 degrees: across 900 px the right column drifts ~31 px,
    # most of a 40 px row pitch, which is what made raw-box grouping merge the rows.
    tokens: list[OcrToken] = []
    for row, y in enumerate((300, 340)):
        for col, x in enumerate((100, 1000)):
            dy = (x - 550) * math.tan(math.radians(2))
            tokens.append(rotated_token(f"r{row}c{col}wide", x, y + dy, 120, 24, 2))
    page = OcrPage(1200, 800, tuple(tokens))
    assert math.degrees(page_angle(page)) == pytest.approx(2.0, abs=1e-6)
    assert [line.text for line in lines(page)] == ["r0c0wide r0c1wide", "r1c0wide r1c1wide"]


def test_empty_page_has_no_lines() -> None:
    assert lines(OcrPage(10, 10, ())) == []
