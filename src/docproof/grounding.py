"""Grounding: is a value printed on the page, and where?

A value the system proposes that cannot be found on the page is wrong or invented, so grounding is
the hallucination check, and the box it returns is the evidence a reviewer sees. The rules are in
``app/technical context/03_PIPELINE.md`` §2; in short:

- Values and OCR text are compared on a key with all whitespace removed, because the commonest OCR
  error on degraded scans is a dropped space (``NJ 07601`` read as ``NJ07601``), not a misread.
- A value may sit inside one long token (a whole OCR line) or span up to eight consecutive tokens on
  one text line.
- Every occurrence is reported, since the same number can be printed twice.

Deliberate boundary: exact matching only. Deciding which occurrence belongs to which field, fuzzy
matching for misread characters and rendering a typed value (a date, an amount) into its printed
forms are the caller's job.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

from docproof.errors import ContractError
from docproof.ocr import OcrPage, OcrToken, Rect

MAX_RUN = 8
"""The longest run of tokens a value may span."""


def key(text: str) -> str:
    """The comparison key: NFKC, casefolded, every whitespace character removed."""
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True, slots=True)
class Match:
    """One place a value occurs on the page.

    Attributes:
        tokens: Indexes of the tokens the value spans, in reading order.
        box: ``(left, top, right, bottom)`` covering just the value's characters, estimated by
            assuming equally wide characters within each token.
        confidence: The lowest engine confidence among those tokens.
        whole_tokens: True when the value is exactly those tokens, false when it is part of a
            longer token.
    """

    tokens: tuple[int, ...]
    box: tuple[int, int, int, int]
    confidence: float
    whole_tokens: bool


def same_line(a: OcrToken, b: OcrToken) -> bool:
    """Whether two tokens overlap vertically by at least half the shorter one's height."""
    _, top_a, _, bottom_a = a.rect
    _, top_b, _, bottom_b = b.rect
    overlap = min(bottom_a, bottom_b) - max(top_a, top_b)
    return overlap >= 0.5 * min(bottom_a - top_a, bottom_b - top_b)


def _sub_rect(rect: Rect, start: int, end: int, length: int) -> tuple[int, int, int, int]:
    """The part of a token's rect covering characters ``[start, end)`` of its ``length``."""
    left, top, right, bottom = rect
    width = right - left
    return (
        math.floor(left + width * start / length),
        math.floor(top),
        math.ceil(left + width * end / length),
        math.ceil(bottom),
    )


def _union(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _matches_in_run(
    page: OcrPage, keys: list[str], first: int, last: int, target: str
) -> list[tuple[int, Match]]:
    """Occurrences of ``target`` that start in token ``first`` and end in token ``last``."""
    offsets: list[int] = []
    total = 0
    for k in range(first, last + 1):
        offsets.append(total)
        total += len(keys[k])
    run_key = "".join(keys[first : last + 1])
    last_offset = offsets[-1]
    found: list[tuple[int, Match]] = []
    start = run_key.find(target)
    while start != -1:
        end = start + len(target)
        # The span must start inside the first token and end inside the last, so each
        # occurrence belongs to exactly one run: the shortest one, with no idle tokens.
        if start < len(keys[first]) and end > last_offset:
            rects: list[tuple[int, int, int, int]] = []
            whole = True
            for position, k in enumerate(range(first, last + 1)):
                length = len(keys[k])
                a = max(start - offsets[position], 0)
                b = min(end - offsets[position], length)
                whole = whole and a == 0 and b == length
                rects.append(_sub_rect(page.tokens[k].rect, a, b, length))
            tokens = tuple(range(first, last + 1))
            confidence = min(page.tokens[k].confidence for k in tokens)
            found.append((start, Match(tokens, _union(rects), confidence, whole)))
        start = run_key.find(target, start + 1)
    return found


def ground(value: str, page: OcrPage) -> tuple[Match, ...]:
    """Find every place ``value`` is printed on ``page``.

    Args:
        value: The text to look for, as printed (not a canonical form).
        page: The OCR page.

    Returns:
        Every match, in reading order of its first token and then by position in that token.
        An empty tuple means the value is not on the page.

    Raises:
        ContractError: ``empty_value`` if the value has no characters after whitespace removal.
    """
    target = key(value)
    if not target:
        raise ContractError("empty_value", "cannot ground a value with no characters")
    keys = [key(t.text) for t in page.tokens]
    ordered: list[tuple[int, int, Match]] = []
    for first in range(len(page.tokens)):
        if not keys[first]:
            continue
        last = first
        while last < len(page.tokens) and last - first < MAX_RUN:
            if last > first and not same_line(page.tokens[last - 1], page.tokens[last]):
                break
            for start, match in _matches_in_run(page, keys, first, last, target):
                ordered.append((first, start, match))
            last += 1
    ordered.sort(key=lambda item: (item[0], item[1]))
    return tuple(match for _, _, match in ordered)
