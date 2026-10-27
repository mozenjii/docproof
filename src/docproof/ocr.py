"""OCR pages: the format the out-of-process OCR engine hands to the package.

The engine runs in its own environment and writes one JSON page per image (ADR-0003). This module
reads that JSON strictly, because it is the boundary between an untyped third-party runtime and a
typed pipeline: a box with three points or a confidence of 7 is a broken engine, not data.

Deliberate boundary: no OCR happens here. :func:`page_from_words` builds a perfect page from the
synthetic generator's word boxes, which is how the pipeline is tested without an engine.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from docproof.errors import ContractError

type Point = tuple[float, float]
type Quad = tuple[Point, Point, Point, Point]
type Rect = tuple[float, float, float, float]

_PAGE_KEYS = frozenset({"width", "height", "tokens"})
_TOKEN_KEYS = frozenset({"text", "box", "confidence"})


@dataclass(frozen=True, slots=True)
class OcrToken:
    """One piece of recognised text: a word or a whole line, depending on the engine.

    Attributes:
        text: The recognised text; never empty or whitespace-only.
        box: The engine's quadrilateral, four ``(x, y)`` points in page pixels.
        confidence: The engine's score in [0, 1].
    """

    text: str
    box: Quad
    confidence: float

    @property
    def rect(self) -> Rect:
        """Axis-aligned bounding rectangle ``(left, top, right, bottom)`` of the box."""
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True, slots=True)
class OcrPage:
    """Everything the engine read on one page image.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        tokens: Tokens in the engine's reading order.
    """

    width: int
    height: int
    tokens: tuple[OcrToken, ...]


def _fail(detail: str) -> ContractError:
    return ContractError("invalid_ocr", detail)


def _number(raw: object, what: str) -> float:
    # bool is an int subclass; a True coordinate is a broken engine, not 1.
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        raise _fail(f"{what} must be a finite number")
    return float(raw)


def _exact_keys(raw: object, keys: frozenset[str], what: str) -> Mapping[str, Any]:
    if not isinstance(raw, dict):
        raise _fail(f"{what} must be an object")
    obj = cast(dict[str, Any], raw)
    if frozenset(obj) != keys:
        raise _fail(f"{what} must have exactly the keys {sorted(keys)}")
    return obj


def _token(raw: object, index: int) -> OcrToken:
    what = f"tokens[{index}]"
    obj = _exact_keys(raw, _TOKEN_KEYS, what)
    text = obj["text"]
    if not isinstance(text, str) or not text.strip():
        raise _fail(f"{what}.text must be non-empty text")
    points = obj["box"]
    if not isinstance(points, list) or len(cast(list[object], points)) != 4:
        raise _fail(f"{what}.box must be four points")
    quad: list[Point] = []
    for p, point in enumerate(cast(list[object], points)):
        if not isinstance(point, list) or len(cast(list[object], point)) != 2:
            raise _fail(f"{what}.box[{p}] must be an [x, y] pair")
        x, y = cast(list[object], point)
        quad.append((_number(x, f"{what}.box[{p}][0]"), _number(y, f"{what}.box[{p}][1]")))
    confidence = _number(obj["confidence"], f"{what}.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise _fail(f"{what}.confidence must be in [0, 1]")
    token = OcrToken(text, (quad[0], quad[1], quad[2], quad[3]), confidence)
    left, top, right, bottom = token.rect
    if right <= left or bottom <= top:
        raise _fail(f"{what}.box has no area")
    return token


def load_ocr_page(raw: object) -> OcrPage:
    """Validate decoded OCR JSON and return the page.

    Args:
        raw: The page as decoded from JSON.

    Returns:
        The page.

    Raises:
        ContractError: ``invalid_ocr`` for any missing or extra key, wrong type, out-of-range
            confidence, box that is not four points, box without area, or empty text.
    """
    obj = _exact_keys(raw, _PAGE_KEYS, "page")
    sizes: list[int] = []
    for key in ("width", "height"):
        value = obj[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _fail(f"page.{key} must be a positive integer")
        sizes.append(value)
    tokens = obj["tokens"]
    if not isinstance(tokens, list):
        raise _fail("page.tokens must be a list")
    return OcrPage(
        sizes[0], sizes[1], tuple(_token(t, i) for i, t in enumerate(cast(list[object], tokens)))
    )


def dump_ocr_page(page: OcrPage) -> dict[str, Any]:
    """Encode a page as JSON-ready data that :func:`load_ocr_page` accepts.

    Args:
        page: The page.

    Returns:
        The JSON-ready form.
    """
    return {
        "width": page.width,
        "height": page.height,
        "tokens": [
            {"text": t.text, "box": [list(p) for p in t.box], "confidence": t.confidence}
            for t in page.tokens
        ],
    }


def page_from_words(
    width: int, height: int, words: Iterable[tuple[str, tuple[int, int, int, int]]]
) -> OcrPage:
    """Build a perfect OCR page from known words and their rectangles.

    Used with the synthetic generator's word boxes to test the pipeline without an OCR engine.

    Args:
        width: Page width in pixels.
        height: Page height in pixels.
        words: ``(text, (left, top, right, bottom))`` pairs in reading order.

    Returns:
        A page with one token per word and confidence 1.
    """
    tokens = tuple(
        OcrToken(text, ((left, top), (right, top), (right, bottom), (left, bottom)), 1.0)
        for text, (left, top, right, bottom) in words
    )
    return OcrPage(width, height, tokens)
