"""Text lines from OCR tokens, whatever the engine's granularity, on straight or skewed pages.

Engines return words (docTR) or whole lines and cells (PaddleOCR). Extraction wants text lines in
reading order. On a skewed scan a row of text drifts tens of pixels across the page, so comparing
raw boxes chains neighbouring rows together. Lines are therefore found in deskewed coordinates: the
page angle is the median angle of the engine's text quadrilaterals, token centres are rotated back
by it, and tokens whose deskewed centres are within half a text height of each other share a line.

Deliberate boundary: geometry only. Nothing here knows what a caption or a field is. Pages whose
boxes carry no angle (axis-aligned word boxes) are treated as straight.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from docproof.ocr import OcrPage, OcrToken


@dataclass(frozen=True, slots=True)
class Line:
    """Tokens that sit on one text line.

    Attributes:
        tokens: Indexes into the page's tokens, left to right in deskewed coordinates.
        text: The tokens' texts joined with single spaces.
        y: Deskewed vertical centre of the line.
        left: Deskewed left edge of the leftmost token.
        height: Median text height of the tokens.
    """

    tokens: tuple[int, ...]
    text: str
    y: float
    left: float
    height: float


def _angle(token: OcrToken) -> float:
    (x0, y0), (x1, y1) = token.box[0], token.box[1]
    return math.atan2(y1 - y0, x1 - x0)


def _height(token: OcrToken) -> float:
    (x1, y1), (x2, y2) = token.box[1], token.box[2]
    return math.hypot(x2 - x1, y2 - y1)


def page_angle(page: OcrPage) -> float:
    """Estimate the page skew in radians from tokens at least twice as wide as they are tall.

    Args:
        page: The OCR page.

    Returns:
        The median text angle, or 0.0 when no token is wide enough to have a reliable angle.
    """
    wide = [_angle(t) for t in page.tokens if t.rect[2] - t.rect[0] >= 2 * _height(t)]
    return statistics.median(wide) if wide else 0.0


@dataclass(frozen=True, slots=True)
class _Placed:
    index: int
    x: float
    y: float
    left: float
    height: float


def _place(page: OcrPage, angle: float) -> list[_Placed]:
    """Rotate each token's centre and left edge back by ``angle`` about the page centre."""
    cx, cy = page.width / 2, page.height / 2
    cos, sin = math.cos(-angle), math.sin(-angle)
    placed: list[_Placed] = []
    for index, token in enumerate(page.tokens):
        xs = [p[0] for p in token.box]
        ys = [p[1] for p in token.box]
        mx, my = sum(xs) / 4, sum(ys) / 4
        dx, dy = mx - cx, my - cy
        x = cx + dx * cos - dy * sin
        y = cy + dx * sin + dy * cos
        half_width = math.hypot(
            token.box[1][0] - token.box[0][0], token.box[1][1] - token.box[0][1]
        )
        placed.append(_Placed(index, x, y, x - half_width / 2, max(_height(token), 1.0)))
    return placed


def lines(page: OcrPage) -> list[Line]:
    """Group a page's tokens into text lines, top to bottom.

    Args:
        page: The OCR page.

    Returns:
        The lines, ordered by deskewed vertical centre; tokens within a line left to right.
    """
    placed = sorted(_place(page, page_angle(page)), key=lambda p: (p.y, p.x))
    groups: list[list[_Placed]] = []
    for token in placed:
        if groups:
            current = groups[-1]
            line_y = statistics.fmean(p.y for p in current)
            tolerance = 0.5 * min(token.height, statistics.median(p.height for p in current))
            if abs(token.y - line_y) <= tolerance:
                current.append(token)
                continue
        groups.append([token])
    built: list[Line] = []
    for group in groups:
        group.sort(key=lambda p: p.x)
        built.append(
            Line(
                tokens=tuple(p.index for p in group),
                text=" ".join(page.tokens[p.index].text.strip() for p in group),
                y=statistics.fmean(p.y for p in group),
                left=min(p.left for p in group),
                height=statistics.median(p.height for p in group),
            )
        )
    return built
