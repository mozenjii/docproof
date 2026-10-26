"""Draw a sampled document as a page image and record the true box of every printed word.

The word boxes are ground truth for grounding: they say exactly where each field's value was
printed, so a grounding step can be scored without hand-drawn boxes.

Deliberate boundary: layout only. The font is Pillow's bundled one so a page renders identically
on every platform; layout variety comes from the template (font size, header columns, captions).
Scan degradation is :func:`degrade`, applied to a finished page.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from docproof.schema import DocumentType
from docproof.synth.sample import PrintedField, PrintedLineItem, SyntheticDocument

PAGE_SIZE = (1654, 2339)
"""A4 at 200 DPI, in pixels."""

MARGIN = 120

_TITLES = {
    DocumentType.INVOICE: "INVOICE",
    DocumentType.FREIGHT_INVOICE: "FREIGHT INVOICE",
    DocumentType.BILL_OF_LADING: "BILL OF LADING",
    DocumentType.RATE_CONFIRMATION: "RATE CONFIRMATION",
}
_PRIMARY_PARTY = ("vendor", "shipper", "broker")
_SECONDARY_PARTY = ("buyer", "consignee", "carrier")
_TOTALS = ("subtotal", "tax_total", "total", "rate")

type Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class WordBox:
    """One printed word and where it is.

    Attributes:
        text: The word as printed.
        box: ``(left, top, right, bottom)`` in page pixels.
        field: Schema field the word belongs to, or ``None`` for captions and titles.
        row: Table row index for line items and accessorials, else ``None``.
    """

    text: str
    box: Box
    field: str | None
    row: int | None = None


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """A drawn page.

    Attributes:
        image: The page, RGB.
        words: Every printed word with its box.
    """

    image: Image.Image
    words: tuple[WordBox, ...]


class _Canvas:
    """A page plus the word boxes drawn on it."""

    def __init__(self, font_size: int) -> None:
        self.image = Image.new("RGB", PAGE_SIZE, "white")
        self.draw = ImageDraw.Draw(self.image)
        self.font = ImageFont.load_default(size=font_size)
        self.bold = ImageFont.load_default(size=round(font_size * 1.6))
        self.line_height = round(font_size * 1.45)
        self.words: list[WordBox] = []

    def text(
        self,
        xy: tuple[int, int],
        text: str,
        field: str | None,
        *,
        row: int | None = None,
        big: bool = False,
    ) -> int:
        """Draw text, record a box per word, and return the x where the text ends."""
        font = self.bold if big else self.font
        x, y = xy
        self.draw.text((x, y), text, fill="black", font=font)
        cursor = 0
        for word in text.split(" "):
            if word:
                start = x + round(font.getlength(text[:cursor]))
                left, top, right, bottom = (round(v) for v in font.getbbox(word))
                self.words.append(
                    WordBox(word, (start + left, y + top, start + right, y + bottom), field, row)
                )
            cursor += len(word) + 1
        return x + round(font.getlength(text))

    def rule(self, y: int) -> None:
        """Draw a horizontal rule across the text area."""
        self.draw.line((MARGIN, y, PAGE_SIZE[0] - MARGIN, y), fill="black", width=2)


def _party_block(canvas: _Canvas, printed: PrintedField, x: int, y: int) -> int:
    for line in printed.lines:
        canvas.text((x, y), line, printed.name)
        y += canvas.line_height
    return y


def _captioned(canvas: _Canvas, printed: PrintedField, x: int, y: int, value_x: int) -> None:
    caption = printed.caption or printed.name.replace("_", " ").title()
    canvas.text((x, y), f"{caption}:", None)
    canvas.text((value_x, y), printed.lines[0], printed.name)


def _table(
    canvas: _Canvas, rows: tuple[PrintedLineItem, ...], field: str, y: int, title: str
) -> int:
    width = PAGE_SIZE[0] - 2 * MARGIN
    columns = (
        MARGIN,
        MARGIN + round(width * 0.50),
        MARGIN + round(width * 0.62),
        MARGIN + round(width * 0.80),
    )
    for x, heading in zip(columns, (title, "Qty", "Unit price", "Amount"), strict=True):
        canvas.text((x, y), heading, None)
    y += canvas.line_height
    canvas.rule(y - 6)
    for index, row in enumerate(rows):
        cells = (row.description, row.quantity, row.unit_price, row.amount)
        for x, cell in zip(columns, cells, strict=True):
            if cell is not None:
                canvas.text((x, y), cell, field, row=index)
        y += canvas.line_height
    return y + canvas.line_height


def render(document: SyntheticDocument) -> RenderedPage:
    """Draw a document on an A4 page.

    Args:
        document: A sampled document.

    Returns:
        The page and the box of every printed word.

    Raises:
        ValueError: If the content runs off the page, because a clipped field would make its
            label false.
    """
    tpl = document.template
    canvas = _Canvas(tpl.font_size)
    by_name = {f.name: f for f in document.fields}
    y = MARGIN
    canvas.text((MARGIN, y), _TITLES[tpl.doc_type], None, big=True)
    y += round(canvas.line_height * 2.2)

    primary = next((by_name[n] for n in _PRIMARY_PARTY if n in by_name), None)
    secondary = next(
        (by_name[n] for n in _SECONDARY_PARTY if n in by_name and by_name[n] is not primary), None
    )
    half = PAGE_SIZE[0] // 2
    left_end = _party_block(canvas, primary, MARGIN, y) if primary else y
    if secondary is not None:
        if tpl.two_column_header:
            right_end = _party_block(canvas, secondary, half, y)
            y = max(left_end, right_end)
        else:
            y = _party_block(canvas, secondary, MARGIN, left_end + canvas.line_height // 2)
    else:
        y = left_end
    y += canvas.line_height

    placed = {f.name for f in (primary, secondary) if f is not None}
    others = [f for f in document.fields if f.name not in placed and f.name not in _TOTALS]
    captions = [f"{f.caption or f.name.replace('_', ' ').title()}:" for f in document.fields]
    widest = max((canvas.font.getlength(c) for c in captions), default=0.0)
    value_offset = max(round(tpl.font_size * 9), round(widest) + tpl.font_size)
    singles = [f for f in others if len(f.lines) == 1]
    for i, printed in enumerate(singles):
        column_x = MARGIN if i % 2 == 0 else half
        _captioned(canvas, printed, column_x, y, column_x + value_offset)
        if i % 2 == 1 or i == len(singles) - 1:
            y += canvas.line_height
    # A party outside the header gets a full row: its address is too wide for half the page.
    for printed in (f for f in others if len(f.lines) > 1):
        canvas.text((MARGIN, y), f"{printed.name.replace('_', ' ').title()}:", None)
        y = _party_block(canvas, printed, MARGIN + value_offset, y)
    y += canvas.line_height

    if document.line_items:
        y = _table(canvas, document.line_items, "line_items", y, "Description")
    if document.accessorials:
        y = _table(canvas, document.accessorials, "accessorials", y, "Accessorial")

    totals_x = MARGIN + round((PAGE_SIZE[0] - 2 * MARGIN) * 0.55)
    for name in _TOTALS:
        if name in by_name:
            _captioned(canvas, by_name[name], totals_x, y, totals_x + value_offset)
            y += canvas.line_height
    if y > PAGE_SIZE[1] - MARGIN or any(
        w.box[2] > PAGE_SIZE[0] - MARGIN // 2 for w in canvas.words
    ):
        raise ValueError(f"{document.doc_id} does not fit on one page")
    return RenderedPage(canvas.image, tuple(canvas.words))


@dataclass(frozen=True, slots=True)
class Degradation:
    """Scan damage applied to a clean page.

    Attributes:
        rotate_degrees: Skew, positive is counter-clockwise.
        blur_radius: Gaussian blur radius in pixels.
        noise: Fraction of pixels flipped to black or white, like fax speckle.
        jpeg_quality: JPEG quality for a lossy round trip, or ``None`` to skip it.
    """

    rotate_degrees: float = 0.0
    blur_radius: float = 0.0
    noise: float = 0.0
    jpeg_quality: int | None = None


LEVELS: tuple[Degradation, ...] = (
    Degradation(),
    Degradation(0.8, 0.6, 0.002, 70),
    Degradation(-1.8, 1.0, 0.008, 45),
    Degradation(3.2, 1.4, 0.02, 25),
)
"""Degradation levels 0 (clean) to 3 (bad fax), used to report accuracy by level."""


def _rotate_box(box: Box, degrees: float) -> Box:
    """Bounding box of a box rotated about the page centre, as Pillow rotates the image."""
    cx, cy = PAGE_SIZE[0] / 2, PAGE_SIZE[1] / 2
    theta = math.radians(degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    left, top, right, bottom = box
    xs: list[float] = []
    ys: list[float] = []
    for x, y in ((left, top), (right, top), (right, bottom), (left, bottom)):
        dx, dy = x - cx, y - cy
        xs.append(cx + dx * cos + dy * sin)
        ys.append(cy - dx * sin + dy * cos)
    return (math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys)))


def degrade(page: RenderedPage, damage: Degradation, seed: int) -> RenderedPage:
    """Apply scan damage and move the word boxes with the skew.

    Args:
        page: A clean rendered page.
        damage: What to apply.
        seed: Seed for the noise.

    Returns:
        The damaged page with boxes that still enclose each word.
    """
    image = page.image
    if damage.rotate_degrees:
        image = image.rotate(
            damage.rotate_degrees, resample=Image.Resampling.BICUBIC, fillcolor="white"
        )
    if damage.blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(damage.blur_radius))
    if damage.noise:
        image = image.copy()
        rng = random.Random(seed)  # reproducible speckle, not a secret
        width, height = image.size
        pixels = image.load()
        if pixels is None:
            raise RuntimeError("could not access page pixels")
        for _ in range(round(width * height * damage.noise)):
            pixels[rng.randrange(width), rng.randrange(height)] = rng.choice(
                ((0, 0, 0), (255, 255, 255))
            )
    if damage.jpeg_quality is not None:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=damage.jpeg_quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    words = page.words
    if damage.rotate_degrees:
        words = tuple(
            WordBox(w.text, _rotate_box(w.box, damage.rotate_degrees), w.field, w.row)
            for w in words
        )
    return RenderedPage(image, words)
