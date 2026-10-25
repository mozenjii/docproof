"""Synthetic invoice pages with known ground-truth text, plus a fixed degradation.

Purpose: give every OCR engine the same deterministic inputs and the exact list of words that
were drawn, so recall can be measured without hand labelling.

Boundary: this module only renders and degrades images and normalises tokens. It knows nothing
about OCR engines, timing or scoring policy beyond the shared token normalisation. Every name,
address and amount is invented; no real party or document is represented.
"""

from __future__ import annotations

import io
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

DPI = 200
"""Render resolution in dots per inch."""

A4_SIZE_PX = (round(8.27 * DPI), round(11.69 * DPI))
"""A4 page size (width, height) in pixels at :data:`DPI`."""

FONT_CANDIDATES: dict[bool, tuple[str, ...]] = {
    False: ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    True: ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
}
"""TrueType fonts tried in order for regular (False) and bold (True) text."""

VENDORS = (
    "Northwind Freight Ltd",
    "Harbourline Logistics Co",
    "Bramble & Finch Haulage",
    "Kestrel Cargo Services",
    "Tidewater Forwarding plc",
    "Copperfield Supply Partners",
)
CUSTOMERS = (
    "Larkspur Retail Group",
    "Oakhollow Manufacturing",
    "Meridian Garden Supplies",
    "Silverbirch Components Ltd",
    "Ravensworth Foods",
)
STREETS = ("Quayside Road", "Mill Lane", "Station Parade", "Harbour Street", "Foundry Way")
TOWNS = (
    ("Felixstowe", "IP11 4TQ"),
    ("Leeds", "LS9 0PZ"),
    ("Bristol", "BS2 7HN"),
    ("Hull", "HU8 6RW"),
    ("Coventry", "CV3 2LB"),
)
ITEMS = (
    "Linehaul Rotterdam to Leeds",
    "Pallet handling charge",
    "Customs clearance fee",
    "Fuel surcharge",
    "Container demurrage per day",
    "Warehouse storage per pallet",
    "Tail lift delivery",
    "Shrink wrap and labelling",
    "Cargo insurance premium",
    "Port security levy",
    "Out of hours collection",
)

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


@dataclass(frozen=True)
class InvoicePage:
    """A rendered page and the words drawn on it.

    Attributes:
        page_id: Stable identifier such as ``"inv-0"``.
        image: RGB page image.
        words: Every whitespace-separated word drawn on the page, in drawing order, unnormalised.
    """

    page_id: str
    image: Image.Image
    words: tuple[str, ...]


def normalize_tokens(text: str) -> list[str]:
    """Split text into comparison tokens.

    Tokens are whitespace-separated, case-folded and stripped of every character that is not an
    ASCII letter or digit. Tokens that become empty (a lone dash or bar) are dropped. The same
    function is applied to ground truth and OCR output, so both sides are judged alike.

    Args:
        text: Any string, possibly multi-line.

    Returns:
        The normalised, non-empty tokens in their original order.
    """
    tokens = (_NON_ALNUM.sub("", raw.casefold()) for raw in text.split())
    return [token for token in tokens if token]


def load_font(size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font at a pixel size, falling back to Pillow's bundled font.

    Args:
        size_px: Font size in pixels.
        bold: Whether to prefer a bold face.

    Returns:
        The first candidate font that exists on this machine, else Pillow's default font at the
        requested size.
    """
    for candidate in FONT_CANDIDATES[bold]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size_px)
    return ImageFont.load_default(size=size_px)


def _pt(points: float) -> int:
    """Convert a typographic point size to pixels at :data:`DPI`."""
    return round(points / 72 * DPI)


class _Canvas:
    """Draws text onto a page while recording every word for ground truth."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.image = Image.new("RGB", size, "white")
        self._draw = ImageDraw.Draw(self.image)
        self.words: list[str] = []

    def text(
        self,
        xy: tuple[int, int],
        content: str,
        points: float = 10,
        bold: bool = False,
        anchor: str = "la",
    ) -> None:
        """Draw one line of text and record its words."""
        self._draw.text(xy, content, fill="black", font=load_font(_pt(points), bold), anchor=anchor)
        self.words.extend(content.split())

    def rule(self, start: tuple[int, int], end: tuple[int, int], width: int = 2) -> None:
        """Draw a straight rule; rules carry no ground-truth text."""
        self._draw.line([start, end], fill="black", width=width)


def _money(amount_pence: int) -> str:
    """Format an integer number of pence as ``1,234.50``."""
    return f"{amount_pence / 100:,.2f}"


def _draw_line_items(
    canvas: _Canvas, rng: random.Random, table_top: int, margin: int, right: int
) -> None:
    """Draw the line-item table with its header at ``table_top``, then the totals block."""
    columns = (margin, margin + _pt(300), margin + _pt(380), right)
    canvas.rule((margin, table_top - _pt(6)), (right, table_top - _pt(6)))
    canvas.text((columns[0], table_top), "Description", bold=True)
    canvas.text((columns[1], table_top), "Qty", bold=True)
    canvas.text((columns[2], table_top), "Unit Price", bold=True)
    canvas.text((columns[3], table_top), "Amount", bold=True, anchor="ra")
    canvas.rule((margin, table_top + _pt(18)), (right, table_top + _pt(18)))

    subtotal = 0
    row_y = table_top + _pt(26)
    for description in rng.sample(ITEMS, rng.randint(5, 8)):
        quantity = rng.randint(1, 40)
        unit_pence = rng.randint(1250, 95000)
        amount = quantity * unit_pence
        subtotal += amount
        canvas.text((columns[0], row_y), description)
        canvas.text((columns[1], row_y), str(quantity))
        canvas.text((columns[2], row_y), _money(unit_pence))
        canvas.text((columns[3], row_y), _money(amount), anchor="ra")
        row_y += _pt(20)
    canvas.rule((margin, row_y), (right, row_y))

    tax = round(subtotal * 0.20)
    label_x = columns[2]
    totals_y = row_y + _pt(12)
    for label, value, bold in (
        ("Subtotal", subtotal, False),
        ("VAT 20%", tax, False),
        ("Total GBP", subtotal + tax, True),
    ):
        canvas.text((label_x, totals_y), label, bold=bold)
        canvas.text((right, totals_y), _money(value), bold=bold, anchor="ra")
        totals_y += _pt(18)


def make_invoice(index: int, seed: int = 20261025) -> InvoicePage:
    """Render one synthetic freight invoice.

    The same ``(index, seed)`` pair always yields the same image and words.

    Args:
        index: Page number; selects different invented content per page.
        seed: Base random seed shared by the whole page set.

    Returns:
        The rendered page and its ground-truth words.
    """
    rng = random.Random(seed * 1000 + index)
    width, height = A4_SIZE_PX
    canvas = _Canvas(A4_SIZE_PX)
    margin = _pt(54)

    vendor = rng.choice(VENDORS)
    town, postcode = rng.choice(TOWNS)
    canvas.text((margin, margin), vendor, points=20, bold=True)
    canvas.text((margin, margin + _pt(30)), f"{rng.randint(2, 180)} {rng.choice(STREETS)}")
    canvas.text((margin, margin + _pt(44)), f"{town} {postcode}")
    vat_number = f"{rng.randint(100, 999)} {rng.randint(1000, 9999)} {rng.randint(10, 99)}"
    canvas.text((margin, margin + _pt(58)), f"VAT Reg GB {vat_number}")

    right = width - margin
    issued = date(2026, 1, 1) + timedelta(days=rng.randint(0, 280))
    due = issued + timedelta(days=30)
    canvas.text((right, margin), "INVOICE", points=24, bold=True, anchor="ra")
    invoice_number = f"INV-{issued.year}-{rng.randint(1000, 9999)}"
    canvas.text((right, margin + _pt(34)), f"Invoice No: {invoice_number}", anchor="ra")
    canvas.text((right, margin + _pt(48)), f"Invoice Date: {issued:%d %b %Y}", anchor="ra")
    canvas.text((right, margin + _pt(62)), f"Due Date: {due:%d %b %Y}", anchor="ra")
    purchase_order = f"PO-{rng.randint(10000, 99999)}"
    canvas.text((right, margin + _pt(76)), f"PO Reference: {purchase_order}", anchor="ra")

    top = margin + _pt(110)
    customer_town, customer_postcode = rng.choice(TOWNS)
    canvas.text((margin, top), "Bill To", points=11, bold=True)
    canvas.text((margin, top + _pt(16)), rng.choice(CUSTOMERS))
    canvas.text((margin, top + _pt(30)), f"{rng.randint(2, 180)} {rng.choice(STREETS)}")
    canvas.text((margin, top + _pt(44)), f"{customer_town} {customer_postcode}")

    _draw_line_items(canvas, rng, top + _pt(84), margin, right)

    footer_y = height - margin - _pt(40)
    canvas.text((margin, footer_y), "Payment terms: 30 days from invoice date.", points=9)
    canvas.text(
        (margin, footer_y + _pt(14)),
        f"Please quote the invoice number with payment to {vendor}.",
        points=9,
    )
    return InvoicePage(page_id=f"inv-{index}", image=canvas.image, words=tuple(canvas.words))


def degrade(image: Image.Image) -> Image.Image:
    """Apply the fixed scan-like degradation used for the degraded variant.

    Rotates 1.5 degrees (canvas expanded, white fill), applies a Gaussian blur of radius 1 and
    round-trips through JPEG at quality 30. The result is deterministic.

    Args:
        image: Clean RGB page image.

    Returns:
        A new degraded RGB image; the input is not modified.
    """
    rotated = image.rotate(1.5, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    blurred = rotated.filter(ImageFilter.GaussianBlur(radius=1))
    buffer = io.BytesIO()
    blurred.save(buffer, format="JPEG", quality=30)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def make_page_set(count: int = 3, seed: int = 20261025) -> list[InvoicePage]:
    """Render the standard set of clean pages.

    Args:
        count: Number of pages.
        seed: Base random seed.

    Returns:
        ``count`` clean pages with ids ``inv-0`` .. ``inv-{count-1}``.
    """
    return [make_invoice(index, seed) for index in range(count)]
