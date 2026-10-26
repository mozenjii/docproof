"""The synthetic document templates: locale, currency, captions, layout knobs and split.

Splits are by template, not by document. Models are trained on templates A-F, tuned on G and
tested on H and I, so a test score measures layouts the model has never seen. Splitting documents
of one template across train and test would measure memorised layout instead.

Deliberate boundary: templates are data. Values are sampled in :mod:`docproof.synth.sample` and
drawn in :mod:`docproof.synth.render`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from docproof.schema import DocumentType
from docproof.synth.formatting import DateStyle, Locale


class Split(StrEnum):
    """Which experiment split a template's documents belong to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class Template:
    """One document layout family.

    Attributes:
        template_id: Stable short id, for example ``"inv-a"``.
        doc_type: Document family.
        split: Experiment split.
        locale: Printing conventions.
        currency: ISO 4217 code for every amount.
        tax_rate: Tax as a fraction of the subtotal; zero means no tax line is printed.
        number_prefix: Prefix of the document's own number.
        amounts_with_code: Print ``1,234.50 USD`` instead of ``1,234.50``.
        two_column_header: Put the counterparty block beside the sender rather than below it.
        font_size: Body font size in pixels at 200 DPI.
        captions: Printed caption for each field shown with a caption.
    """

    template_id: str
    doc_type: DocumentType
    split: Split
    locale: Locale
    currency: str
    tax_rate: Decimal = Decimal(0)
    number_prefix: str = ""
    amounts_with_code: bool = False
    two_column_header: bool = False
    font_size: int = 30
    captions: dict[str, str] = field(default_factory=dict[str, str])


_US = Locale(day_first=False, decimal_comma=False, date_style=DateStyle.NUMERIC)
_UK = Locale(day_first=True, decimal_comma=False, date_style=DateStyle.NUMERIC)
_EU = Locale(day_first=True, decimal_comma=True, date_style=DateStyle.NUMERIC, separator=".")

_INVOICE_CAPTIONS = {
    "invoice_number": "Invoice No.",
    "invoice_date": "Invoice Date",
    "due_date": "Due Date",
    "purchase_order_number": "PO Number",
    "payment_terms": "Terms",
    "subtotal": "Subtotal",
    "tax_total": "Tax",
    "total": "Total",
}

_INV, _FRT = DocumentType.INVOICE, DocumentType.FREIGHT_INVOICE
_BOL, _RC = DocumentType.BILL_OF_LADING, DocumentType.RATE_CONFIRMATION

TEMPLATES: tuple[Template, ...] = (
    Template(
        "inv-a",
        _INV,
        Split.TRAIN,
        _US,
        "USD",
        Decimal("0.0825"),
        "INV-",
        captions=_INVOICE_CAPTIONS,
    ),
    Template(
        "inv-b",
        _INV,
        Split.TRAIN,
        _UK,
        "GBP",
        Decimal("0.20"),
        "",
        two_column_header=True,
        captions=_INVOICE_CAPTIONS | {"invoice_number": "Invoice #", "tax_total": "VAT 20%"},
    ),
    Template(
        "inv-c",
        _INV,
        Split.TRAIN,
        _EU,
        "EUR",
        Decimal("0.19"),
        "RE-",
        amounts_with_code=True,
        captions=_INVOICE_CAPTIONS | {"invoice_number": "Rechnung Nr.", "tax_total": "MwSt 19%"},
    ),
    Template(
        "inv-d",
        _INV,
        Split.TRAIN,
        Locale(False, False, DateStyle.MONTH_NAME_DAY),
        "USD",
        Decimal(0),
        "",
        font_size=26,
        captions=_INVOICE_CAPTIONS | {"invoice_number": "Invoice Number"},
    ),
    Template(
        "inv-e",
        _INV,
        Split.TRAIN,
        Locale(True, False, DateStyle.DAY_MONTH_NAME),
        "AUD",
        Decimal("0.10"),
        "TI-",
        two_column_header=True,
        captions=_INVOICE_CAPTIONS | {"tax_total": "GST"},
    ),
    Template(
        "inv-f",
        _INV,
        Split.TRAIN,
        Locale(True, False, DateStyle.ISO),
        "CAD",
        Decimal("0.05"),
        "",
        amounts_with_code=True,
        captions=_INVOICE_CAPTIONS | {"tax_total": "GST 5%"},
    ),
    Template(
        "inv-g",
        _INV,
        Split.VALIDATION,
        Locale(True, False, DateStyle.DAY_MON_DASH),
        "INR",
        Decimal("0.18"),
        "GST/",
        captions=_INVOICE_CAPTIONS | {"tax_total": "GST 18%", "invoice_number": "Bill No."},
    ),
    Template(
        "inv-h",
        _INV,
        Split.TEST,
        Locale(True, True, DateStyle.NUMERIC, "-"),
        "EUR",
        Decimal("0.21"),
        "F",
        two_column_header=True,
        font_size=28,
        captions=_INVOICE_CAPTIONS | {"invoice_number": "Factuur", "tax_total": "BTW 21%"},
    ),
    Template(
        "inv-i",
        _INV,
        Split.TEST,
        _US,
        "USD",
        Decimal("0.06"),
        "",
        amounts_with_code=True,
        font_size=32,
        captions=_INVOICE_CAPTIONS | {"invoice_number": "Ref", "total": "Amount Due"},
    ),
    Template(
        "frt-a",
        _FRT,
        Split.TRAIN,
        _US,
        "USD",
        Decimal(0),
        "FI-",
        captions=_INVOICE_CAPTIONS
        | {"pro_number": "PRO #", "bol_number": "BOL #", "weight": "Weight"},
    ),
    Template(
        "frt-b",
        _FRT,
        Split.TEST,
        _US,
        "USD",
        Decimal(0),
        "",
        two_column_header=True,
        captions=_INVOICE_CAPTIONS
        | {
            "invoice_number": "Freight Bill",
            "pro_number": "Pro No.",
            "bol_number": "B/L No.",
            "weight": "Total Weight",
        },
    ),
    Template(
        "bol-a",
        _BOL,
        Split.TRAIN,
        _US,
        "USD",
        number_prefix="BOL",
        captions={
            "bol_number": "Bill of Lading No.",
            "ship_date": "Ship Date",
            "carrier_scac": "SCAC",
            "pro_number": "PRO",
            "pieces": "Pieces",
            "weight": "Weight",
            "freight_class": "Class",
            "hazmat": "Hazmat",
        },
    ),
    Template(
        "bol-b",
        _BOL,
        Split.TEST,
        Locale(False, False, DateStyle.ISO),
        "USD",
        two_column_header=True,
        captions={
            "bol_number": "B/L #",
            "ship_date": "Date",
            "carrier_scac": "Carrier SCAC",
            "pro_number": "Pro Number",
            "pieces": "Handling Units",
            "weight": "Gross Weight",
            "freight_class": "NMFC Class",
            "hazmat": "HM",
        },
    ),
    Template(
        "rc-a",
        _RC,
        Split.TRAIN,
        _US,
        "USD",
        number_prefix="L",
        captions={
            "load_number": "Load #",
            "carrier_mc_number": "Carrier MC",
            "pickup_date": "Pickup",
            "delivery_date": "Delivery",
            "rate": "Total Rate",
        },
    ),
    Template(
        "rc-b",
        _RC,
        Split.TEST,
        Locale(False, False, DateStyle.MONTH_NAME_DAY),
        "USD",
        amounts_with_code=True,
        two_column_header=True,
        captions={
            "load_number": "Order No.",
            "carrier_mc_number": "MC Number",
            "pickup_date": "Ship Date",
            "delivery_date": "Deliver By",
            "rate": "Agreed Rate",
        },
    ),
)
"""Every template. Ids are stable; appending is safe, renaming or removing breaks old datasets."""


def template(template_id: str) -> Template:
    """Look up a template by id.

    Args:
        template_id: The template's id.

    Returns:
        The template.

    Raises:
        KeyError: If no template has that id.
    """
    for t in TEMPLATES:
        if t.template_id == template_id:
            return t
    raise KeyError(f"no template {template_id!r}")
