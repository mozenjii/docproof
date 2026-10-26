"""Sample one synthetic document: labels for every schema field and the text each is printed as.

Values are drawn so that the document is internally consistent: line amounts are exactly quantity
times unit price, they sum to the subtotal, and subtotal plus tax is the total. A validator that
fails on a clean synthetic document is therefore a validator bug, which is what makes this set
useful for measuring validator false-positive rates.

Deliberate boundary: everything is invented from small word lists and a seeded generator; no
real company, person or document is used. Fields whose product rules are still open (bank
account, tax ids, equipment type) are left unlabelled rather than guessed.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from docproof.normalize import minor_units
from docproof.schema import (
    NMFC_CLASSES,
    SCHEMAS,
    DocumentType,
    FieldValue,
    LineItem,
    Money,
    Party,
    Quantity,
    ValueType,
)
from docproof.synth.formatting import format_date, format_money, format_quantity
from docproof.synth.templates import Split, Template

_COMPANY_FIRST = (
    "Northwind",
    "Bluegate",
    "Harbor Pine",
    "Kestrel",
    "Ironleaf",
    "Silver Fern",
    "Copperline",
    "Tallgrass",
    "Redwater",
    "Oakmont",
    "Brightfield",
    "Larkspur",
)
_COMPANY_LAST = (
    "Freight Ltd",
    "Logistics",
    "Supply Co",
    "Office Goods",
    "Trading",
    "Carriers",
    "Distribution",
    "Packaging",
    "Imports",
    "Hauling Inc",
)
_STREETS = (
    "Quarry Lane",
    "Dock Road",
    "Mill Street",
    "Station Way",
    "Ridge Avenue",
    "Canal Street",
    "Harbour View",
    "Orchard Close",
    "Foundry Row",
)
_CITIES = (
    "Dunmore, OH 44012",
    "Port Ellis, WA 98221",
    "Greywick, TX 75431",
    "Ashford, GA 30144",
    "Millbrook, IL 60511",
    "Cold Harbor, ME 04011",
    "Fenwick, KS 67401",
    "Riverton, NJ 07601",
)
_ITEMS = (
    "Corrugated boxes",
    "Stretch wrap",
    "Pallet rental",
    "Printer toner",
    "Copy paper",
    "Labour hours",
    "Forklift hire",
    "Storage per pallet",
    "Shrink film",
    "Packing tape",
    "Desk chairs",
    "Cleaning service",
)
_ACCESSORIALS = (
    "Fuel surcharge",
    "Liftgate",
    "Detention",
    "Residential delivery",
    "Inside delivery",
)
_INSTRUCTIONS = ("Call before delivery", "Do not stack", "Keep dry", "Deliver to dock 4")
_TERMS = (0, 15, 30, 45, 60)


@dataclass(frozen=True, slots=True)
class PrintedField:
    """A field as it appears on the page.

    Attributes:
        name: Schema field name.
        caption: Printed caption, or ``None`` for an uncaptioned block such as an address.
        lines: The printed text, one entry per printed line.
    """

    name: str
    caption: str | None
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrintedLineItem:
    """One table row as printed; ``None`` cells are left blank.

    Attributes:
        description: Description cell.
        quantity: Quantity cell.
        unit_price: Unit price cell.
        amount: Amount cell.
    """

    description: str
    quantity: str | None
    unit_price: str | None
    amount: str


@dataclass(frozen=True, slots=True)
class SyntheticDocument:
    """A sampled document: its truth and how it is printed.

    Attributes:
        doc_id: ``<template_id>-<index>``, opaque to everything but the generator.
        template: The template it was drawn from.
        labels: The true value of every schema field; ``None`` when not printed.
        fields: Printed fields in drawing order.
        line_items: Printed rows of the main table.
        accessorials: Printed rows of the accessorial table, for freight documents.
    """

    doc_id: str
    template: Template
    labels: dict[str, FieldValue | None]
    fields: tuple[PrintedField, ...]
    line_items: tuple[PrintedLineItem, ...]
    accessorials: tuple[PrintedLineItem, ...]

    @property
    def split(self) -> Split:
        """The experiment split, inherited from the template."""
        return self.template.split


class _Sampler:
    """Draws values for one document from a seeded generator."""

    def __init__(self, tpl: Template, rng: random.Random) -> None:
        self.tpl = tpl
        self.rng = rng
        self.places = minor_units(tpl.currency)

    def money(self, low: int, high: int) -> Money:
        """A whole-minor-unit amount between ``low`` and ``high`` major units."""
        minor = self.rng.randrange(low * 10**self.places, high * 10**self.places)
        return Money(Decimal(minor).scaleb(-self.places), self.tpl.currency)

    def round(self, amount: Decimal) -> Money:
        """Round an exact amount to the currency's minor unit, half-even, as invoices do."""
        quantum = Decimal(1).scaleb(-self.places)
        return Money(amount.quantize(quantum, rounding=ROUND_HALF_EVEN), self.tpl.currency)

    def company(self) -> Party:
        """An invented company with a one-line street and a city line."""
        name = f"{self.rng.choice(_COMPANY_FIRST)} {self.rng.choice(_COMPANY_LAST)}"
        street = f"{self.rng.randrange(1, 999)} {self.rng.choice(_STREETS)}"
        return Party(name, f"{street}, {self.rng.choice(_CITIES)}")

    def number(self, digits: int) -> str:
        """A document number with the template's prefix."""
        return f"{self.tpl.number_prefix}{self.rng.randrange(10 ** (digits - 1), 10**digits)}"

    def date(self) -> dt.date:
        """A date in 2025 or 2026."""
        return dt.date(2025, 1, 1) + dt.timedelta(days=self.rng.randrange(700))

    def line_items(self, names: tuple[str, ...], low: int, high: int) -> tuple[LineItem, ...]:
        """Rows whose amount is exactly quantity times unit price."""
        rows: list[LineItem] = []
        for description in self.rng.sample(names, self.rng.randint(low, high)):
            quantity = Quantity(
                Decimal(self.rng.randint(1, 24)), self.rng.choice(("EA", "PCS", "HR"))
            )
            price = self.money(1, 400)
            rows.append(
                LineItem(
                    description,
                    Money(quantity.value * price.amount, price.currency),
                    quantity,
                    price,
                )
            )
        return tuple(rows)


def _printed_money(tpl: Template, value: Money) -> str:
    return format_money(value, tpl.locale, with_code=tpl.amounts_with_code)


def _printed(tpl: Template, value_type: ValueType, value: FieldValue) -> tuple[str, ...]:
    """The printed lines of one scalar field."""
    match value:
        case Money():
            return (_printed_money(tpl, value),)
        case dt.date():
            return (format_date(value, tpl.locale),)
        case Quantity():
            return (f"{format_quantity(value, tpl.locale)} {value.unit}",)
        case Party():
            return (value.name,) if value.address is None else (value.name, value.address)
        case str():
            return (value,)
        case _:
            raise TypeError(f"{value_type.value} is printed as a table, not a field")


def _printed_rows(tpl: Template, items: tuple[LineItem, ...]) -> tuple[PrintedLineItem, ...]:
    return tuple(
        PrintedLineItem(
            description=item.description,
            quantity=None if item.quantity is None else format_quantity(item.quantity, tpl.locale),
            unit_price=None if item.unit_price is None else _printed_money(tpl, item.unit_price),
            amount=_printed_money(tpl, item.amount),
        )
        for item in items
    )


def _invoice_values(s: _Sampler, freight: bool) -> dict[str, FieldValue | None]:
    tpl, rng = s.tpl, s.rng
    issued = s.date()
    terms = rng.choice(_TERMS)
    items = s.line_items(_ITEMS, 1, 6) if not freight else ()
    if freight:
        base = s.money(300, 4000)
        items = (LineItem("Linehaul", base),)
    accessorials: tuple[LineItem, ...] = ()
    if freight and rng.random() < 0.7:
        accessorials = tuple(
            LineItem(d, s.money(15, 250)) for d in rng.sample(_ACCESSORIALS, rng.randint(1, 3))
        )
    subtotal = s.round(sum((i.amount.amount for i in items + accessorials), Decimal(0)))
    tax = s.round(subtotal.amount * tpl.tax_rate) if tpl.tax_rate else None
    total = Money(subtotal.amount + (tax.amount if tax else Decimal(0)), tpl.currency)
    values: dict[str, FieldValue | None] = {
        "invoice_number": s.number(6),
        "invoice_date": issued,
        "due_date": issued + dt.timedelta(days=terms) if terms else None,
        "vendor": s.company(),
        "buyer": s.company() if rng.random() < 0.8 else None,
        "purchase_order_number": f"PO{rng.randrange(10000, 99999)}" if rng.random() < 0.5 else None,
        "currency": tpl.currency,
        "line_items": items,
        "subtotal": subtotal if tax is not None or accessorials else None,
        "tax_total": tax,
        "total": total,
        "payment_terms": f"Net {terms}" if terms else "Due on receipt",
    }
    if freight:
        weight_unit = rng.choice(("LB", "KG"))
        values |= {
            "pro_number": str(rng.randrange(10**8, 10**9)),
            "bol_number": f"BOL{rng.randrange(10**5, 10**6)}" if rng.random() < 0.8 else None,
            "shipper": s.company(),
            "consignee": s.company(),
            "origin": rng.choice(_CITIES),
            "destination": rng.choice(_CITIES),
            "weight": Quantity(Decimal(rng.randrange(150, 20000)), weight_unit),
            "accessorials": accessorials,
        }
    return values


def _bol_values(s: _Sampler) -> dict[str, FieldValue | None]:
    rng = s.rng
    return {
        "bol_number": s.number(7),
        "ship_date": s.date(),
        "shipper": s.company(),
        "consignee": s.company(),
        "carrier": s.company(),
        "carrier_scac": "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(4)),
        "pro_number": str(rng.randrange(10**8, 10**9)) if rng.random() < 0.7 else None,
        "pieces": Quantity(Decimal(rng.randint(1, 26)), rng.choice(("PLT", "CTN", "PCS"))),
        "weight": Quantity(Decimal(rng.randrange(100, 30000)), rng.choice(("LB", "KG"))),
        "freight_class": rng.choice(sorted(NMFC_CLASSES, key=Decimal)),
        "hazmat": rng.choice(("Y", "N", "N", "N")),
        "special_instructions": rng.choice(_INSTRUCTIONS) if rng.random() < 0.5 else None,
    }


def _rate_confirmation_values(s: _Sampler) -> dict[str, FieldValue | None]:
    rng = s.rng
    pickup = s.date()
    accessorials = (
        tuple(LineItem(d, s.money(25, 300)) for d in rng.sample(_ACCESSORIALS, rng.randint(1, 2)))
        if rng.random() < 0.4
        else ()
    )
    return {
        "load_number": s.number(7),
        "broker": s.company(),
        "carrier": s.company(),
        "carrier_mc_number": f"MC-{rng.randrange(100000, 999999)}",
        "pickup_date": pickup,
        "pickup_location": rng.choice(_CITIES),
        "delivery_date": pickup + dt.timedelta(days=rng.randint(0, 6)),
        "delivery_location": rng.choice(_CITIES),
        "rate": s.money(600, 9000),
        "accessorials": accessorials,
    }


def sample_document(tpl: Template, index: int, seed: int) -> SyntheticDocument:
    """Sample one document from a template.

    The same ``(template, index, seed)`` always gives the same document.

    Args:
        tpl: The template.
        index: The document's index within its template.
        seed: Dataset seed.

    Returns:
        The document with labels for every schema field and its printed text.
    """
    rng = random.Random(f"{seed}:{tpl.template_id}:{index}")  # reproducible data, not a secret
    sampler = _Sampler(tpl, rng)
    # No default branch: strict pyright rejects a non-exhaustive match, so a new document type
    # must get its own sampler instead of falling into another one's.
    match tpl.doc_type:
        case DocumentType.INVOICE:
            values = _invoice_values(sampler, freight=False)
        case DocumentType.FREIGHT_INVOICE:
            values = _invoice_values(sampler, freight=True)
        case DocumentType.BILL_OF_LADING:
            values = _bol_values(sampler)
        case DocumentType.RATE_CONFIRMATION:
            values = _rate_confirmation_values(sampler)
    labels = {spec.name: values.get(spec.name) for spec in SCHEMAS[tpl.doc_type]}
    for name in ("line_items", "accessorials"):
        if name in labels and labels[name] == ():
            labels[name] = None
    fields: list[PrintedField] = []
    for spec in SCHEMAS[tpl.doc_type]:
        value = labels[spec.name]
        if value is None or spec.value_type is ValueType.LINE_ITEMS:
            continue
        fields.append(
            PrintedField(
                spec.name, tpl.captions.get(spec.name), _printed(tpl, spec.value_type, value)
            )
        )
    main_items = labels.get("line_items")
    extra_items = labels.get("accessorials")
    return SyntheticDocument(
        doc_id=f"{tpl.template_id}-{index:04d}",
        template=tpl,
        labels=labels,
        fields=tuple(fields),
        line_items=_printed_rows(tpl, main_items) if isinstance(main_items, tuple) else (),
        accessorials=_printed_rows(tpl, extra_items) if isinstance(extra_items, tuple) else (),
    )
