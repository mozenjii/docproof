"""Tests for label JSON: lossless round trips and strict rejection of malformed labels."""

import datetime as dt
import json
from decimal import Decimal

import pytest

from docproof.labels import LabelError, decode_labels, decode_value, encode_labels, encode_value
from docproof.schema import (
    SCHEMAS,
    DocumentType,
    FieldValue,
    LineItem,
    Money,
    Party,
    Quantity,
    ValueType,
)

USD = Money(Decimal("12.50"), "USD")
ITEMS = (
    LineItem("Pallet", USD, Quantity(Decimal(2), "PLT"), Money(Decimal("6.25"), "USD")),
    LineItem("Fuel", Money(Decimal("1.00"), "USD")),
)


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (ValueType.TEXT, "Net 30"),
        (ValueType.IDENTIFIER, "INV-1"),
        (ValueType.CODE, "77.5"),
        (ValueType.MONEY, USD),
        (ValueType.MONEY, Money(Decimal("1200"), "JPY")),
        (ValueType.DATE, dt.date(2026, 2, 28)),
        (ValueType.QUANTITY, Quantity(Decimal("1250.5"), "LB")),
        (ValueType.PARTY, Party("Northwind", "1 Dock Rd")),
        (ValueType.PARTY, Party("Northwind", None)),
        (ValueType.LINE_ITEMS, ITEMS),
        (ValueType.LINE_ITEMS, ()),
        (ValueType.MONEY, None),
    ],
)
def test_value_round_trips_through_json_text(
    value_type: ValueType, value: FieldValue | None
) -> None:
    text = json.dumps(encode_value(value_type, value))
    decoded = decode_value(value_type, json.loads(text))
    assert decoded == value
    if isinstance(value, Money):
        assert isinstance(decoded, Money)
        assert str(decoded.amount) == str(value.amount)  # precision survives, not just equality


def test_money_amount_is_a_string_in_json() -> None:
    assert encode_value(ValueType.MONEY, USD) == {"amount": "12.50", "currency": "USD"}


def test_encode_rejects_a_mismatched_type() -> None:
    with pytest.raises(LabelError, match="not a money value"):
        encode_value(ValueType.MONEY, "12.50")


@pytest.mark.parametrize(
    ("value_type", "raw"),
    [
        (ValueType.MONEY, {"amount": 12.5, "currency": "USD"}),  # a float amount
        (ValueType.MONEY, {"amount": "NaN", "currency": "USD"}),
        (ValueType.MONEY, {"amount": "abc", "currency": "USD"}),
        (ValueType.MONEY, {"amount": "1", "currency": "usd"}),
        (ValueType.DATE, "2026-02-30"),
        (ValueType.DATE, 20260228),
        (ValueType.QUANTITY, {"value": "1", "unit": "TONNE"}),
        (ValueType.TEXT, 5),
        (ValueType.PARTY, {"name": None}),
        (ValueType.LINE_ITEMS, {"description": "x"}),
        (ValueType.LINE_ITEMS, [{"description": "x", "amount": None}]),
    ],
)
def test_decode_is_strict(value_type: ValueType, raw: object) -> None:
    with pytest.raises(LabelError):
        decode_value(value_type, raw)


def _labels() -> dict[str, FieldValue | None]:
    labels: dict[str, FieldValue | None] = dict.fromkeys(
        (s.name for s in SCHEMAS[DocumentType.INVOICE]), None
    )
    labels.update(invoice_number="A-1", total=USD, line_items=ITEMS, currency="USD")
    return labels


def test_label_set_round_trip() -> None:
    raw = json.loads(json.dumps(encode_labels(DocumentType.INVOICE, _labels())))
    assert decode_labels(raw) == (DocumentType.INVOICE, _labels())


def test_label_sets_must_cover_exactly_the_schema() -> None:
    partial = _labels()
    del partial["buyer"]
    with pytest.raises(LabelError, match="exactly the schema"):
        encode_labels(DocumentType.INVOICE, partial)
    raw = encode_labels(DocumentType.INVOICE, _labels())
    raw["fields"]["surprise"] = None
    with pytest.raises(LabelError, match="exactly the schema"):
        decode_labels(raw)
    with pytest.raises(LabelError, match="unknown doc_type"):
        decode_labels({"doc_type": "receipt", "fields": {}})
