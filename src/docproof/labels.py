"""JSON encoding of field values, for label files and recorded predictions.

Money is ``{"amount": "12.50", "currency": "USD"}`` with the amount as a string, so no JSON reader
ever turns it into a float. Dates are ISO strings. ``null`` means the field is not on the
document.

Deliberate boundary: decoding validates types strictly and raises on anything unexpected. A label
file is ground truth; a silently coerced label would make every score computed from it wrong.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, cast

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

type Json = Any


class LabelError(ValueError):
    """A label file or value does not match the schema."""


def _money_out(m: Money) -> Json:
    return {"amount": str(m.amount), "currency": m.currency}


def _quantity_out(q: Quantity) -> Json:
    return {"value": str(q.value), "unit": q.unit}


def encode_value(value_type: ValueType, value: FieldValue | None) -> Json:
    """Encode one field value as JSON-ready data.

    Args:
        value_type: The field's type.
        value: The value, or ``None`` when the field is not on the document.

    Returns:
        JSON-ready data.

    Raises:
        LabelError: If the value's Python type does not match ``value_type``.
    """
    if value is None:
        return None
    match value_type, value:
        case ValueType.TEXT | ValueType.IDENTIFIER | ValueType.CODE, str():
            return value
        case ValueType.MONEY, Money():
            return _money_out(value)
        case ValueType.DATE, dt.date():
            return value.isoformat()
        case ValueType.QUANTITY, Quantity():
            return _quantity_out(value)
        case ValueType.PARTY, Party():
            return {"name": value.name, "address": value.address}
        case ValueType.LINE_ITEMS, tuple():
            return [_line_item_out(i) for i in value]
        case _:
            raise LabelError(f"{type(value).__name__} is not a {value_type.value} value")


def _line_item_out(item: LineItem) -> Json:
    return {
        "description": item.description,
        "amount": _money_out(item.amount),
        "quantity": None if item.quantity is None else _quantity_out(item.quantity),
        "unit_price": None if item.unit_price is None else _money_out(item.unit_price),
    }


def _require(raw: Json, kind: type, what: str) -> Any:
    if not isinstance(raw, kind):
        raise LabelError(f"{what}: expected {kind.__name__}, got {type(raw).__name__}")
    return raw


def _decimal(raw: Json, what: str) -> Decimal:
    text = _require(raw, str, what)
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise LabelError(f"{what}: {text!r} is not a decimal") from exc
    if not value.is_finite():
        raise LabelError(f"{what}: {text!r} is not finite")
    return value


def _money_in(raw: Json, what: str) -> Money:
    obj = _require(raw, dict, what)
    try:
        return Money(
            _decimal(obj.get("amount"), f"{what}.amount"),
            _require(obj.get("currency"), str, f"{what}.currency"),
        )
    except (TypeError, ValueError) as exc:
        raise LabelError(f"{what}: {exc}") from exc


def _quantity_in(raw: Json, what: str) -> Quantity:
    obj = _require(raw, dict, what)
    try:
        return Quantity(
            _decimal(obj.get("value"), f"{what}.value"),
            _require(obj.get("unit"), str, f"{what}.unit"),
        )
    except (TypeError, ValueError) as exc:
        raise LabelError(f"{what}: {exc}") from exc


def _line_item_in(raw: Json, what: str) -> LineItem:
    obj = _require(raw, dict, what)
    quantity, price = obj.get("quantity"), obj.get("unit_price")
    return LineItem(
        description=_require(obj.get("description"), str, f"{what}.description"),
        amount=_money_in(obj.get("amount"), f"{what}.amount"),
        quantity=None if quantity is None else _quantity_in(quantity, f"{what}.quantity"),
        unit_price=None if price is None else _money_in(price, f"{what}.unit_price"),
    )


def decode_value(value_type: ValueType, raw: Json, what: str = "value") -> FieldValue | None:
    """Decode one field value, strictly.

    Args:
        value_type: The field's type.
        raw: JSON data.
        what: Path used in error messages.

    Returns:
        The value, or ``None`` for JSON ``null``.

    Raises:
        LabelError: On any type or format mismatch.
    """
    if raw is None:
        return None
    match value_type:
        case ValueType.TEXT | ValueType.IDENTIFIER | ValueType.CODE:
            return cast(str, _require(raw, str, what))
        case ValueType.MONEY:
            return _money_in(raw, what)
        case ValueType.DATE:
            try:
                return dt.date.fromisoformat(_require(raw, str, what))
            except ValueError as exc:
                raise LabelError(f"{what}: {exc}") from exc
        case ValueType.QUANTITY:
            return _quantity_in(raw, what)
        case ValueType.PARTY:
            obj = _require(raw, dict, what)
            address = obj.get("address")
            return Party(
                _require(obj.get("name"), str, f"{what}.name"),
                None if address is None else _require(address, str, f"{what}.address"),
            )
        case ValueType.LINE_ITEMS:
            items = _require(raw, list, what)
            return tuple(_line_item_in(item, f"{what}[{i}]") for i, item in enumerate(items))


def encode_labels(doc_type: DocumentType, labels: Mapping[str, FieldValue | None]) -> Json:
    """Encode a full label set in schema order.

    Args:
        doc_type: Document family.
        labels: A value or ``None`` for every schema field.

    Returns:
        ``{"doc_type": ..., "fields": {name: encoded}}``.

    Raises:
        LabelError: If a schema field is missing, an extra field is present, or a value has the
            wrong type.
    """
    specs = SCHEMAS[doc_type]
    names = {s.name for s in specs}
    if set(labels) != names:
        raise LabelError(f"labels must cover exactly the schema fields of {doc_type.value}")
    return {
        "doc_type": doc_type.value,
        "fields": {s.name: encode_value(s.value_type, labels[s.name]) for s in specs},
    }


def decode_labels(raw: Json) -> tuple[DocumentType, dict[str, FieldValue | None]]:
    """Decode a label set produced by :func:`encode_labels`.

    Args:
        raw: JSON data.

    Returns:
        The document type and every field's value.

    Raises:
        LabelError: On an unknown document type, missing or extra fields, or a bad value.
    """
    obj = _require(raw, dict, "labels")
    try:
        doc_type = DocumentType(obj.get("doc_type"))
    except ValueError as exc:
        raise LabelError(f"unknown doc_type {obj.get('doc_type')!r}") from exc
    fields = _require(obj.get("fields"), dict, "labels.fields")
    specs = SCHEMAS[doc_type]
    if set(fields) != {s.name for s in specs}:
        raise LabelError(f"labels must cover exactly the schema fields of {doc_type.value}")
    return doc_type, {s.name: decode_value(s.value_type, fields[s.name], s.name) for s in specs}
