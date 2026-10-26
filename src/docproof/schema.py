"""Field vocabulary: value types, document types, field specifications and outcome enums.

Every other module speaks in these types. They are immutable, carry no I/O, and encode the rules
that must hold for any value regardless of where it came from: money is a ``Decimal`` with an
ISO 4217 code, never a float; a field's status and reason codes come from closed sets.

Deliberate boundary: this module says what a field *is*. Parsing raw text into these values
lives in :mod:`docproof.parsing`, comparison keys in :mod:`docproof.normalize`, and the rules
that check values against each other in :mod:`docproof.validators`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class ValueType(StrEnum):
    """The kind of value a field holds; decides how it is parsed, normalised and compared."""

    TEXT = "text"
    IDENTIFIER = "identifier"
    MONEY = "money"
    DATE = "date"
    QUANTITY = "quantity"
    CODE = "code"
    PARTY = "party"
    LINE_ITEMS = "line_items"


class Sensitivity(StrEnum):
    """How much harm a leaked value does; drives logging, masking and retention rules."""

    PUBLIC = "public"
    BUSINESS = "business"
    PERSONAL = "personal"
    FINANCIAL = "financial"


class DocumentType(StrEnum):
    """The document families docproof extracts from."""

    INVOICE = "invoice"
    FREIGHT_INVOICE = "freight_invoice"
    BILL_OF_LADING = "bill_of_lading"
    RATE_CONFIRMATION = "rate_confirmation"


class Status(StrEnum):
    """The one outcome every field leaves the pipeline with. There is no default member."""

    ACCEPTED = "accepted"
    FLAGGED = "flagged"
    ABSTAINED = "abstained"


class Reason(StrEnum):
    """Why a field was flagged. A closed set: a new reason is a schema change, not a string."""

    UNGROUNDED = "ungrounded"
    VALIDATOR_FAILED = "validator_failed"
    VALIDATOR_UNKNOWN = "validator_unknown"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS_RENDERING = "ambiguous_rendering"
    CONFLICTING_CANDIDATES = "conflicting_candidates"
    REQUIRED_NOT_FOUND = "required_not_found"
    OUT_OF_VOCABULARY = "out_of_vocabulary"


class Check(StrEnum):
    """Result of one validator. ``UNKNOWN`` means an input was missing, and is never a pass."""

    PASSED = "pass"
    FAILED = "fail"
    UNKNOWN = "unknown"


def _require_decimal(label: str, value: object) -> None:
    """Raise unless ``value`` is a ``Decimal``.

    The annotation already says ``Decimal``, but values arrive from JSON and model output where
    nothing enforces annotations, and one float turns ``0.1 + 0.2`` into a validator failure.

    Raises:
        TypeError: If ``value`` is any other type, including ``float`` and ``int``.
    """
    if not isinstance(value, Decimal):
        raise TypeError(f"{label} must be Decimal, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Money:
    """An amount in a named currency.

    Attributes:
        amount: Exact decimal amount; floats are rejected so rounding never happens silently.
        currency: ISO 4217 alphabetic code, uppercase.
    """

    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        """Reject floats and malformed currency codes at construction."""
        _require_decimal("Money.amount", self.amount)
        if not self.amount.is_finite():
            raise ValueError(f"Money.amount must be finite, got {self.amount}")
        if len(self.currency) != 3 or not self.currency.isalpha() or not self.currency.isupper():
            raise ValueError(
                f"currency must be a 3-letter uppercase ISO 4217 code: {self.currency!r}"
            )


@dataclass(frozen=True, slots=True)
class Quantity:
    """A measured count or weight.

    Attributes:
        value: Exact decimal value.
        unit: Unit code from :data:`UNITS`.
    """

    value: Decimal
    unit: str

    def __post_init__(self) -> None:
        """Reject floats and units outside the closed set."""
        _require_decimal("Quantity.value", self.value)
        if self.unit not in UNITS:
            raise ValueError(f"unknown unit {self.unit!r}; expected one of {sorted(UNITS)}")


@dataclass(frozen=True, slots=True)
class Party:
    """A business or person named on a document.

    Attributes:
        name: Name as printed.
        address: Address as one block of text, or ``None`` when not printed.
    """

    name: str
    address: str | None = None


@dataclass(frozen=True, slots=True)
class LineItem:
    """One row of an itemised table.

    Attributes:
        description: Description as printed.
        amount: Line total.
        quantity: Quantity, when printed.
        unit_price: Price per unit, when printed.
    """

    description: str
    amount: Money
    quantity: Quantity | None = None
    unit_price: Money | None = None


type FieldValue = str | Money | Quantity | Party | dt.date | tuple[LineItem, ...]
"""Any value a field can hold. Text, identifiers and codes are ``str``."""


UNITS: frozenset[str] = frozenset({"EA", "PCS", "PLT", "CTN", "LB", "KG", "HR", "MI", "KM"})
"""Closed set of quantity units: counts, weights, hours and distances."""

NMFC_CLASSES: frozenset[str] = frozenset(
    {
        "50",
        "55",
        "60",
        "65",
        "70",
        "77.5",
        "85",
        "92.5",
        "100",
        "110",
        "125",
        "150",
        "175",
        "200",
        "250",
        "300",
        "400",
        "500",
    }
)
"""The eighteen NMFC freight classes."""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """The contract for one field of one document type.

    Attributes:
        name: Field name, unique within its document type.
        value_type: Kind of value.
        required: Whether the document is incomplete without it.
        sensitivity: Harm class of the value.
        validators: IDs of the validators that read this field.
    """

    name: str
    value_type: ValueType
    required: bool
    sensitivity: Sensitivity
    validators: tuple[str, ...] = ()


def _f(
    name: str,
    value_type: ValueType,
    required: bool,
    sensitivity: Sensitivity,
    *validators: str,
) -> FieldSpec:
    """Build a :class:`FieldSpec` compactly for the tables below."""
    return FieldSpec(name, value_type, required, sensitivity, validators)


_T, _ID, _M, _D, _Q, _C, _P, _L = (
    ValueType.TEXT,
    ValueType.IDENTIFIER,
    ValueType.MONEY,
    ValueType.DATE,
    ValueType.QUANTITY,
    ValueType.CODE,
    ValueType.PARTY,
    ValueType.LINE_ITEMS,
)
_PUB, _BUS, _PER, _FIN = (
    Sensitivity.PUBLIC,
    Sensitivity.BUSINESS,
    Sensitivity.PERSONAL,
    Sensitivity.FINANCIAL,
)

_INVOICE_FIELDS: tuple[FieldSpec, ...] = (
    _f("invoice_number", _ID, True, _BUS),
    _f("invoice_date", _D, True, _BUS, "V4"),
    _f("due_date", _D, False, _BUS, "V4"),
    # A sole trader's vendor address is a home address, so the party is classed personal.
    _f("vendor", _P, True, _PER, "V6"),
    _f("vendor_tax_id", _ID, False, _BUS, "V7"),
    _f("buyer", _P, False, _PER),
    _f("purchase_order_number", _ID, False, _BUS),
    _f("currency", _C, True, _PUB, "V5"),
    _f("line_items", _L, False, _BUS, "V1", "V2"),
    _f("subtotal", _M, False, _BUS, "V1", "V3"),
    _f("tax_total", _M, False, _BUS, "V3"),
    _f("total", _M, True, _BUS, "V3", "V5"),
    _f("payment_terms", _T, False, _BUS),
    _f("bank_account", _ID, False, _FIN, "V7"),
)

_FREIGHT_EXTRA: tuple[FieldSpec, ...] = (
    _f("pro_number", _ID, False, _BUS),
    _f("bol_number", _ID, False, _BUS),
    _f("shipper", _P, False, _PER),
    _f("consignee", _P, False, _PER),
    _f("origin", _T, False, _PER),
    _f("destination", _T, False, _PER),
    _f("weight", _Q, False, _BUS, "V8"),
    _f("accessorials", _L, False, _BUS, "V1"),
)

_BOL_FIELDS: tuple[FieldSpec, ...] = (
    _f("bol_number", _ID, True, _BUS),
    _f("ship_date", _D, True, _BUS),
    _f("shipper", _P, True, _PER),
    _f("consignee", _P, True, _PER),
    _f("carrier", _P, False, _BUS),
    _f("carrier_scac", _ID, False, _BUS, "V7"),
    _f("pro_number", _ID, False, _BUS),
    _f("pieces", _Q, False, _BUS, "V8"),
    _f("weight", _Q, False, _BUS, "V8"),
    _f("freight_class", _C, False, _BUS, "V7"),
    _f("hazmat", _C, False, _BUS),
    _f("special_instructions", _T, False, _BUS),
)

_RATE_CONFIRMATION_FIELDS: tuple[FieldSpec, ...] = (
    _f("load_number", _ID, True, _BUS),
    _f("broker", _P, True, _BUS),
    _f("carrier", _P, True, _BUS),
    _f("carrier_mc_number", _ID, False, _BUS, "V7"),
    _f("pickup_date", _D, True, _BUS, "V4"),
    _f("pickup_location", _T, True, _PER),
    _f("delivery_date", _D, True, _BUS, "V4"),
    _f("delivery_location", _T, True, _PER),
    _f("equipment_type", _C, False, _BUS, "V7"),
    _f("rate", _M, True, _BUS, "V5"),
    _f("accessorials", _L, False, _BUS),
)

SCHEMAS: dict[DocumentType, tuple[FieldSpec, ...]] = {
    DocumentType.INVOICE: _INVOICE_FIELDS,
    DocumentType.FREIGHT_INVOICE: _INVOICE_FIELDS + _FREIGHT_EXTRA,
    DocumentType.BILL_OF_LADING: _BOL_FIELDS,
    DocumentType.RATE_CONFIRMATION: _RATE_CONFIRMATION_FIELDS,
}
"""Field specifications per document type, in display order."""


def field_spec(document_type: DocumentType, name: str) -> FieldSpec:
    """Return the specification of one field.

    Args:
        document_type: The document family.
        name: The field name.

    Returns:
        The field's specification.

    Raises:
        KeyError: If the document type has no such field; an unknown field is a bug, not a gap.
    """
    for spec in SCHEMAS[document_type]:
        if spec.name == name:
            return spec
    raise KeyError(f"{document_type.value} has no field {name!r}")
