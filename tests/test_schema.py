"""Tests for the field vocabulary: value invariants and the shape of the field tables."""

from decimal import Decimal
from typing import cast

import pytest

from docproof.schema import (
    SCHEMAS,
    Check,
    DocumentType,
    Money,
    Quantity,
    Reason,
    Sensitivity,
    Status,
    field_spec,
)

KNOWN_VALIDATORS = {f"V{i}" for i in range(1, 9)}


@pytest.mark.parametrize("bad", [1.5, 2, "3.00", None])
def test_money_rejects_anything_but_decimal(bad: object) -> None:
    with pytest.raises(TypeError, match="must be Decimal"):
        # The cast lies to the type checker on purpose: this is the untyped-boundary case.
        Money(cast(Decimal, bad), "USD")


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity")])
def test_money_rejects_non_finite_amounts(bad: Decimal) -> None:
    with pytest.raises(ValueError, match="finite"):
        Money(bad, "USD")


@pytest.mark.parametrize("code", ["usd", "US", "USDD", "U$D", ""])
def test_money_rejects_malformed_currency(code: str) -> None:
    with pytest.raises(ValueError, match="ISO 4217"):
        Money(Decimal("1.00"), code)


def test_money_keeps_exact_decimal() -> None:
    assert Money(Decimal("0.10"), "EUR").amount + Decimal("0.20") == Decimal("0.30")


def test_quantity_rejects_unknown_unit_and_float() -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        Quantity(Decimal(1), "TONNE")
    with pytest.raises(TypeError, match="must be Decimal"):
        Quantity(cast(Decimal, 1.0), "KG")


@pytest.mark.parametrize("doc_type", list(DocumentType))
def test_every_document_type_has_unique_fields_and_a_required_one(doc_type: DocumentType) -> None:
    names = [spec.name for spec in SCHEMAS[doc_type]]
    assert len(names) == len(set(names))
    assert any(spec.required for spec in SCHEMAS[doc_type])


@pytest.mark.parametrize("doc_type", list(DocumentType))
def test_fields_reference_only_known_validators(doc_type: DocumentType) -> None:
    for spec in SCHEMAS[doc_type]:
        assert set(spec.validators) <= KNOWN_VALIDATORS, spec.name


def test_financial_fields_are_classed_financial() -> None:
    assert field_spec(DocumentType.INVOICE, "bank_account").sensitivity is Sensitivity.FINANCIAL


def test_freight_invoice_extends_invoice() -> None:
    invoice = {s.name for s in SCHEMAS[DocumentType.INVOICE]}
    freight = {s.name for s in SCHEMAS[DocumentType.FREIGHT_INVOICE]}
    assert invoice < freight


def test_unknown_field_is_an_error_not_a_default() -> None:
    with pytest.raises(KeyError, match="no field"):
        field_spec(DocumentType.BILL_OF_LADING, "total")


def test_outcome_vocabularies_match_the_contract() -> None:
    assert {s.value for s in Status} == {"accepted", "flagged", "abstained"}
    assert {c.value for c in Check} == {"pass", "fail", "unknown"}
    assert len(Reason) == 8
