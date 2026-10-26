"""Tests for comparison keys: text, identifiers and money."""

from decimal import Decimal

import pytest

from docproof.errors import ContractError
from docproof.normalize import minor_units, normalize_identifier, normalize_money, normalize_text
from docproof.schema import Money

# Built from code points so the source stays ASCII and the test input is unambiguous.
FULLWIDTH_ABC_LTD = "".join(map(chr, (0xFF21, 0xFF22, 0xFF23, 0x3000))) + "Ltd"
FULLWIDTH_INV_1 = "".join(map(chr, (0xFF29, 0xFF2E, 0xFF36, 0xFF0D, 0xFF11)))


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("  Net   30 days. ", "net 30 days"),
        ("(Freight charges)", "freight charges"),
        # Full-width ABC and an ideographic space fold to ASCII under NFKC.
        (FULLWIDTH_ABC_LTD, "abc ltd"),
        ("STRASSE", "strasse"),
        ("Straße", "strasse"),  # casefold, not lower: ß compares equal to ss
        ('"-- quoted --"', "quoted"),
        ("", ""),
        ("...", ""),
    ],
)
def test_normalize_text(raw: str, key: str) -> None:
    assert normalize_text(raw) == key


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("inv-2026/0042", "INV20260042"),
        (" pro 1234.567 ", "PRO1234567"),
        (FULLWIDTH_INV_1, "INV1"),
    ],
)
def test_normalize_identifier(raw: str, key: str) -> None:
    assert normalize_identifier(raw) == key


def test_identifier_does_not_fold_lookalikes() -> None:
    assert normalize_identifier("INVO01") != normalize_identifier("INV001")


@pytest.mark.parametrize(
    ("amount", "currency", "expected"),
    [
        ("12.5", "usd", Money(Decimal("12.50"), "USD")),
        ("1200", "JPY", Money(Decimal("1200"), "JPY")),
        ("1.5", "KWD", Money(Decimal("1.500"), "KWD")),
        ("-3", "EUR", Money(Decimal("-3.00"), "EUR")),
        (".5", "GBP", Money(Decimal("0.50"), "GBP")),
    ],
)
def test_normalize_money_quantises_to_minor_unit(
    amount: str, currency: str, expected: Money
) -> None:
    result = normalize_money(amount, currency)
    assert result == expected
    assert str(result.amount) == str(expected.amount)


@pytest.mark.parametrize(
    ("amount", "currency", "code"),
    [
        ("12.505", "USD", "excess_precision"),
        ("1200.5", "JPY", "excess_precision"),
        ("12.50", "XYZ", "unknown_currency"),
        ("1,200.00", "USD", "invalid_amount"),
        ("12.5.0", "USD", "invalid_amount"),
        ("", "USD", "invalid_amount"),
        ("NaN", "USD", "invalid_amount"),
    ],
)
def test_normalize_money_errors(amount: str, currency: str, code: str) -> None:
    with pytest.raises(ContractError) as info:
        normalize_money(amount, currency)
    assert info.value.code == code


def test_minor_units() -> None:
    assert minor_units("jpy") == 0
    assert minor_units("BHD") == 3
