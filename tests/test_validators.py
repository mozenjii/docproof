"""Tests for the rule checks, with the boundaries and the fail-closed paths spelled out."""

import datetime as dt
from decimal import Decimal

import pytest

from docproof.errors import ContractError
from docproof.schema import Check, Money
from docproof.validators import (
    parse_quantity,
    v1_line_sum,
    v2_line_arith,
    v3_total_arith,
    v4_date_order,
    v5_currency,
    v7_format,
    v8_positive,
)

PASS, FAIL, UNKNOWN = Check.PASSED, Check.FAILED, Check.UNKNOWN


def usd(amount: str) -> Money:
    return Money(Decimal(amount), "USD")


def jpy(amount: str) -> Money:
    return Money(Decimal(amount), "JPY")


def test_line_sum_exact_sum_passes() -> None:
    assert v1_line_sum([usd("10.00"), usd("2.50")], usd("12.50"), 0) is PASS


def test_line_sum_tolerance_is_inclusive() -> None:
    assert v1_line_sum([usd("10.00"), usd("2.51")], usd("12.50"), 1) is PASS
    assert v1_line_sum([usd("10.00"), usd("2.52")], usd("12.50"), 1) is FAIL


def test_line_sum_zero_decimal_currency_tolerance_is_whole_units() -> None:
    assert v1_line_sum([jpy("1000"), jpy("201")], jpy("1200"), 1) is PASS
    assert v1_line_sum([jpy("1000"), jpy("202")], jpy("1200"), 1) is FAIL


def test_line_sum_missing_inputs_are_unknown_never_pass() -> None:
    assert v1_line_sum([], usd("0.00"), 0) is UNKNOWN
    assert v1_line_sum([usd("1.00"), None], usd("1.00"), 0) is UNKNOWN
    assert v1_line_sum([usd("1.00")], None, 0) is UNKNOWN


def test_line_sum_mixed_currency_fails_even_with_a_missing_line() -> None:
    assert v1_line_sum([usd("1.00"), jpy("1"), None], usd("2.00"), 0) is FAIL


def test_line_sum_negative_tolerance_is_an_error() -> None:
    with pytest.raises(ContractError) as info:
        v1_line_sum([usd("1.00")], usd("1.00"), -1)
    assert info.value.code == "invalid_tolerance"


def test_line_sum_credit_lines_net_off() -> None:
    assert v1_line_sum([usd("10.00"), usd("-2.00")], usd("8.00"), 0) is PASS


def test_line_arithmetic() -> None:
    assert v2_line_arith(Decimal(3), usd("2.50"), usd("7.50"), 0) is PASS
    assert v2_line_arith(Decimal("0.333"), usd("3.00"), usd("1.00"), 1) is PASS  # 0.999 vs 1.00
    assert v2_line_arith(Decimal("0.333"), usd("3.00"), usd("1.00"), 0) is FAIL
    assert v2_line_arith(None, usd("2.50"), usd("7.50"), 0) is UNKNOWN
    assert v2_line_arith(Decimal(1), usd("1.00"), jpy("1"), 0) is FAIL


def test_total_arithmetic() -> None:
    assert v3_total_arith(usd("100.00"), usd("8.25"), usd("108.25"), 0) is PASS
    assert v3_total_arith(usd("100.00"), usd("8.25"), usd("108.26"), 0) is FAIL
    assert v3_total_arith(usd("100.00"), None, usd("100.00"), 0) is UNKNOWN
    assert v3_total_arith(usd("100.00"), jpy("8"), None, 0) is FAIL


def test_date_order() -> None:
    day = dt.date(2026, 10, 1)
    assert v4_date_order(day, day) is PASS
    assert v4_date_order(day, dt.date(2026, 10, 31)) is PASS
    assert v4_date_order(day, dt.date(2026, 9, 30)) is FAIL
    assert v4_date_order(None, day) is UNKNOWN
    assert v4_date_order(day, None) is UNKNOWN


def test_currency_consistency() -> None:
    assert v5_currency("USD", [usd("1.00"), usd("2.00")]) is PASS
    assert v5_currency("USD", [usd("1.00"), jpy("2")]) is FAIL
    assert v5_currency("USD", [jpy("2"), None]) is FAIL
    assert v5_currency("USD", [usd("1.00"), None]) is UNKNOWN
    assert v5_currency(None, [usd("1.00")]) is UNKNOWN
    assert v5_currency("USD", []) is UNKNOWN


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("iban", "GB82 WEST 1234 5698 7654 32", PASS),
        ("iban", "DE89370400440532013000", PASS),
        ("iban", "GB82WEST12345698765433", FAIL),  # one digit changed
        ("iban", "GB82WEST123", FAIL),  # too short
        ("iban", "1282WEST12345698765432", FAIL),  # country must be letters
        ("scac", "abcd", PASS),
        ("scac", "AB", PASS),
        ("scac", "ABCDE", FAIL),
        ("scac", "AB1", FAIL),
        ("mc", "MC-123456", PASS),
        ("mc", "123456", PASS),
        ("mc", "123456789", FAIL),
        ("mc", "DOT123", FAIL),
        ("nmfc", "77.5", PASS),
        ("nmfc", " 500 ", PASS),
        ("nmfc", "75", FAIL),
        ("nmfc", "775", FAIL),  # the dot matters for a class
        ("scac", None, UNKNOWN),
    ],
)
def test_formats(kind: str, value: str | None, expected: Check) -> None:
    assert v7_format(kind, value) is expected


def test_unknown_format_kind_is_an_error_not_a_pass() -> None:
    with pytest.raises(ContractError) as info:
        v7_format("equipment", "VAN")
    assert info.value.code == "unknown_format"


def test_positive_quantity() -> None:
    assert v8_positive(Decimal(12), "PLT") is PASS
    assert v8_positive(Decimal(0), "KG") is FAIL
    assert v8_positive(Decimal(-1), "KG") is FAIL
    assert v8_positive(Decimal(5), "TONNE") is FAIL
    assert v8_positive(None, "KG") is UNKNOWN
    assert v8_positive(Decimal(5), None) is UNKNOWN


def test_parse_quantity() -> None:
    assert parse_quantity(" 2.5 ") == Decimal("2.5")
    for bad in ("two", "NaN", "Infinity", ""):
        with pytest.raises(ContractError) as info:
            parse_quantity(bad)
        assert info.value.code == "invalid_quantity"
