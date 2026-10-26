"""Tests for date and amount parsing, especially the cases that must not be guessed."""

import datetime as dt

import pytest

from docproof.parsing import Ambiguous, Invalid, Parsed, parse_amount, parse_date

D = dt.date


@pytest.mark.parametrize(
    ("text", "day_first", "expected"),
    [
        ("2026-04-03", None, Parsed(D(2026, 4, 3))),
        (" 2026-04-03 ", False, Parsed(D(2026, 4, 3))),
        ("03/04/2026", True, Parsed(D(2026, 4, 3))),
        ("03/04/2026", False, Parsed(D(2026, 3, 4))),
        ("03/04/2026", None, Ambiguous((D(2026, 3, 4), D(2026, 4, 3)))),
        ("13/04/2026", None, Parsed(D(2026, 4, 13))),
        ("13/04/2026", False, Parsed(D(2026, 4, 13))),  # the rendering beats the hint
        ("04/13/2026", True, Parsed(D(2026, 4, 13))),
        ("12.12.2026", None, Parsed(D(2026, 12, 12))),
        ("5-6-2026", True, Parsed(D(2026, 6, 5))),
        ("3 April 2026", None, Parsed(D(2026, 4, 3))),
        ("03-Apr-2026", None, Parsed(D(2026, 4, 3))),
        ("April 3, 2026", None, Parsed(D(2026, 4, 3))),
        ("apr 3 2026", None, Parsed(D(2026, 4, 3))),
    ],
)
def test_parse_date(text: str, day_first: bool | None, expected: object) -> None:
    assert parse_date(text, day_first) == expected


@pytest.mark.parametrize(
    "text",
    [
        "03/04/26",  # two-digit year: the century is a guess
        "31/02/2026",
        "2026-02-30",
        "13/13/2026",
        "03/04-2026",  # mixed separators
        "3 Sept 2026",
        "3-Apr 2026",
        "Smarch 3, 2026",
        "3 Smarch 2026",
        "",
        "tomorrow",
    ],
)
def test_parse_date_invalid(text: str) -> None:
    assert isinstance(parse_date(text, None), Invalid)


@pytest.mark.parametrize(
    ("text", "decimal_comma", "expected"),
    [
        ("1,234.56", None, Parsed("1234.56")),
        ("1.234,56", None, Parsed("1234.56")),
        ("12,50", None, Parsed("12.50")),
        ("12.5", None, Parsed("12.5")),
        ("1,234,567", None, Parsed("1234567")),
        ("1.234.567", None, Parsed("1234567")),
        ("1,234", None, Ambiguous(("1.234", "1234"))),
        ("1,234", True, Parsed("1.234")),
        ("1,234", False, Parsed("1234")),
        ("1.234", True, Parsed("1234")),
        ("1.234", False, Parsed("1.234")),
        ("12345,678", None, Parsed("12345.678")),  # cannot be grouping, so it is a decimal
        ("1,2345", None, Parsed("1.2345")),
        ("$1,200.00", None, Parsed("1200.00")),
        ("USD 99.95", None, Parsed("99.95")),
        ("99,95 EUR", None, Parsed("99.95")),
        ("€ 5", None, Parsed("5")),
        ("-12.00", None, Parsed("-12.00")),
        ("(12.00)", None, Parsed("-12.00")),
        ("-$12.00", None, Parsed("-12.00")),
        ("-1,234", None, Ambiguous(("-1234", "-1.234"))),
        ("1234", None, Parsed("1234")),
    ],
)
def test_parse_amount(text: str, decimal_comma: bool | None, expected: object) -> None:
    assert parse_amount(text, decimal_comma) == expected


@pytest.mark.parametrize(
    "text",
    [
        "12,34,567",  # Indian lakh grouping is not 3-digit grouping
        "1,23.45",
        "1.234.56,7.8",
        "1,234.567.8",
        "--12",
        "-(12)",
        "(-12)",
        "12a",
        ",50",
        "12.",
        "",
        "USD",
    ],
)
def test_parse_amount_invalid(text: str) -> None:
    assert isinstance(parse_amount(text, None), Invalid)
