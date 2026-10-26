"""Printing must be the exact inverse of parsing under the same locale."""

import datetime as dt
import itertools
import random
from decimal import Decimal

import pytest

from docproof.normalize import MINOR_UNITS, normalize_money
from docproof.parsing import Parsed, parse_amount, parse_date
from docproof.schema import Money
from docproof.synth.formatting import (
    DateStyle,
    Locale,
    format_date,
    format_money,
    format_number,
)

LOCALES = [
    Locale(day_first, decimal_comma, style, sep)
    for day_first, decimal_comma, style, sep in itertools.product(
        [True, False], [True, False], list(DateStyle), ["/", "-", "."]
    )
]


def _dates(n: int) -> list[dt.date]:
    rng = random.Random(11)
    start = dt.date(2024, 1, 1)
    return [start + dt.timedelta(days=rng.randrange(1200)) for _ in range(n)]


@pytest.mark.parametrize("locale", LOCALES, ids=str)
def test_dates_round_trip(locale: Locale) -> None:
    for day in [*_dates(40), dt.date(2026, 12, 12), dt.date(2026, 1, 31)]:
        assert parse_date(format_date(day, locale), locale.day_first) == Parsed(day)


@pytest.mark.parametrize(
    ("decimal_comma", "with_code"), list(itertools.product([True, False], repeat=2))
)
@pytest.mark.parametrize("currency", sorted(MINOR_UNITS))
def test_money_round_trips(currency: str, decimal_comma: bool, with_code: bool) -> None:
    locale = Locale(True, decimal_comma, DateStyle.ISO)
    rng = random.Random(currency)
    places = MINOR_UNITS[currency]
    samples = [rng.randrange(-(10**9), 10**9) for _ in range(60)] + [0, 1, 999, 1000, 1234, -1234]
    for minor in samples:
        money = Money(Decimal(minor).scaleb(-places), currency)
        printed = format_money(money, locale, with_code=with_code)
        parsed = parse_amount(printed, decimal_comma)
        assert isinstance(parsed, Parsed), printed
        assert normalize_money(parsed.value, currency) == money, printed


def test_format_number_examples() -> None:
    assert format_number(Decimal("1234567.5"), 2, decimal_comma=False) == "1,234,567.50"
    assert format_number(Decimal("1234567.5"), 2, decimal_comma=True) == "1.234.567,50"
    assert format_number(Decimal(-1234), 0, decimal_comma=False) == "-1,234"
