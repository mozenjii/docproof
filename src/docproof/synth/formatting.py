"""Print field values the way a given template's locale prints them.

These are the inverse of :mod:`docproof.parsing`: for any value and locale, parsing the printed
text with the same locale's hints must give the value back. The generator's tests hold that line.

Deliberate boundary: formatting only. Which style a template uses is decided in
:mod:`docproof.synth.templates`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from docproof.normalize import minor_units
from docproof.schema import Money, Quantity

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


class DateStyle(StrEnum):
    """How a template prints dates."""

    ISO = "iso"  # 2026-04-03
    NUMERIC = "numeric"  # 03/04/2026 or 04/03/2026, by day_first
    DAY_MONTH_NAME = "day_month_name"  # 3 April 2026
    MONTH_NAME_DAY = "month_name_day"  # April 3, 2026
    DAY_MON_DASH = "day_mon_dash"  # 03-Apr-2026


@dataclass(frozen=True, slots=True)
class Locale:
    """A template's printing conventions.

    Attributes:
        day_first: Numeric dates print the day before the month.
        decimal_comma: Amounts use ``,`` for decimals and ``.`` for grouping.
        date_style: How dates are printed.
        separator: Separator for numeric dates.
    """

    day_first: bool
    decimal_comma: bool
    date_style: DateStyle
    separator: str = "/"


def format_date(value: dt.date, locale: Locale) -> str:
    """Print a date in the locale's style.

    Args:
        value: The date.
        locale: Printing conventions.

    Returns:
        The printed date.
    """
    month = _MONTHS[value.month - 1]
    match locale.date_style:
        case DateStyle.ISO:
            return value.isoformat()
        case DateStyle.NUMERIC:
            first, second = (
                (value.day, value.month) if locale.day_first else (value.month, value.day)
            )
            sep = locale.separator
            return f"{first:02d}{sep}{second:02d}{sep}{value.year}"
        case DateStyle.DAY_MONTH_NAME:
            return f"{value.day} {month} {value.year}"
        case DateStyle.MONTH_NAME_DAY:
            return f"{month} {value.day}, {value.year}"
        case DateStyle.DAY_MON_DASH:
            return f"{value.day:02d}-{month[:3]}-{value.year}"


def format_number(value: Decimal, places: int, decimal_comma: bool) -> str:
    """Print a decimal with 3-digit grouping and a fixed number of places.

    Args:
        value: The number.
        places: Decimal places to print.
        decimal_comma: Use ``,`` for decimals and ``.`` for grouping.

    Returns:
        The printed number, with a leading ``-`` when negative.
    """
    text = f"{abs(value):,.{places}f}"
    if decimal_comma:
        text = text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return f"-{text}" if value < 0 else text


def format_money(value: Money, locale: Locale, *, with_code: bool) -> str:
    """Print an amount at its currency's precision.

    Args:
        value: The amount.
        locale: Printing conventions.
        with_code: Append the ISO code (``1,234.50 USD``).

    Returns:
        The printed amount.
    """
    text = format_number(value.amount, minor_units(value.currency), locale.decimal_comma)
    return f"{text} {value.currency}" if with_code else text


def format_quantity(value: Quantity, locale: Locale) -> str:
    """Print a quantity's number without its unit, keeping its own precision.

    Args:
        value: The quantity.
        locale: Printing conventions.

    Returns:
        The printed number.
    """
    exponent = value.value.as_tuple().exponent
    places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return format_number(value.value, places, locale.decimal_comma)
