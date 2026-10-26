"""Turn printed dates and amounts into values, keeping every reading the text allows.

A printed ``03/04/2026`` is the 3rd of April or the 4th of March, and ``1,234`` is one thousand
two hundred and thirty-four or one point two three four. Picking one is invention, so these
parsers return :class:`Ambiguous` with both readings unless the text itself or a document-level
hint settles it. Invalid text returns :class:`Invalid`; nothing here raises on bad input.

Deliberate boundary: the hints (``day_first``, ``decimal_comma``) are evidence gathered elsewhere
from the whole document. These functions only apply them, and the rendering beats the hint: a
day above 12 proves the order whatever the hint says.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from decimal import Decimal

MONTHS: dict[str, int] = {
    name: number
    for number, full in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
    for name in (full, full[:3])
}
"""English month names, full and three-letter, lowercase, to month number."""


@dataclass(frozen=True, slots=True)
class Parsed[T]:
    """The text has exactly one reading.

    Attributes:
        value: That reading.
    """

    value: T


@dataclass(frozen=True, slots=True)
class Ambiguous[T]:
    """The text has more than one valid reading and nothing settles it.

    Attributes:
        readings: Every valid reading, sorted ascending.
    """

    readings: tuple[T, ...]


@dataclass(frozen=True, slots=True)
class Invalid:
    """The text is not a supported rendering.

    Attributes:
        reason: Short explanation for logs and reviewers.
    """

    reason: str


type ParseResult[T] = Parsed[T] | Ambiguous[T] | Invalid
"""Outcome of a parse: one reading, several, or none."""

_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_NUMERIC_DATE = re.compile(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{4})")
_DAY_MONTH_NAME = re.compile(r"(\d{1,2})([ -])([A-Za-z]+)\2(\d{4})")
_MONTH_NAME_DAY = re.compile(r"([A-Za-z]+) (\d{1,2}),? (\d{4})")


def _date(year: int, month: int, day: int) -> dt.date | None:
    """Build a date, or return ``None`` if it does not exist on the calendar."""
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _one(candidate: dt.date | None, text: str) -> ParseResult[dt.date]:
    """Wrap a single candidate date, reporting an impossible one as invalid."""
    if candidate is None:
        return Invalid(f"{text!r} is not a calendar date")
    return Parsed(candidate)


def _numeric_date(
    a: int, b: int, year: int, day_first: bool | None, text: str
) -> ParseResult[dt.date]:
    """Resolve ``A/B/YYYY`` using the rendering first and the hint second."""
    if a > 12 and b > 12:
        return Invalid(f"{text!r} has no month")
    if a > 12:
        return _one(_date(year, b, a), text)
    if b > 12 or a == b:
        return _one(_date(year, a, b), text)
    if day_first is True:
        return _one(_date(year, b, a), text)
    if day_first is False:
        return _one(_date(year, a, b), text)
    # Both numbers are 1-12 here, so both readings are real calendar dates.
    return Ambiguous(tuple(sorted((dt.date(year, b, a), dt.date(year, a, b)))))


def parse_date(text: str, day_first: bool | None) -> ParseResult[dt.date]:
    """Parse a printed date.

    Accepts ISO ``YYYY-MM-DD``; numeric ``A/B/YYYY`` with ``/``, ``-`` or ``.`` used twice; and
    English month names (``3 April 2026``, ``03-Apr-2026``, ``April 3, 2026``, ``Apr 3 2026``).
    Two-digit years are invalid because the century is a guess.

    Args:
        text: The printed date.
        day_first: Document-level evidence of day-month order, or ``None`` when there is none.

    Returns:
        The date, both readings when the order is undecidable, or :class:`Invalid`.
    """
    stripped = text.strip()
    if match := _ISO_DATE.fullmatch(stripped):
        year, month, day = (int(g) for g in match.groups())
        return _one(_date(year, month, day), text)
    if match := _NUMERIC_DATE.fullmatch(stripped):
        a, b, year = int(match[1]), int(match[3]), int(match[4])
        return _numeric_date(a, b, year, day_first, text)
    if match := _DAY_MONTH_NAME.fullmatch(stripped):
        month = MONTHS.get(match[3].lower())
        if month is None:
            return Invalid(f"{match[3]!r} is not a month name")
        return _one(_date(int(match[4]), month, int(match[1])), text)
    if match := _MONTH_NAME_DAY.fullmatch(stripped):
        month = MONTHS.get(match[1].lower())
        if month is None:
            return Invalid(f"{match[1]!r} is not a month name")
        return _one(_date(int(match[3]), month, int(match[2])), text)
    return Invalid(f"{text!r} is not a supported date rendering")


_CURRENCY_CODE_EDGE = re.compile(r"^[A-Z]{3}\s*|\s*[A-Z]{3}$")
_SYMBOLS = "$€£¥"
_DIGITS_AND_SEPARATORS = re.compile(r"\d(?:[\d.,]*\d)?")


def _strip_decoration(text: str) -> tuple[str, bool] | None:
    """Remove whitespace, a currency code, symbols and the sign.

    Returns:
        The bare digits-and-separators core and whether the amount is negative, or ``None`` if
        the decoration is malformed (for example two signs).
    """
    core = _CURRENCY_CODE_EDGE.sub("", text.strip()).strip()
    negative = False
    if core.startswith("(") and core.endswith(")"):
        negative, core = True, core[1:-1]
    core = core.strip().strip(_SYMBOLS).strip()
    if core.startswith("-"):
        if negative:
            return None
        negative, core = True, core[1:]
    core = core.strip().strip(_SYMBOLS).strip()
    if not _DIGITS_AND_SEPARATORS.fullmatch(core):
        return None
    return core, negative


def _grouped(integer: str, separator: str) -> str | None:
    """Return ``integer`` without grouping if it is valid 3-digit grouping, else ``None``."""
    groups = integer.split(separator)
    if not 1 <= len(groups[0]) <= 3 or any(len(g) != 3 for g in groups[1:]):
        return None
    return "".join(groups)


def _signed(digits: str, negative: bool) -> str:
    """Prefix a minus sign when negative."""
    return f"-{digits}" if negative else digits


def _single_separator(core: str, sep: str, decimal_comma: bool | None) -> list[str] | None:
    """Return the readings of a core with exactly one separator, or ``None`` if there are none."""
    integer, fraction = core.split(sep)
    as_decimal = f"{integer}.{fraction}"
    if len(fraction) != 3:
        return [as_decimal]
    as_grouping = _grouped(core, sep)
    if as_grouping is None:
        return [as_decimal]
    if decimal_comma is None:
        return [as_decimal, as_grouping]
    separator_is_decimal = decimal_comma == (sep == ",")
    return [as_decimal if separator_is_decimal else as_grouping]


def _readings(core: str, decimal_comma: bool | None) -> list[str] | None:
    """Return every unsigned reading of a bare amount, or ``None`` if it is invalid."""
    commas, dots = core.count(","), core.count(".")
    if commas and dots:
        decimal_sep = "," if core.rfind(",") > core.rfind(".") else "."
        group_sep = "." if decimal_sep == "," else ","
        if core.count(decimal_sep) != 1:
            return None
        integer, fraction = core.split(decimal_sep)
        plain = _grouped(integer, group_sep)
        return None if plain is None else [f"{plain}.{fraction}"]
    if commas > 1 or dots > 1:
        plain = _grouped(core, "," if commas else ".")
        return None if plain is None else [plain]
    if commas == 1 or dots == 1:
        return _single_separator(core, "," if commas else ".", decimal_comma)
    return [core]


def parse_amount(text: str, decimal_comma: bool | None) -> ParseResult[str]:
    """Parse a printed amount into a plain decimal string.

    The result keeps the printed precision (``12,50`` gives ``"12.50"``) so a later quantisation
    can tell a rounded amount from an exact one.

    Args:
        text: The printed amount, possibly with a currency code, symbol or sign.
        decimal_comma: Document-level evidence that ``,`` is the decimal separator, or ``None``.

    Returns:
        The amount, both readings when ``1,234``-style text is undecidable, or :class:`Invalid`.
    """
    stripped = _strip_decoration(text)
    if stripped is None:
        return Invalid(f"{text!r} is not a supported amount rendering")
    core, negative = stripped
    readings = _readings(core, decimal_comma)
    if readings is None:
        return Invalid(f"{text!r} has malformed digit grouping")
    signed = sorted((_signed(r, negative) for r in readings), key=Decimal)
    if len(signed) == 1:
        return Parsed(signed[0])
    return Ambiguous(tuple(signed))
