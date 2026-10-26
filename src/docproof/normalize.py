"""Canonical comparison keys for text, identifiers and money.

Scoring and grounding both compare values through these functions, so "the same value" means one
thing everywhere. They are total over their valid inputs and raise :class:`ContractError` with a
stable code otherwise.

Deliberate boundary: normalisation never guesses. It does not fold ``O`` into ``0``, does not
round an amount that has more decimals than its currency allows, and does not infer a currency.
Turning printed text into values is :mod:`docproof.parsing`.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import ROUND_HALF_EVEN, Decimal

from docproof.errors import ContractError
from docproof.schema import Money

MINOR_UNITS: dict[str, int] = {
    **dict.fromkeys(("USD", "CAD", "AUD", "EUR", "GBP", "CHF", "CNY", "INR", "PKR", "MXN"), 2),
    **dict.fromkeys(("JPY", "KRW"), 0),
    **dict.fromkeys(("BHD", "KWD", "OMR"), 3),
}
"""Decimal places of each supported ISO 4217 currency. Unlisted currencies are rejected."""

_EDGE_PUNCTUATION = ".,;:!?\"'()[]{}-_/"
_WHITESPACE = re.compile(r"\s+")
_IDENTIFIER_SEPARATORS = re.compile(r"[\s\-./]")
_PLAIN_DECIMAL = re.compile(r"-?(\d+(\.\d*)?|\.\d+)")


def normalize_text(text: str) -> str:
    """Return the comparison key for free text.

    NFKC, casefold, collapse whitespace runs to one space, then strip whitespace and edge
    punctuation repeatedly from both ends.

    Args:
        text: Text as printed or extracted.

    Returns:
        The comparison key.
    """
    folded = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold())
    previous = None
    while previous != folded:
        previous = folded
        folded = folded.strip().strip(_EDGE_PUNCTUATION)
    return folded


def normalize_identifier(text: str) -> str:
    """Return the comparison key for an identifier such as an invoice or PRO number.

    NFKC, uppercase, and remove spaces, hyphens, dots and slashes. Look-alike characters are left
    alone on purpose: ``O`` versus ``0`` is a reviewer's call, not a normaliser's.

    Args:
        text: Identifier as printed or extracted.

    Returns:
        The comparison key.
    """
    return _IDENTIFIER_SEPARATORS.sub("", unicodedata.normalize("NFKC", text).upper())


def minor_units(currency: str) -> int:
    """Return how many decimal places a currency uses.

    Args:
        currency: ISO 4217 code, any case.

    Returns:
        Number of decimal places.

    Raises:
        ContractError: ``unknown_currency`` if the code is not in :data:`MINOR_UNITS`.
    """
    code = currency.upper()
    if code not in MINOR_UNITS:
        raise ContractError("unknown_currency", f"{currency!r} is not a supported currency")
    return MINOR_UNITS[code]


def normalize_money(amount: str, currency: str) -> Money:
    """Return the canonical form of an amount: quantised to its currency's minor unit.

    Args:
        amount: Plain decimal string such as ``"12.5"`` (no grouping, ``.`` as separator).
        currency: ISO 4217 code, any case.

    Returns:
        The amount quantised to the currency's decimals, with the code uppercased.

    Raises:
        ContractError: ``unknown_currency``; ``invalid_amount`` for a malformed amount;
            ``excess_precision`` if the amount has more decimals than the currency allows,
            because rounding it would hide an extraction error.
    """
    places = minor_units(currency)
    if not _PLAIN_DECIMAL.fullmatch(amount.strip()):
        raise ContractError("invalid_amount", f"{amount!r} is not a plain decimal amount")
    value = Decimal(amount.strip())  # the regex above admits only strings Decimal accepts
    quantum = Decimal(1).scaleb(-places)
    quantised = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    if quantised != value:
        raise ContractError(
            "excess_precision", f"{amount!r} has more than {places} decimals for {currency}"
        )
    return Money(quantised, currency.upper())
