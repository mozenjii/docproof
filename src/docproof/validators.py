"""Rule checks that read extracted values against each other: arithmetic, dates, currency, formats.

Each validator is a pure function returning :class:`~docproof.schema.Check`. ``UNKNOWN`` means an
input the rule needs is missing; it is never a pass, and routing treats it as a reason to ask a
human. A definite violation among the inputs that are present outranks a missing one: if two line
amounts are in different currencies the answer is ``FAILED`` even when a third is missing.

Deliberate boundary: tolerances are always passed in, never defaulted, because the acceptable
rounding slack is a product decision that has not been made. Vendor matching (V6) and equipment
vocabulary (part of V7) wait on their own decisions and are not implemented.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from docproof.errors import ContractError
from docproof.normalize import minor_units, normalize_identifier
from docproof.schema import NMFC_CLASSES, UNITS, Check, Money

_IBAN = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}")
_SCAC = re.compile(r"[A-Z]{2,4}")
_MC = re.compile(r"(?:MC)?\d{1,8}")


def _slack(currency: str, tol: int) -> Decimal:
    """Convert a tolerance in minor units into an amount in the currency."""
    return Decimal(tol).scaleb(-minor_units(currency))


def _check_tolerance(tol: int) -> None:
    """Reject a negative tolerance, which would make every comparison fail."""
    if tol < 0:
        raise ContractError("invalid_tolerance", f"tolerance must be >= 0 minor units, got {tol}")


def _within(left: Decimal, right: Decimal, currency: str, tol: int) -> Check:
    """Compare two amounts in one currency within ``tol`` minor units."""
    return Check.PASSED if abs(left - right) <= _slack(currency, tol) else Check.FAILED


def _same_currency(amounts: Sequence[Money | None]) -> bool:
    """Return whether every present amount shares one currency."""
    return len({m.currency for m in amounts if m is not None}) <= 1


def v1_line_sum(line_amounts: Sequence[Money | None], subtotal: Money | None, tol: int) -> Check:
    """V1: the line amounts add up to the subtotal.

    Args:
        line_amounts: Amount of every line, ``None`` where a line's amount was not extracted.
        subtotal: The printed subtotal.
        tol: Allowed difference in minor units.

    Returns:
        ``UNKNOWN`` with no lines or any missing input; ``FAILED`` on mixed currencies or a
        difference above ``tol``; otherwise ``PASSED``.

    Raises:
        ContractError: ``invalid_tolerance`` or ``unknown_currency``.
    """
    _check_tolerance(tol)
    if not _same_currency([*line_amounts, subtotal]):
        return Check.FAILED
    if not line_amounts or subtotal is None or any(a is None for a in line_amounts):
        return Check.UNKNOWN
    total = sum((a.amount for a in line_amounts if a is not None), Decimal(0))
    return _within(total, subtotal.amount, subtotal.currency, tol)


def v2_line_arith(
    quantity: Decimal | None, unit_price: Money | None, amount: Money | None, tol: int
) -> Check:
    """V2: quantity times unit price equals the line amount.

    Args:
        quantity: Line quantity.
        unit_price: Price per unit.
        amount: Printed line amount.
        tol: Allowed difference in minor units.

    Returns:
        ``FAILED`` on mixed currencies or a difference above ``tol``; ``UNKNOWN`` if any input
        is missing; otherwise ``PASSED``.

    Raises:
        ContractError: ``invalid_tolerance`` or ``unknown_currency``.
    """
    _check_tolerance(tol)
    if not _same_currency([unit_price, amount]):
        return Check.FAILED
    if quantity is None or unit_price is None or amount is None:
        return Check.UNKNOWN
    return _within(quantity * unit_price.amount, amount.amount, amount.currency, tol)


def v3_total_arith(
    subtotal: Money | None, tax_total: Money | None, total: Money | None, tol: int
) -> Check:
    """V3: subtotal plus tax equals the total.

    Args:
        subtotal: Printed subtotal.
        tax_total: Printed tax.
        total: Printed total.
        tol: Allowed difference in minor units.

    Returns:
        ``FAILED`` on mixed currencies or a difference above ``tol``; ``UNKNOWN`` if any input
        is missing; otherwise ``PASSED``.

    Raises:
        ContractError: ``invalid_tolerance`` or ``unknown_currency``.
    """
    _check_tolerance(tol)
    if not _same_currency([subtotal, tax_total, total]):
        return Check.FAILED
    if subtotal is None or tax_total is None or total is None:
        return Check.UNKNOWN
    return _within(subtotal.amount + tax_total.amount, total.amount, total.currency, tol)


def v4_date_order(earlier: dt.date | None, later: dt.date | None) -> Check:
    """V4: the later date (due, delivery) is not before the earlier one (invoice, pickup).

    Args:
        earlier: The date that must come first.
        later: The date that must not precede it; the same day passes.

    Returns:
        ``UNKNOWN`` if either is missing, else ``PASSED`` or ``FAILED``.
    """
    if earlier is None or later is None:
        return Check.UNKNOWN
    return Check.PASSED if later >= earlier else Check.FAILED


def v5_currency(document_currency: str | None, amounts: Sequence[Money | None]) -> Check:
    """V5: every amount is in the document's currency.

    Args:
        document_currency: The document's ISO 4217 code.
        amounts: The money fields to check.

    Returns:
        ``FAILED`` if a present amount has another currency; ``UNKNOWN`` if the currency or any
        amount is missing, or there are no amounts; otherwise ``PASSED``.
    """
    if document_currency is not None and any(
        m is not None and m.currency != document_currency for m in amounts
    ):
        return Check.FAILED
    if document_currency is None or not amounts or any(m is None for m in amounts):
        return Check.UNKNOWN
    return Check.PASSED


def _iban_checksum_ok(iban: str) -> bool:
    """Return whether an IBAN passes ISO 13616 mod-97 (rearranged number mod 97 equals 1)."""
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


def v7_format(kind: str, value: str | None) -> Check:
    """V7: an identifier has the shape its kind requires.

    Kinds: ``iban`` (shape and mod-97), ``scac`` (2-4 letters), ``mc`` (optional ``MC`` then 1-8
    digits), ``nmfc`` (one of the 18 freight classes, compared as printed).

    Args:
        kind: Which format to check.
        value: The extracted identifier.

    Returns:
        ``UNKNOWN`` if the value is missing, else ``PASSED`` or ``FAILED``.

    Raises:
        ContractError: ``unknown_format`` for any other kind; an unlisted kind is a bug, and
            answering it with a pass is how a partial enum fails permissively.
    """
    if kind not in {"iban", "scac", "mc", "nmfc"}:
        raise ContractError("unknown_format", f"no format rule for {kind!r}")
    if value is None:
        return Check.UNKNOWN
    if kind == "nmfc":
        ok = value.strip() in NMFC_CLASSES
    else:
        key = normalize_identifier(value)
        if kind == "iban":
            ok = bool(_IBAN.fullmatch(key)) and _iban_checksum_ok(key)
        elif kind == "scac":
            ok = bool(_SCAC.fullmatch(key))
        else:
            ok = bool(_MC.fullmatch(key))
    return Check.PASSED if ok else Check.FAILED


def v8_positive(value: Decimal | None, unit: str | None) -> Check:
    """V8: a weight or count is positive and its unit is one we know.

    Args:
        value: The quantity's value.
        unit: The quantity's unit code.

    Returns:
        ``UNKNOWN`` if either is missing; ``FAILED`` for a non-positive value or unknown unit;
        otherwise ``PASSED``.
    """
    if value is None or unit is None:
        return Check.UNKNOWN
    return Check.PASSED if value > 0 and unit in UNITS else Check.FAILED


def parse_quantity(text: str) -> Decimal:
    """Parse a plain decimal quantity string for V2.

    Args:
        text: Plain decimal, ``.`` as separator, no grouping.

    Returns:
        The quantity.

    Raises:
        ContractError: ``invalid_quantity`` if the text is not a finite decimal.
    """
    try:
        value = Decimal(text.strip())
    except InvalidOperation as exc:
        raise ContractError("invalid_quantity", f"{text!r} is not a decimal quantity") from exc
    if not value.is_finite():
        raise ContractError("invalid_quantity", f"{text!r} is not a finite quantity")
    return value
