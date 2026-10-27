"""The first end-to-end path: OCR page in, every field routed with its evidence out.

For each field of the recognised document type: take the extractor's candidate, ground its printed
text on the page, run the validators that read it, and route it to accepted, flagged or abstained.
A field the extractor did not find is a "not on this document" claim, which a required field can
never pass.

Confidence here is **uncalibrated**: the lowest OCR confidence among the tokens the value was
grounded on. It only orders fields by risk. Calibration is B4's work, so the accept threshold, like
the arithmetic tolerance, is a required argument with no default: both are open product decisions.

Deliberate boundary: no I/O. Reading pages and labels from disk and printing results belong to
:mod:`docproof.evaluate`.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from docproof.extract.baseline import Candidate, Extraction, extract
from docproof.grounding import Match, ground
from docproof.normalize import MINOR_UNITS
from docproof.ocr import OcrPage
from docproof.routing import CheckOutcome, Decision, Proposal, route
from docproof.schema import (
    NMFC_CLASSES,
    SCHEMAS,
    Check,
    DocumentType,
    FieldSpec,
    FieldValue,
    LineItem,
    Money,
    Quantity,
    Status,
    ValueType,
)
from docproof.scoring import NOT_PRESENT, FieldPrediction, PredictedValue
from docproof.validators import (
    v1_line_sum,
    v2_line_arith,
    v3_total_arith,
    v4_date_order,
    v5_currency,
    v7_format,
    v8_positive,
)

_FORMATS = {
    "carrier_scac": "scac",
    "carrier_mc_number": "mc",
    "freight_class": "nmfc",
    "bank_account": "iban",
}
_DATE_PAIRS = {
    "invoice_date": ("invoice_date", "due_date"),
    "due_date": ("invoice_date", "due_date"),
    "pickup_date": ("pickup_date", "delivery_date"),
    "delivery_date": ("pickup_date", "delivery_date"),
}
_MONEY_FIELDS = ("subtotal", "tax_total", "total", "rate")


@dataclass(frozen=True, slots=True)
class FieldResult:
    """One routed field and what a reviewer needs to judge it.

    Attributes:
        name: Schema field name.
        value: The proposed value, :data:`~docproof.scoring.NOT_PRESENT`, or ``None`` (abstained).
        decision: Status and reasons from routing.
        evidence: Boxes on the page where the value was found.
        confidence: The uncalibrated confidence routing used, or ``None``.
        printed: The text the value was read from.
    """

    name: str
    value: PredictedValue
    decision: Decision
    evidence: tuple[tuple[int, int, int, int], ...]
    confidence: float | None
    printed: tuple[str, ...]

    def prediction(self) -> FieldPrediction:
        """This field as the scorer reads it."""
        return FieldPrediction(self.value, self.decision.status)


@dataclass(frozen=True, slots=True)
class DocumentResult:
    """A whole page, routed.

    Attributes:
        doc_type: The recognised type, or ``None`` when nothing could be routed.
        fields: One result per schema field of that type, in schema order.
    """

    doc_type: DocumentType | None
    fields: tuple[FieldResult, ...]

    def predictions(self) -> dict[str, FieldPrediction]:
        """Every field as the scorer reads it."""
        return {f.name: f.prediction() for f in self.fields}


def _worst(results: Sequence[Check]) -> Check:
    """Combine several checks of one rule: any failure fails, else any unknown is unknown."""
    if Check.FAILED in results:
        return Check.FAILED
    if Check.UNKNOWN in results or not results:
        return Check.UNKNOWN
    return Check.PASSED


def _money(values: Mapping[str, FieldValue], name: str) -> Money | None:
    value = values.get(name)
    return value if isinstance(value, Money) else None


def _items(values: Mapping[str, FieldValue], name: str) -> tuple[LineItem, ...]:
    value = values.get(name)
    return value if isinstance(value, tuple) else ()


def _date(values: Mapping[str, FieldValue], name: str) -> dt.date | None:
    value = values.get(name)
    return value if isinstance(value, dt.date) else None


def checks_for(spec: FieldSpec, values: Mapping[str, FieldValue], tol: int) -> list[CheckOutcome]:
    """Run every validator that reads ``spec``'s field, with the other fields as inputs.

    Args:
        spec: The field.
        values: Every candidate value on the document, by field name.
        tol: Arithmetic tolerance in minor units (OPEN-1; always passed in).

    Returns:
        One outcome per validator. A validator whose inputs are missing reports ``UNKNOWN``.
    """
    outcomes: list[Check] = []
    items = _items(values, "line_items") + _items(values, "accessorials")
    subtotal, tax, total = (
        _money(values, "subtotal"),
        _money(values, "tax_total"),
        _money(values, "total"),
    )
    for rule in spec.validators:
        match rule:
            case "V1":
                target = subtotal if subtotal is not None or tax is not None else total
                outcomes.append(v1_line_sum([i.amount for i in items], target, tol))
            case "V2":
                outcomes.append(
                    _worst(
                        [
                            v2_line_arith(
                                None if i.quantity is None else i.quantity.value,
                                i.unit_price,
                                i.amount,
                                tol,
                            )
                            for i in items
                            if i.quantity is not None or i.unit_price is not None
                        ]
                    )
                )
            case "V3":
                outcomes.append(v3_total_arith(subtotal, tax, total, tol))
            case "V4":
                earlier, later = _DATE_PAIRS[spec.name]
                outcomes.append(v4_date_order(_date(values, earlier), _date(values, later)))
            case "V5":
                currency = values.get("currency")
                amounts = [_money(values, n) for n in _MONEY_FIELDS if n in values]
                outcomes.append(
                    v5_currency(currency if isinstance(currency, str) else None, amounts)
                )
            case "V7":
                value = values.get(spec.name)
                outcomes.append(
                    v7_format(_FORMATS[spec.name], value if isinstance(value, str) else None)
                )
            case "V8":
                value = values.get(spec.name)
                if isinstance(value, Quantity):
                    outcomes.append(v8_positive(value.value, value.unit))
                else:
                    outcomes.append(Check.UNKNOWN)
            case _:
                # V6 (vendor match) waits on OPEN-7; an unimplemented rule is reported as unknown,
                # never silently skipped and never a pass.
                outcomes.append(Check.UNKNOWN)
    return [CheckOutcome(result) for result in outcomes]


def _in_vocabulary(spec: FieldSpec, value: FieldValue) -> bool:
    if spec.value_type is not ValueType.CODE or not isinstance(value, str):
        return True
    match spec.name:
        case "currency":
            return value in MINOR_UNITS
        case "freight_class":
            return value.strip() in NMFC_CLASSES
        case "hazmat":
            return value.strip().upper() in {"Y", "N"}
        case _:
            return True


def _grounding(candidate: Candidate, page: OcrPage) -> tuple[bool, list[Match]]:
    """Ground every printed piece of a candidate; it is grounded only if all of them are."""
    found: list[Match] = []
    for text in candidate.printed:
        if not text.strip():
            continue
        matches = ground(text, page)
        if not matches:
            return False, found
        found.append(matches[0])
    return bool(found), found


def page_confidence(page: OcrPage) -> float | None:
    """Median OCR confidence of the page: the uncalibrated confidence of a "not printed" claim."""
    return statistics.median(t.confidence for t in page.tokens) if page.tokens else None


def _route_found(
    spec: FieldSpec,
    candidate: Candidate,
    page: OcrPage,
    values: Mapping[str, FieldValue],
    tol: int,
    threshold: float,
) -> FieldResult:
    grounded, matches = _grounding(candidate, page)
    confidence = min((m.confidence for m in matches), default=None) if grounded else None
    decision = route(
        required=spec.required,
        proposal=Proposal.VALUE,
        grounded=grounded,
        checks=checks_for(spec, values, tol),
        confidence=confidence,
        threshold=threshold,
        ambiguous=candidate.ambiguous,
        conflicting=False,
        in_vocabulary=_in_vocabulary(spec, candidate.value),
    )
    return FieldResult(
        spec.name,
        candidate.value,
        decision,
        tuple(m.box for m in matches),
        confidence,
        candidate.printed,
    )


def route_extraction(
    extraction: Extraction, page: OcrPage, *, tolerance_minor: int, threshold: float
) -> DocumentResult:
    """Route an extraction field by field.

    Args:
        extraction: The extractor's output for ``page``.
        page: The OCR page the extraction came from.
        tolerance_minor: Arithmetic tolerance for V1-V3, in minor units.
        threshold: Minimum (uncalibrated) confidence to accept.

    Returns:
        The routed document; empty when the document type was not recognised.
    """
    if extraction.doc_type is None:
        return DocumentResult(None, ())
    values = {name: c.value for name, c in extraction.candidates.items()}
    absent_confidence = page_confidence(page)
    results: list[FieldResult] = []
    for spec in SCHEMAS[extraction.doc_type]:
        candidate = extraction.candidates.get(spec.name)
        if candidate is not None:
            results.append(_route_found(spec, candidate, page, values, tolerance_minor, threshold))
            continue
        if spec.name in extraction.unreadable:
            decision = route(
                required=spec.required,
                proposal=Proposal.NONE,
                grounded=None,
                checks=[],
                confidence=None,
                threshold=threshold,
                ambiguous=False,
                conflicting=False,
                in_vocabulary=True,
            )
            results.append(FieldResult(spec.name, None, decision, (), None, ()))
            continue
        decision = route(
            required=spec.required,
            proposal=Proposal.NOT_PRESENT,
            grounded=None,
            checks=[],
            confidence=absent_confidence,
            threshold=threshold,
            ambiguous=False,
            conflicting=False,
            in_vocabulary=True,
        )
        results.append(FieldResult(spec.name, NOT_PRESENT, decision, (), absent_confidence, ()))
    return DocumentResult(extraction.doc_type, tuple(results))


def process(page: OcrPage, *, tolerance_minor: int, threshold: float) -> DocumentResult:
    """Extract and route one OCR page with the baseline extractor.

    Args:
        page: The OCR page.
        tolerance_minor: Arithmetic tolerance for V1-V3, in minor units.
        threshold: Minimum (uncalibrated) confidence to accept.

    Returns:
        The routed document.
    """
    return route_extraction(
        extract(page), page, tolerance_minor=tolerance_minor, threshold=threshold
    )


def review_order(result: DocumentResult) -> list[FieldResult]:
    """Fields in the order a reviewer should see them: flagged first, riskiest first."""
    rank = {Status.FLAGGED: 0, Status.ABSTAINED: 1, Status.ACCEPTED: 2}

    def risk(field: FieldResult) -> tuple[int, float, int]:
        return (rank[field.decision.status], field.confidence or 0.0, -len(field.decision.reasons))

    return sorted(result.fields, key=risk)


__all__ = [
    "DocumentResult",
    "FieldResult",
    "checks_for",
    "page_confidence",
    "process",
    "review_order",
    "route_extraction",
]
