"""Score routed predictions against gold labels: field F1, completeness, straight-through rate.

The headline number is not accuracy but how many fields are accepted without a human, and how
often an accepted field is right. Both are computed here, per document and pooled, so they can
be bootstrapped over documents.

Deliberate boundary: scoring compares values through :mod:`docproof.normalize` and never
re-parses text. A gold value of ``None`` means "this field is not on the document"; a prediction
says so explicitly with :data:`NOT_PRESENT`, and an abstention is ``None``. The three are kept
apart because conflating "not printed" with "no answer" is how a scorer rewards silence.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from docproof.assignment import max_weight_assignment
from docproof.normalize import normalize_identifier, normalize_text
from docproof.schema import (
    SCHEMAS,
    DocumentType,
    FieldValue,
    LineItem,
    Party,
    Status,
    ValueType,
)


class _Absent(Enum):
    """Sentinel type for an explicit "this field is not on the document" claim."""

    NOT_PRESENT = "not_present"


NOT_PRESENT = _Absent.NOT_PRESENT
"""A prediction that the field does not appear on the document."""


@dataclass(frozen=True, slots=True)
class FieldPrediction:
    """One routed field.

    Attributes:
        value: The proposed value, :data:`NOT_PRESENT`, or ``None`` when the system abstained.
        status: The routing outcome.
    """

    value: FieldValue | _Absent | None
    status: Status


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    """Gold labels and routed predictions for one document.

    Attributes:
        doc_id: Opaque document id.
        doc_type: Document family; decides the field list.
        gold: Every schema field's true value, ``None`` when not on the document.
        predicted: Routed predictions; a missing field counts as an abstention.
    """

    doc_id: str
    doc_type: DocumentType
    gold: Mapping[str, FieldValue | None]
    predicted: Mapping[str, FieldPrediction]


@dataclass(slots=True)
class Counts:
    """Additive tallies; the metrics are ratios of these.

    Attributes:
        tp: Gold values predicted correctly.
        fp: Predicted values that are wrong or not on the document.
        fn: Gold values missed or predicted wrongly.
        documents: Documents scored.
        complete_documents: Documents with every required field correct.
        fields: Scalar fields routed (line-item lists count once).
        accepted: Fields routed ``accepted``.
        accepted_correct: Accepted fields that were correct.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    documents: int = 0
    complete_documents: int = 0
    fields: int = 0
    accepted: int = 0
    accepted_correct: int = 0

    def add(self, other: Counts) -> None:
        """Add another tally into this one."""
        for name in self.__slots__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass(frozen=True, slots=True)
class Metrics:
    """Ratios derived from :class:`Counts`; ``None`` where the denominator is zero.

    Attributes:
        field_f1: Micro-averaged F1 over (document, field) pairs and line items.
        document_complete_rate: Share of documents with every required field correct.
        straight_through_rate: Share of fields accepted without review.
        accepted_precision: Share of accepted fields that were correct.
    """

    field_f1: float | None
    document_complete_rate: float | None
    straight_through_rate: float | None
    accepted_precision: float | None


def _ratio(numerator: int, denominator: int) -> float | None:
    """Divide, or return ``None`` for an empty denominator instead of inventing 0 or 1."""
    return numerator / denominator if denominator else None


def metrics(counts: Counts) -> Metrics:
    """Turn tallies into the four reported metrics.

    Args:
        counts: Pooled tallies.

    Returns:
        The metrics.
    """
    return Metrics(
        field_f1=_ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn),
        document_complete_rate=_ratio(counts.complete_documents, counts.documents),
        straight_through_rate=_ratio(counts.accepted, counts.fields),
        accepted_precision=_ratio(counts.accepted_correct, counts.accepted),
    )


def _optional_text_equal(a: str | None, b: str | None) -> bool:
    """Compare optional free text; two absences are equal."""
    if a is None or b is None:
        return a is b
    return normalize_text(a) == normalize_text(b)


def values_match(value_type: ValueType, gold: object, predicted: object) -> bool:
    """Decide whether a predicted scalar value equals the gold one.

    Args:
        value_type: How to compare.
        gold: Gold value (never ``None`` here).
        predicted: Predicted value.

    Returns:
        Whether they are the same value after normalisation. A value of the wrong Python type is
        simply wrong, not an error, because predictions come from models.

    Raises:
        ValueError: For ``LINE_ITEMS``, which is scored item by item, not as one value.
    """
    match value_type:
        case ValueType.TEXT:
            return (
                isinstance(gold, str)
                and isinstance(predicted, str)
                and normalize_text(gold) == normalize_text(predicted)
            )
        case ValueType.IDENTIFIER:
            return (
                isinstance(gold, str)
                and isinstance(predicted, str)
                and normalize_identifier(gold) == normalize_identifier(predicted)
            )
        case ValueType.CODE:
            return (
                isinstance(gold, str)
                and isinstance(predicted, str)
                and gold.strip().upper() == predicted.strip().upper()
            )
        case ValueType.PARTY:
            return (
                isinstance(gold, Party)
                and isinstance(predicted, Party)
                and normalize_text(gold.name) == normalize_text(predicted.name)
                and _optional_text_equal(gold.address, predicted.address)
            )
        case ValueType.MONEY | ValueType.DATE | ValueType.QUANTITY:
            return type(gold) is type(predicted) and gold == predicted
        case ValueType.LINE_ITEMS:
            raise ValueError("line items are scored with line_item_counts")


def _item_components_equal(gold: LineItem, predicted: LineItem) -> list[bool]:
    """Per-component agreement of two line items: description, amount, quantity, unit price."""
    return [
        normalize_text(gold.description) == normalize_text(predicted.description),
        gold.amount == predicted.amount,
        gold.quantity == predicted.quantity,
        gold.unit_price == predicted.unit_price,
    ]


def line_item_counts(
    gold: Sequence[LineItem], predicted: Sequence[LineItem]
) -> tuple[int, int, int]:
    """Count line items as true positives, false positives and false negatives.

    Items are paired one-to-one to maximise component agreement, so order does not matter. A pair
    is a true positive only if every component agrees; anything less is a wrong item, counted as
    both a false positive and a false negative. Unpaired items are plain FPs or FNs.

    Args:
        gold: Gold items.
        predicted: Predicted items.

    Returns:
        ``(tp, fp, fn)``.
    """
    if not gold or not predicted:
        return 0, len(predicted), len(gold)
    agreement = [[_item_components_equal(g, p) for p in predicted] for g in gold]
    scores = [[float(sum(cell)) for cell in row] for row in agreement]
    pairs = max_weight_assignment(scores)
    tp = sum(1 for g, p in pairs if all(agreement[g][p]))
    return tp, len(predicted) - tp, len(gold) - tp


def _score_field(
    value_type: ValueType, gold: FieldValue | None, prediction: FieldPrediction
) -> tuple[Counts, bool]:
    """Score one field. Returns its tallies and whether the prediction was correct."""
    counts = Counts(fields=1, accepted=int(prediction.status is Status.ACCEPTED))
    value = prediction.value
    if value_type is ValueType.LINE_ITEMS:
        gold_items = gold if isinstance(gold, tuple) else ()
        pred_items = value if isinstance(value, tuple) else ()
        counts.tp, counts.fp, counts.fn = line_item_counts(gold_items, pred_items)
        correct = counts.fp == 0 and counts.fn == 0
    elif gold is None:
        counts.fp = int(value is not None and value is not NOT_PRESENT)
        correct = value is NOT_PRESENT
    elif value is None or value is NOT_PRESENT:
        counts.fn = 1
        correct = False
    elif values_match(value_type, gold, value):
        counts.tp = 1
        correct = True
    else:
        counts.fp = counts.fn = 1
        correct = False
    counts.accepted_correct = int(correct and prediction.status is Status.ACCEPTED)
    return counts, correct


_ABSTAINED = FieldPrediction(None, Status.ABSTAINED)


def score_document(document: ScoredDocument) -> Counts:
    """Score every schema field of one document.

    Args:
        document: Labels and predictions.

    Returns:
        The document's tallies.

    Raises:
        KeyError: If the gold labels omit a schema field, or either side names a field the schema
            does not have. Incomplete labels would silently shrink the denominator.
    """
    specs = SCHEMAS[document.doc_type]
    names = {s.name for s in specs}
    unknown = (set(document.gold) | set(document.predicted)) - names
    if unknown:
        raise KeyError(f"{document.doc_id}: fields not in the schema: {sorted(unknown)}")
    missing = names - set(document.gold)
    if missing:
        raise KeyError(f"{document.doc_id}: gold labels missing: {sorted(missing)}")
    total = Counts(documents=1)
    complete = True
    for spec in specs:
        prediction = document.predicted.get(spec.name, _ABSTAINED)
        counts, correct = _score_field(spec.value_type, document.gold[spec.name], prediction)
        total.add(counts)
        complete = complete and (correct or not spec.required)
    total.complete_documents = int(complete)
    return total


def score(documents: Sequence[ScoredDocument]) -> Counts:
    """Pool the tallies of many documents.

    Args:
        documents: Documents to score.

    Returns:
        Pooled tallies.
    """
    total = Counts()
    for document in documents:
        total.add(score_document(document))
    return total


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a percentile bootstrap interval.

    Attributes:
        estimate: The metric on the full set.
        low: Lower bound.
        high: Upper bound.
        level: Coverage, for example 0.95.
        resamples: Bootstrap resamples whose metric was defined.
    """

    estimate: float | None
    low: float | None
    high: float | None
    level: float
    resamples: int = field(default=0)


def bootstrap(
    per_document: Sequence[Counts],
    metric: Callable[[Metrics], float | None],
    *,
    resamples: int,
    seed: int,
    level: float = 0.95,
) -> Interval:
    """Percentile bootstrap of a metric, resampling documents with replacement.

    Documents, not fields, are the unit: fields on one document share a layout and a scan, so
    resampling fields would understate the uncertainty.

    Args:
        per_document: Each document's tallies, from :func:`score_document`.
        metric: Picks one metric, for example ``lambda m: m.field_f1``.
        resamples: Number of bootstrap resamples.
        seed: Seed for reproducibility.
        level: Interval coverage in (0, 1).

    Returns:
        The interval. Bounds are ``None`` when no resample had a defined metric.

    Raises:
        ValueError: If ``resamples`` < 1 or ``level`` is not in (0, 1).
    """
    if resamples < 1 or not 0.0 < level < 1.0:
        raise ValueError("need resamples >= 1 and 0 < level < 1")
    pooled = Counts()
    for counts in per_document:
        pooled.add(counts)
    estimate = metric(metrics(pooled))
    if not per_document:
        return Interval(estimate, None, None, level)
    rng = random.Random(seed)  # reproducible resampling, not a secret
    values: list[float] = []
    n = len(per_document)
    for _ in range(resamples):
        sample = Counts()
        for _ in range(n):
            sample.add(per_document[rng.randrange(n)])
        value = metric(metrics(sample))
        if value is not None:
            values.append(value)
    if not values:
        return Interval(estimate, None, None, level)
    values.sort()
    tail = (1.0 - level) / 2.0
    low = values[int(tail * (len(values) - 1))]
    high = values[int((1.0 - tail) * (len(values) - 1))]
    return Interval(estimate, low, high, level, len(values))
