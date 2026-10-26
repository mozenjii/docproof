"""Tests for the scorer: value matching, line-item pairing, tallies, metrics and the bootstrap."""

import datetime as dt
from decimal import Decimal

import pytest

from docproof.schema import (
    SCHEMAS,
    DocumentType,
    FieldValue,
    LineItem,
    Money,
    Party,
    Quantity,
    Status,
    ValueType,
)
from docproof.scoring import (
    NOT_PRESENT,
    Counts,
    FieldPrediction,
    ScoredDocument,
    bootstrap,
    line_item_counts,
    metrics,
    score,
    score_document,
    values_match,
)

ACC, FLAG, ABST = Status.ACCEPTED, Status.FLAGGED, Status.ABSTAINED


def usd(amount: str) -> Money:
    return Money(Decimal(amount), "USD")


def item(description: str, amount: str) -> LineItem:
    return LineItem(description, usd(amount))


def bol_gold(**overrides: FieldValue | None) -> dict[str, FieldValue | None]:
    gold: dict[str, FieldValue | None] = dict.fromkeys(
        (s.name for s in SCHEMAS[DocumentType.BILL_OF_LADING]), None
    )
    gold.update(
        bol_number="BOL-1001",
        ship_date=dt.date(2026, 10, 1),
        shipper=Party("Northwind Freight Ltd", "1 Dock Rd"),
        consignee=Party("Contoso Retail", None),
    )
    gold.update(overrides)
    return gold


def all_correct(gold: dict[str, FieldValue | None]) -> dict[str, FieldPrediction]:
    return {
        name: FieldPrediction(NOT_PRESENT if value is None else value, ACC)
        for name, value in gold.items()
    }


@pytest.mark.parametrize(
    ("value_type", "gold", "predicted", "expected"),
    [
        (ValueType.TEXT, "Net 30", "net  30.", True),
        (ValueType.IDENTIFIER, "INV-001", "inv 001", True),
        (ValueType.IDENTIFIER, "INV-001", "INV-0O1", False),
        (ValueType.CODE, "77.5", " 77.5", True),
        (ValueType.MONEY, usd("12.50"), Money(Decimal("12.5"), "USD"), True),
        (ValueType.MONEY, usd("12.50"), Money(Decimal("12.50"), "CAD"), False),
        (ValueType.DATE, dt.date(2026, 4, 3), dt.date(2026, 3, 4), False),
        (ValueType.QUANTITY, Quantity(Decimal(5), "PLT"), Quantity(Decimal("5.0"), "PLT"), True),
        (ValueType.PARTY, Party("ACME Ltd", None), Party("acme ltd", None), True),
        (ValueType.PARTY, Party("ACME Ltd", "x"), Party("ACME Ltd", None), False),
        (ValueType.MONEY, usd("1.00"), "1.00", False),  # wrong type is wrong, not an error
        (ValueType.TEXT, "x", usd("1.00"), False),
    ],
)
def test_values_match(
    value_type: ValueType, gold: object, predicted: object, expected: bool
) -> None:
    assert values_match(value_type, gold, predicted) is expected


def test_line_items_are_not_scored_as_one_value() -> None:
    with pytest.raises(ValueError, match="line items"):
        values_match(ValueType.LINE_ITEMS, (), ())


def test_line_items_pair_by_content_not_order() -> None:
    gold = [item("Pallet fee", "10.00"), item("Fuel", "5.00")]
    predicted = [item("Fuel", "5.00"), item("Pallet fee", "10.00")]
    assert line_item_counts(gold, predicted) == (2, 0, 0)


def test_line_item_partial_match_is_wrong_and_extras_are_false_positives() -> None:
    gold = [item("Pallet fee", "10.00"), item("Fuel", "5.00")]
    predicted = [item("Pallet fee", "10.00"), item("Fuel", "5.50"), item("Toll", "2.00")]
    assert line_item_counts(gold, predicted) == (1, 2, 1)
    assert line_item_counts([], predicted) == (0, 3, 0)
    assert line_item_counts(gold, []) == (0, 0, 2)


def test_perfect_document() -> None:
    gold = bol_gold()
    counts = score_document(
        ScoredDocument("d1", DocumentType.BILL_OF_LADING, gold, all_correct(gold))
    )
    m = metrics(counts)
    assert counts.tp == 4
    assert counts.fp == counts.fn == 0
    assert m.field_f1 == 1.0
    assert m.document_complete_rate == 1.0
    assert m.straight_through_rate == 1.0
    assert m.accepted_precision == 1.0


def test_wrong_value_counts_as_both_false_positive_and_negative() -> None:
    gold = bol_gold()
    predicted = all_correct(gold) | {"bol_number": FieldPrediction("BOL-1002", ACC)}
    counts = score_document(ScoredDocument("d", DocumentType.BILL_OF_LADING, gold, predicted))
    assert (counts.tp, counts.fp, counts.fn) == (3, 1, 1)
    assert counts.complete_documents == 0
    assert counts.accepted_correct == counts.accepted - 1


def test_not_present_abstain_and_hallucination_are_kept_apart() -> None:
    gold = bol_gold()
    predicted = all_correct(gold) | {
        "pro_number": FieldPrediction("PRO-9", FLAG),  # not on the page: a hallucination
        "carrier_scac": FieldPrediction(None, ABST),  # abstaining on an absent field costs nothing
        "consignee": FieldPrediction(NOT_PRESENT, FLAG),  # claiming absent a present field
    }
    counts = score_document(ScoredDocument("d", DocumentType.BILL_OF_LADING, gold, predicted))
    assert (counts.tp, counts.fp, counts.fn) == (3, 1, 1)
    assert counts.complete_documents == 0  # consignee is required


def test_missing_prediction_is_an_abstention() -> None:
    gold = bol_gold()
    predicted = all_correct(gold)
    del predicted["bol_number"]
    counts = score_document(ScoredDocument("d", DocumentType.BILL_OF_LADING, gold, predicted))
    assert counts.fn == 1
    assert counts.complete_documents == 0


def test_accepted_precision_counts_only_accepted_fields() -> None:
    gold = bol_gold()
    predicted = {n: FieldPrediction(p.value, FLAG) for n, p in all_correct(gold).items()}
    m = metrics(score_document(ScoredDocument("d", DocumentType.BILL_OF_LADING, gold, predicted)))
    assert m.straight_through_rate == 0.0
    assert m.accepted_precision is None  # undefined, not 0 or 1


def test_line_items_feed_field_f1_item_by_item() -> None:
    names = [s.name for s in SCHEMAS[DocumentType.INVOICE]]
    gold: dict[str, FieldValue | None] = dict.fromkeys(names, None)
    gold.update(
        invoice_number="1",
        invoice_date=dt.date(2026, 1, 1),
        vendor=Party("V", None),
        currency="USD",
        total=usd("15.00"),
        line_items=(item("A", "10.00"), item("B", "5.00")),
    )
    predicted = all_correct(gold) | {
        "line_items": FieldPrediction((item("A", "10.00"),), ACC),
    }
    counts = score_document(ScoredDocument("d", DocumentType.INVOICE, gold, predicted))
    assert (counts.tp, counts.fn) == (5 + 1, 1)
    assert counts.complete_documents == 1  # line items are optional
    assert counts.accepted_correct == counts.accepted - 1


def test_incomplete_labels_and_unknown_fields_are_errors() -> None:
    gold = bol_gold()
    del gold["hazmat"]
    with pytest.raises(KeyError, match="gold labels missing"):
        score_document(ScoredDocument("d", DocumentType.BILL_OF_LADING, gold, {}))
    with pytest.raises(KeyError, match="not in the schema"):
        score_document(
            ScoredDocument(
                "d", DocumentType.BILL_OF_LADING, bol_gold(), {"total": FieldPrediction(None, ABST)}
            )
        )


def test_score_pools_documents() -> None:
    gold = bol_gold()
    doc = ScoredDocument("d", DocumentType.BILL_OF_LADING, gold, all_correct(gold))
    pooled = score([doc, doc, doc])
    assert pooled.documents == 3
    assert pooled.tp == 12


def test_bootstrap_brackets_the_estimate_and_is_reproducible() -> None:
    good = Counts(tp=1, documents=1, complete_documents=1, fields=1, accepted=1, accepted_correct=1)
    bad = Counts(fp=1, fn=1, documents=1, fields=1, accepted=1)
    per_doc = [good] * 8 + [bad] * 2
    first = bootstrap(per_doc, lambda m: m.accepted_precision, resamples=500, seed=7)
    again = bootstrap(per_doc, lambda m: m.accepted_precision, resamples=500, seed=7)
    assert first == again
    assert first.estimate == pytest.approx(0.8)
    assert first.low is not None
    assert first.high is not None
    assert first.low < 0.8 < first.high
    assert first.resamples == 500


def test_bootstrap_edge_cases() -> None:
    empty = bootstrap([], lambda m: m.field_f1, resamples=10, seed=1)
    assert (empty.estimate, empty.low, empty.high) == (None, None, None)
    never = bootstrap([Counts(documents=1)], lambda m: m.accepted_precision, resamples=10, seed=1)
    assert never.low is None
    assert never.resamples == 0
    with pytest.raises(ValueError, match="resamples"):
        bootstrap([], lambda m: m.field_f1, resamples=0, seed=1)
    with pytest.raises(ValueError, match="level"):
        bootstrap([], lambda m: m.field_f1, resamples=5, seed=1, level=1.0)
