"""Generator tests: labels round-trip, printed text parses back, validators pass on clean docs."""

import datetime as dt
import json
from collections import Counter
from decimal import Decimal

import pytest

from docproof.labels import decode_labels, encode_labels
from docproof.normalize import normalize_money
from docproof.parsing import Parsed, parse_amount, parse_date
from docproof.schema import Check, DocumentType, LineItem, Money, Party, Quantity, Status
from docproof.scoring import NOT_PRESENT, FieldPrediction, ScoredDocument, metrics, score
from docproof.synth.sample import SyntheticDocument, sample_document
from docproof.synth.templates import TEMPLATES, Split, template
from docproof.validators import (
    v1_line_sum,
    v2_line_arith,
    v3_total_arith,
    v4_date_order,
    v5_currency,
    v7_format,
    v8_positive,
)

PER_TEMPLATE = 8
DOCS = [sample_document(t, i, seed=2026) for t in TEMPLATES for i in range(PER_TEMPLATE)]


def test_at_least_fifty_documents_across_every_template() -> None:
    assert len(DOCS) >= 50
    assert {d.template.template_id for d in DOCS} == {t.template_id for t in TEMPLATES}
    assert len({d.doc_id for d in DOCS}) == len(DOCS)


def test_labels_round_trip_through_json_and_score_perfectly() -> None:
    scored: list[ScoredDocument] = []
    for doc in DOCS:
        raw = json.loads(json.dumps(encode_labels(doc.template.doc_type, doc.labels)))
        doc_type, gold = decode_labels(raw)
        predicted = {
            name: FieldPrediction(NOT_PRESENT if value is None else value, Status.ACCEPTED)
            for name, value in doc.labels.items()
        }
        scored.append(ScoredDocument(doc.doc_id, doc_type, gold, predicted))
    m = metrics(score(scored))
    assert m.field_f1 == 1.0
    assert m.document_complete_rate == 1.0
    assert m.accepted_precision == 1.0


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_every_printed_field_parses_back_to_its_label(doc: SyntheticDocument) -> None:
    locale = doc.template.locale
    for printed in doc.fields:
        label = doc.labels[printed.name]
        text = printed.lines[0]
        match label:
            case Money():
                parsed = parse_amount(text, locale.decimal_comma)
                assert isinstance(parsed, Parsed), text
                assert normalize_money(parsed.value, label.currency) == label, text
            case Quantity():
                number, unit = text.rsplit(" ", 1)
                parsed = parse_amount(number, locale.decimal_comma)
                assert isinstance(parsed, Parsed), text
                assert Quantity(Decimal(parsed.value), unit) == label, text
            case Party():
                assert printed.lines == tuple(x for x in (label.name, label.address) if x)
            case str():
                assert text == label
            case _:
                assert parse_date(text, locale.day_first) == Parsed(label), text


def _money(doc: SyntheticDocument, name: str) -> Money | None:
    value = doc.labels.get(name)
    return value if isinstance(value, Money) else None


def _items(doc: SyntheticDocument, name: str) -> tuple[LineItem, ...]:
    value = doc.labels.get(name)
    return value if isinstance(value, tuple) else ()


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_validators_never_fail_on_a_clean_document(doc: SyntheticDocument) -> None:
    results: list[Check] = []
    labels = doc.labels
    if doc.template.doc_type in (DocumentType.INVOICE, DocumentType.FREIGHT_INVOICE):
        items = _items(doc, "line_items") + _items(doc, "accessorials")
        subtotal = _money(doc, "subtotal") or _money(doc, "total")
        results.append(v1_line_sum([i.amount for i in items], subtotal, 0))
        results.extend(
            v2_line_arith(i.quantity.value, i.unit_price, i.amount, 0)
            for i in items
            if i.quantity is not None
        )
        if labels["tax_total"] is not None:
            results.append(
                v3_total_arith(
                    _money(doc, "subtotal"), _money(doc, "tax_total"), _money(doc, "total"), 0
                )
            )
        issued, due = labels["invoice_date"], labels["due_date"]
        if isinstance(issued, dt.date) and isinstance(due, dt.date):
            results.append(v4_date_order(issued, due))
        currency = labels["currency"]
        assert isinstance(currency, str)
        amounts = [m for m in (_money(doc, n) for n in ("subtotal", "tax_total", "total")) if m]
        results.append(v5_currency(currency, amounts))
    for name, kind in (
        ("carrier_scac", "scac"),
        ("carrier_mc_number", "mc"),
        ("freight_class", "nmfc"),
    ):
        value = labels.get(name)
        if isinstance(value, str):
            results.append(v7_format(kind, value))
    for name in ("weight", "pieces"):
        value = labels.get(name)
        if isinstance(value, Quantity):
            results.append(v8_positive(value.value, value.unit))
    assert results
    assert all(r is Check.PASSED for r in results), results


def test_same_seed_same_document_and_different_seed_differs() -> None:
    tpl = template("inv-a")
    assert sample_document(tpl, 3, 7) == sample_document(tpl, 3, 7)
    assert sample_document(tpl, 3, 7).labels != sample_document(tpl, 3, 8).labels


def test_splits_hold_out_whole_templates() -> None:
    splits = Counter((t.doc_type, t.split) for t in TEMPLATES)
    invoice_test = [
        t.template_id
        for t in TEMPLATES
        if t.doc_type is DocumentType.INVOICE and t.split is Split.TEST
    ]
    assert invoice_test == ["inv-h", "inv-i"]
    for doc_type in DocumentType:
        assert splits[(doc_type, Split.TEST)] >= 1
        assert splits[(doc_type, Split.TRAIN)] >= 1


def test_unknown_template_is_an_error() -> None:
    with pytest.raises(KeyError, match="no template"):
        template("inv-z")


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_every_line_item_label_is_printed(doc: SyntheticDocument) -> None:
    items = _items(doc, "line_items") + _items(doc, "accessorials")
    rows = doc.line_items + doc.accessorials
    assert len(items) == len(rows)
    for item, row in zip(items, rows, strict=True):
        assert row.description == item.description
        if item.quantity is not None:
            assert row.quantity is not None
            number, unit = row.quantity.rsplit(" ", 1)
            assert unit == item.quantity.unit
            parsed = parse_amount(number, doc.template.locale.decimal_comma)
            assert parsed == Parsed(str(item.quantity.value))
