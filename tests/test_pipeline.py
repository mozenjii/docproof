"""Tests for the end-to-end path on synthetic pages with perfect OCR."""

from decimal import Decimal

import pytest

from docproof.extract.baseline import Candidate, Extraction, extract
from docproof.ocr import OcrPage, page_from_words
from docproof.pipeline import checks_for, process, review_order, route_extraction
from docproof.schema import Check, DocumentType, LineItem, Money, Reason, Status, field_spec
from docproof.scoring import NOT_PRESENT, ScoredDocument, metrics, score
from docproof.synth.render import render
from docproof.synth.sample import SyntheticDocument, sample_document
from docproof.synth.templates import TEMPLATES, Split, template


def perfect_page(doc: SyntheticDocument) -> OcrPage:
    return page_from_words(1654, 2339, [(w.text, w.box) for w in render(doc).words])


TRAIN_DOCS = [
    sample_document(t, i, seed=5) for t in TEMPLATES if t.split is Split.TRAIN for i in range(3)
]


def test_on_perfect_ocr_everything_accepted_is_correct() -> None:
    scored: list[ScoredDocument] = []
    for doc in TRAIN_DOCS:
        result = process(perfect_page(doc), tolerance_minor=0, threshold=0.9)
        assert result.doc_type is doc.template.doc_type
        scored.append(
            ScoredDocument(doc.doc_id, doc.template.doc_type, doc.labels, result.predictions())
        )
    m = metrics(score(scored))
    assert m.accepted_precision == 1.0
    assert m.straight_through_rate is not None
    assert m.straight_through_rate > 0.8


def test_a_missing_required_field_goes_to_review() -> None:
    doc = sample_document(template("inv-h"), 0, seed=5)  # "Factuur" is not a known caption
    result = process(perfect_page(doc), tolerance_minor=0, threshold=0.9)
    invoice_number = next(f for f in result.fields if f.name == "invoice_number")
    assert invoice_number.value is NOT_PRESENT
    assert invoice_number.decision.status is Status.FLAGGED
    assert invoice_number.decision.reasons == (Reason.REQUIRED_NOT_FOUND,)


def test_a_value_not_on_the_page_is_ungrounded() -> None:
    doc = sample_document(template("inv-a"), 0, seed=5)
    page = perfect_page(doc)
    extraction = extract(page)
    forged = Candidate("payment_terms", "Net 90", ("Net 90",))
    tampered = Extraction(
        extraction.doc_type, {**extraction.candidates, "payment_terms": forged}, set()
    )
    result = route_extraction(tampered, page, tolerance_minor=0, threshold=0.9)
    terms = next(f for f in result.fields if f.name == "payment_terms")
    assert terms.decision.status is Status.FLAGGED
    assert Reason.UNGROUNDED in terms.decision.reasons
    assert terms.evidence == ()


def test_an_unrecognised_page_routes_nothing() -> None:
    page = page_from_words(100, 100, [("RECEIPT", (0, 0, 50, 10))])
    result = process(page, tolerance_minor=0, threshold=0.9)
    assert result.doc_type is None
    assert result.fields == ()


def test_review_order_puts_flagged_and_least_confident_first() -> None:
    doc = sample_document(template("inv-h"), 0, seed=5)
    ordered = review_order(process(perfect_page(doc), tolerance_minor=0, threshold=0.9))
    statuses = [f.decision.status for f in ordered]
    assert statuses == sorted(
        statuses, key=lambda s: {Status.FLAGGED: 0, Status.ABSTAINED: 1, Status.ACCEPTED: 2}[s]
    )


def usd(amount: str) -> Money:
    return Money(Decimal(amount), "USD")


def test_line_sum_checks_against_total_when_no_subtotal_or_tax() -> None:
    spec = field_spec(DocumentType.INVOICE, "line_items")
    values = {
        "line_items": (LineItem("a", usd("5.00")), LineItem("b", usd("7.00"))),
        "total": usd("12.00"),
    }
    v1 = checks_for(spec, values, 0)[0]
    assert v1.result is Check.PASSED
    values["subtotal"] = usd("11.00")
    assert checks_for(spec, values, 0)[0].result is Check.FAILED


def test_vendor_match_is_unknown_until_it_exists() -> None:
    (outcome,) = checks_for(field_spec(DocumentType.INVOICE, "vendor"), {}, 0)
    assert outcome.result is Check.UNKNOWN
    assert not outcome.advisory


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_threshold_extremes(threshold: float) -> None:
    doc = TRAIN_DOCS[0]
    result = process(perfect_page(doc), tolerance_minor=0, threshold=threshold)
    assert {f.name for f in result.fields} == set(doc.labels)
