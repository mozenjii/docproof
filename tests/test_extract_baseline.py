"""Tests for the baseline extractor: its pieces, and its behaviour on whole synthetic pages."""

import datetime as dt
from decimal import Decimal

import pytest

from docproof.extract.baseline import (
    Locale,
    caption_values,
    classify,
    extract,
    find_captions,
    is_caption_line,
    read_locale,
    training_captions,
    typed_value,
)
from docproof.extract.layout import lines
from docproof.ocr import OcrPage, page_from_words
from docproof.schema import DocumentType, Money, Quantity, ValueType
from docproof.synth.render import render
from docproof.synth.sample import SyntheticDocument, sample_document
from docproof.synth.templates import TEMPLATES, Split, template

INVOICE_CAPTIONS = training_captions(DocumentType.INVOICE)


def perfect_page(doc: SyntheticDocument) -> OcrPage:
    return page_from_words(1654, 2339, [(w.text, w.box) for w in render(doc).words])


def text_page(*rows: str) -> OcrPage:
    words: list[tuple[str, tuple[int, int, int, int]]] = []
    for row, text in enumerate(rows):
        x = 100
        for word in text.split():
            words.append((word, (x, 100 + 50 * row, x + 12 * len(word), 130 + 50 * row)))
            x += 12 * len(word) + 12
    return page_from_words(2000, 2000, words)


def test_training_captions_ignore_held_out_templates() -> None:
    assert "invoiceno." in INVOICE_CAPTIONS
    assert "factuur" not in INVOICE_CAPTIONS  # inv-h is a test template
    assert "ref" not in INVOICE_CAPTIONS  # inv-i is a test template
    assert INVOICE_CAPTIONS["total"] == "total"  # generic fallback caption


def test_captions_prefer_the_longest_and_split_a_line() -> None:
    values = caption_values("Invoice No.: INV-1 Invoice Date: 01/02/2026", INVOICE_CAPTIONS)
    assert values == {"invoice_number": "INV-1", "invoice_date": "01/02/2026"}
    assert caption_values("Subtotal: 10.00", INVOICE_CAPTIONS) == {"subtotal": "10.00"}


def test_a_dropped_colon_still_matches_whole_words_only() -> None:
    assert [h.field for h in find_captions("Total 12.50", INVOICE_CAPTIONS)] == ["total"]
    assert find_captions("Totality 12.50", INVOICE_CAPTIONS) == []


def test_an_unknown_caption_ends_a_value() -> None:
    caps = training_captions(DocumentType.BILL_OF_LADING)
    assert caption_values("Class: 55 HM: N", caps)["freight_class"] == "55"
    assert is_caption_line("HM: N", caps)
    assert not is_caption_line("Northwind Freight Ltd", caps)


def test_classify_reads_the_title() -> None:
    assert classify(lines(text_page("BILL OF LADING", "x"))) is DocumentType.BILL_OF_LADING
    assert classify(lines(text_page("FREIGHT INVOICE"))) is DocumentType.FREIGHT_INVOICE
    assert classify(lines(text_page("RECEIPT"))) is None


def test_locale_comes_from_evidence_on_the_page() -> None:
    page_lines = lines(text_page("Date: 13/04/2026", "Total: 1.234,56 EUR"))
    assert read_locale(page_lines) == Locale(day_first=True, decimal_comma=True, currency="EUR")
    contradicting = lines(text_page("13/04/2026 04/13/2026"))
    assert read_locale(contradicting).day_first is None


def test_typed_values_flag_ambiguity_instead_of_guessing() -> None:
    unknown = Locale(None, None, "USD")
    assert typed_value(ValueType.DATE, "03/04/2026", unknown) == (dt.date(2026, 3, 4), True)
    assert typed_value(ValueType.DATE, "13/04/2026", unknown) == (dt.date(2026, 4, 13), False)
    assert typed_value(ValueType.MONEY, "1,234.50", unknown) == (
        Money(Decimal("1234.50"), "USD"),
        False,
    )
    weight = typed_value(ValueType.QUANTITY, "9,169 LB", unknown)
    assert weight is not None
    assert weight[1] is True
    assert typed_value(ValueType.QUANTITY, "12 PLT", unknown) == (
        Quantity(Decimal(12), "PLT"),
        False,
    )
    assert typed_value(ValueType.MONEY, "12.50", Locale(None, None, None)) is None  # no currency
    assert typed_value(ValueType.DATE, "tomorrow", unknown) is None


TRAIN_DOCS = [
    sample_document(t, i, seed=99) for t in TEMPLATES if t.split is Split.TRAIN for i in range(3)
]


@pytest.mark.parametrize("doc", TRAIN_DOCS, ids=[d.doc_id for d in TRAIN_DOCS])
def test_on_clean_training_pages_every_candidate_is_right_or_flagged(
    doc: SyntheticDocument,
) -> None:
    extraction = extract(perfect_page(doc))
    assert extraction.doc_type is doc.template.doc_type
    for name, candidate in extraction.candidates.items():
        assert candidate.value == doc.labels[name] or candidate.ambiguous, name
    present = {n for n, v in doc.labels.items() if v is not None}
    assert present <= set(extraction.candidates) | extraction.unreadable


def test_held_out_captions_are_missed_not_misread() -> None:
    doc = sample_document(template("inv-h"), 0, seed=99)
    extraction = extract(perfect_page(doc))
    assert "invoice_number" not in extraction.candidates  # printed as "Factuur"
    for name, candidate in extraction.candidates.items():
        assert candidate.value == doc.labels[name] or candidate.ambiguous, name


def test_an_unrecognised_document_yields_nothing() -> None:
    extraction = extract(text_page("RECEIPT", "Total: 12.50 USD"))
    assert extraction.doc_type is None
    assert extraction.candidates == {}


def test_money_edge_cases() -> None:
    unknown = Locale(None, None, None)
    assert typed_value(ValueType.MONEY, "12.50 XYZ", unknown) is None  # not a currency we know
    # Four decimals: neither a USD amount nor 3-digit grouping.
    assert typed_value(ValueType.MONEY, "12.5055 USD", unknown) is None
    # Three after the dot: not cents, so the dot must be grouping.
    assert typed_value(ValueType.MONEY, "12.505 USD", unknown) == (
        Money(Decimal("12505.00"), "USD"),
        False,
    )
    # USD has two decimals, so "1,234" can only be one thousand two hundred and thirty-four.
    assert typed_value(ValueType.MONEY, "1,234 USD", unknown) == (
        Money(Decimal("1234.00"), "USD"),
        False,
    )
    # The dinar has three decimals, so both readings are real amounts.
    dinar = typed_value(ValueType.MONEY, "1,234 KWD", unknown)
    assert dinar is not None
    assert dinar[1] is True
    assert typed_value(ValueType.MONEY, "abc USD", unknown) is None
    assert typed_value(ValueType.QUANTITY, "12 TONNE", unknown) is None
    assert typed_value(ValueType.LINE_ITEMS, "x", unknown) is None
