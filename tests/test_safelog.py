"""Tests for safe logging: redaction by sensitivity, no raw strings, errors without messages."""

import logging
from decimal import Decimal
from enum import Enum
from typing import cast

import pytest

from docproof.errors import ContractError
from docproof.normalize import normalize_money
from docproof.safelog import LogValue, Redactor, Safe, SafeLogger
from docproof.schema import SCHEMAS, DocumentType, Money, Party, Sensitivity, field_spec
from docproof.synth.sample import sample_document
from docproof.synth.templates import TEMPLATES

KEY = b"k" * 32
INV = DocumentType.INVOICE


def test_redaction_follows_sensitivity() -> None:
    r = Redactor(KEY)
    assert r.redact(field_spec(INV, "currency"), "USD") == Safe("USD")
    assert r.redact(field_spec(INV, "buyer"), Party("Jane Roe", "1 Elm St")) == Safe("[personal]")
    assert r.redact(field_spec(INV, "bank_account"), "GB82WEST12345698765432") == Safe(
        "[financial]"
    )
    assert r.redact(field_spec(INV, "total"), None) == Safe("[none]")
    hashed = r.redact(field_spec(INV, "total"), Money(Decimal("12.50"), "USD")).text
    assert hashed.startswith("hmac:")
    assert len(hashed) == len("hmac:") + 12
    assert "12.50" not in hashed


def test_business_hashes_correlate_within_a_key_only() -> None:
    total = field_spec(INV, "total")
    a, b = Money(Decimal("12.50"), "USD"), Money(Decimal("12.51"), "USD")
    same_key = Redactor(KEY)
    assert same_key.redact(total, a) == same_key.redact(total, a)
    assert same_key.redact(total, a) != same_key.redact(total, b)
    assert Redactor(b"x" * 32).redact(total, a) != same_key.redact(total, a)
    # The default key is random, so two processes cannot be joined on a hash.
    assert Redactor().redact(total, a) != Redactor().redact(total, a)


def test_every_schema_field_has_a_rendering() -> None:
    r = Redactor(KEY)
    for doc_type in DocumentType:
        for spec in SCHEMAS[doc_type]:
            assert isinstance(r.redact(spec, None), Safe)


class Colour(Enum):
    """A stand-in enum for log fields."""

    RED = "red"


def test_logger_renders_allowed_fields_sorted(caplog: pytest.LogCaptureFixture) -> None:
    log = SafeLogger("docproof.test")
    with caplog.at_level(logging.INFO, logger="docproof.test"):
        log.info("doc.routed", fields=12, ratio=0.5, ok=True, colour=Colour.RED, doc=Safe("d-1"))
        log.debug("doc.ignored", n=1)  # below the level: not emitted
        log.warning("doc.slow", seconds=2.5)
    assert caplog.messages == [
        "doc.routed colour=red doc=d-1 fields=12 ok=True ratio=0.5",
        "doc.slow seconds=2.5",
    ]


@pytest.mark.parametrize("bad", ["Jane Roe", b"bytes", ["a"], {"a": 1}, Decimal("1.5")])
def test_logger_refuses_raw_values(bad: object) -> None:
    log = SafeLogger("docproof.test")
    with pytest.raises(TypeError, match="wrap text in Safe"):
        # The cast hides the bad type from the checker so the runtime guard is what gets tested.
        log.info("doc.routed", value=cast(LogValue, bad))


@pytest.mark.parametrize("event", ["Routed doc", "doc.routed for Jane", "doc..x", "", "DOC.x"])
def test_event_names_cannot_carry_free_text(event: str) -> None:
    with pytest.raises(ValueError, match="event"):
        SafeLogger("docproof.test").info(event)


def test_field_names_are_checked() -> None:
    with pytest.raises(ValueError, match="field name"):
        SafeLogger("docproof.test").info("doc.x", **{"Jane Roe": 1})


def test_errors_are_logged_by_type_and_code_only(caplog: pytest.LogCaptureFixture) -> None:
    log = SafeLogger("docproof.test")
    with caplog.at_level(logging.ERROR, logger="docproof.test"):
        try:
            normalize_money("1234.567", "USD")
        except ContractError as exc:
            log.error("money.invalid", exc, doc=Safe("d-9"))
        log.error("io.failed", OSError("C:/clients/Jane Roe/invoice.pdf not found"))
    text = "\n".join(caplog.messages)
    assert "error_code=excess_precision" in text
    assert "error_type=ContractError" in text
    assert "error_type=OSError" in text
    assert "1234.567" not in text
    assert "Jane" not in text
    assert all(record.exc_info is None for record in caplog.records)


def test_no_document_text_reaches_the_log(caplog: pytest.LogCaptureFixture) -> None:
    # Log every field of generated documents and look for any printed value in the output.
    redactor = Redactor(KEY)
    log = SafeLogger("docproof.test")
    leaks: list[str] = []
    with caplog.at_level(logging.INFO, logger="docproof.test"):
        for tpl in TEMPLATES:
            doc = sample_document(tpl, 0, seed=31)
            for spec in SCHEMAS[tpl.doc_type]:
                value = doc.labels[spec.name]
                log.info("field.seen", field=Safe(spec.name), value=redactor.redact(spec, value))
            text = "\n".join(caplog.messages)
            for printed in doc.fields:
                spec = field_spec(tpl.doc_type, printed.name)
                if spec.sensitivity is Sensitivity.PUBLIC:
                    continue
                leaks.extend(line for line in printed.lines if line in text)
            caplog.clear()
    assert leaks == []
