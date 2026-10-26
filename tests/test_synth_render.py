"""Renderer tests: word boxes are complete, on the page, non-overlapping, and survive skew."""

import hashlib
from collections import defaultdict

import pytest

from docproof.synth.render import (
    LEVELS,
    PAGE_SIZE,
    Degradation,
    WordBox,
    degrade,
    render,
)
from docproof.synth.sample import SyntheticDocument, sample_document
from docproof.synth.templates import TEMPLATES, template

DOCS = [sample_document(t, i, seed=2026) for t in TEMPLATES for i in range(4)]


def _overlap(a: WordBox, b: WordBox) -> bool:
    return (
        a.box[0] < b.box[2] and b.box[0] < a.box[2] and a.box[1] < b.box[3] and b.box[1] < a.box[3]
    )


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_every_printed_value_is_boxed_word_for_word(doc: SyntheticDocument) -> None:
    by_field: dict[str | None, list[str]] = defaultdict(list)
    for word in render(doc).words:
        by_field[word.field].append(word.text)
    for printed in doc.fields:
        assert by_field[printed.name] == " ".join(printed.lines).split()
    for name, rows in (("line_items", doc.line_items), ("accessorials", doc.accessorials)):
        cells = [c for r in rows for c in (r.description, r.quantity, r.unit_price, r.amount) if c]
        assert by_field[name] == " ".join(cells).split()


@pytest.mark.parametrize("doc", DOCS, ids=[d.doc_id for d in DOCS])
def test_boxes_are_on_the_page_and_do_not_overlap(doc: SyntheticDocument) -> None:
    words = render(doc).words
    width, height = PAGE_SIZE
    for w in words:
        assert 0 <= w.box[0] < w.box[2] <= width
        assert 0 <= w.box[1] < w.box[3] <= height
    for i, a in enumerate(words):
        for b in words[i + 1 :]:
            assert not _overlap(a, b), (a, b)


def test_rendering_is_deterministic() -> None:
    doc = sample_document(template("inv-c"), 5, 99)
    first, second = render(doc), render(doc)
    assert first.words == second.words
    digest = hashlib.sha256(first.image.tobytes()).hexdigest()
    assert digest == hashlib.sha256(second.image.tobytes()).hexdigest()


def test_degradation_keeps_size_and_word_list_and_changes_pixels() -> None:
    page = render(sample_document(template("inv-a"), 1, 3))
    for level in LEVELS[1:]:
        damaged = degrade(page, level, seed=4)
        assert damaged.image.size == page.image.size
        assert [w.text for w in damaged.words] == [w.text for w in page.words]
        assert damaged.image.tobytes() != page.image.tobytes()
    assert degrade(page, LEVELS[0], seed=4).words == page.words


def test_skewed_boxes_still_cover_the_ink() -> None:
    page = render(sample_document(template("bol-a"), 2, 3))
    skewed = degrade(page, Degradation(rotate_degrees=3.0), seed=0)
    gray = skewed.image.convert("L")
    for word in skewed.words[:40]:
        crop = gray.crop(word.box)
        lo, _ = crop.getextrema()
        assert isinstance(lo, int)
        assert lo < 128, word  # some dark ink lies inside the moved box
    # The box moves the way the page turns: counter-clockwise lifts the right side.
    right_word = max(page.words, key=lambda w: w.box[0])
    moved = next(
        w for w in skewed.words if w.text == right_word.text and w.field == right_word.field
    )
    assert moved.box[1] < right_word.box[1]
