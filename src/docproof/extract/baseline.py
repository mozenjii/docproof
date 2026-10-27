"""A rule-based baseline extractor: captions from the training templates, positions for the rest.

It finds field captions ("Invoice No.:", "Total:") on each text line and reads the text up to the
next caption; header parties (vendor and buyer, shipper and consignee) by their position under the
title; and item tables by the columns of their header row. Printed text is turned into typed values
with the same parsers the rest of the system uses, after reading the page's own locale evidence (a
day above 12 settles date order; an amount with both separators settles the decimal mark).

The caption vocabulary comes **only from templates in the training split**, plus the generic
captions every template falls back to. Held-out templates use captions it has never seen, which is
the honest test of a layout-bound method; its result there is the number to beat in B4.

Deliberate boundary: candidates only. Whether a candidate is grounded, valid or confident enough is
decided downstream, in :mod:`docproof.pipeline`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from docproof.errors import ContractError
from docproof.extract.layout import Line, lines
from docproof.grounding import key
from docproof.normalize import MINOR_UNITS, normalize_money
from docproof.ocr import OcrPage
from docproof.parsing import Ambiguous, Parsed, parse_amount, parse_date
from docproof.schema import (
    SCHEMAS,
    UNITS,
    DocumentType,
    FieldValue,
    LineItem,
    Money,
    Party,
    Quantity,
    ValueType,
)
from docproof.synth.templates import TEMPLATES, Split

TITLES: tuple[tuple[str, DocumentType], ...] = (
    ("billoflading", DocumentType.BILL_OF_LADING),
    ("rateconfirmation", DocumentType.RATE_CONFIRMATION),
    ("freightinvoice", DocumentType.FREIGHT_INVOICE),
    ("invoice", DocumentType.INVOICE),
)
"""Title keys checked in order; the first that appears near the top decides the document type."""

HEADER_PARTIES: dict[DocumentType, tuple[str, str]] = {
    DocumentType.INVOICE: ("vendor", "buyer"),
    DocumentType.FREIGHT_INVOICE: ("vendor", "buyer"),
    DocumentType.BILL_OF_LADING: ("shipper", "consignee"),
    DocumentType.RATE_CONFIRMATION: ("broker", "carrier"),
}
"""Which fields the first and second uncaptioned party blocks under the title are."""

_TABLE_FIELDS = {"description": "line_items", "accessorial": "accessorials"}
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-]\d{4}\b")
_CODE_SUFFIX = re.compile(r"^(.*?)\s*\b([A-Z]{3})$")
_CAPTION_LIKE = re.compile(r"(?:^|\s)[^\s:]+:(?=\s|$)")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A proposed value for one field and where its text came from.

    Attributes:
        field: Schema field name.
        value: The typed value.
        printed: The text as read, which grounding looks for on the page.
        ambiguous: The text has more than one reading; ``value`` is the first.
    """

    field: str
    value: FieldValue
    printed: tuple[str, ...]
    ambiguous: bool = False


@dataclass(slots=True)
class Extraction:
    """What the extractor found on a page.

    Attributes:
        doc_type: The recognised document type, or ``None`` when the title was not recognised.
        candidates: One candidate per field it could read, keyed by field name.
        unreadable: Fields whose caption was found but whose text could not be parsed.
    """

    doc_type: DocumentType | None
    candidates: dict[str, Candidate] = field(default_factory=dict[str, Candidate])
    unreadable: set[str] = field(default_factory=set[str])


def classify(page_lines: Sequence[Line]) -> DocumentType | None:
    """Recognise the document type from the title in the first three lines."""
    head = "".join(key(line.text) for line in page_lines[:3])
    return next((doc_type for title, doc_type in TITLES if title in head), None)


def training_captions(doc_type: DocumentType) -> dict[str, str]:
    """Caption keys the baseline knows for a document type, mapped to field names.

    Built from training-split templates and from the generic caption every template falls back to
    (the field name in title case). Validation and test templates contribute nothing.
    """
    known: dict[str, str] = {}
    for spec in SCHEMAS[doc_type]:
        known[spec.name.replace("_", " ").title()] = spec.name
    for tpl in TEMPLATES:
        if tpl.split is Split.TRAIN and tpl.doc_type is doc_type:
            for name, caption in tpl.captions.items():
                known[caption] = name
    return {key(caption): name for caption, name in known.items()}


@dataclass(frozen=True, slots=True)
class _Hit:
    start: int  # position in the line text where the caption starts
    end: int  # position just after the caption (and its colon, if any)
    field: str


def _key_index(text: str) -> tuple[str, list[int]]:
    """The line's key and, for each key character, its position in ``text``."""
    chars = [(i, c.lower()) for i, c in enumerate(text) if not c.isspace()]
    return "".join(c for _, c in chars), [i for i, _ in chars]


def _at_word_start(text: str, pos: int) -> bool:
    return pos == 0 or text[pos - 1].isspace()


def _at_word_end(text: str, pos: int) -> bool:
    return pos >= len(text) or text[pos].isspace() or text[pos] == ":"


def find_captions(text: str, captions: dict[str, str]) -> list[_Hit]:
    """Caption occurrences in a line, longest caption first, without overlaps.

    A caption matches with its colon anywhere, or without the colon when it stands as whole words
    (OCR often drops a colon on a bad scan). Each field is taken at most once per line.
    """
    text_key, index = _key_index(text)
    taken: list[tuple[int, int]] = []
    hits: list[_Hit] = []
    seen: set[str] = set()
    for caption_key in sorted(captions, key=len, reverse=True):
        name = captions[caption_key]
        for variant, needs_words in ((caption_key + ":", False), (caption_key, True)):
            position = text_key.find(variant)
            while position != -1:
                end = position + len(variant)
                start_char, end_char = index[position], index[end - 1] + 1
                free = all(end <= a or position >= b for a, b in taken)
                whole = _at_word_start(text, start_char) and _at_word_end(text, end_char)
                if free and name not in seen and (not needs_words or whole):
                    taken.append((position, end))
                    seen.add(name)
                    hits.append(_Hit(start_char, end_char, name))
                    break
                position = text_key.find(variant, position + 1)
            if name in seen:
                break
    return sorted(hits, key=lambda h: h.start)


def caption_values(text: str, captions: dict[str, str]) -> dict[str, str]:
    """Each captioned field's raw text on a line: from its caption to the next caption."""
    hits = find_captions(text, captions)
    values: dict[str, str] = {}
    if not hits:
        return values
    for current, following in zip(hits, [*hits[1:], None], strict=True):
        stop = following.start if following else len(text)
        raw = text[current.end : stop].strip().lstrip(":").strip()
        # A caption the baseline does not know ("HM:") still ends the value before it.
        unknown = _CAPTION_LIKE.search(raw)
        if unknown:
            raw = raw[: unknown.start()].strip()
        if raw:
            values[current.field] = raw
    return values


def is_caption_line(text: str, captions: dict[str, str]) -> bool:
    """Whether a line carries a caption, known or not (a word ending in a colon)."""
    return bool(_CAPTION_LIKE.search(text)) or bool(find_captions(text, captions))


@dataclass(frozen=True, slots=True)
class Locale:
    """Locale evidence read from the whole page."""

    day_first: bool | None
    decimal_comma: bool | None
    currency: str | None


def read_locale(page_lines: Sequence[Line]) -> Locale:
    """Infer date order, decimal mark and currency from evidence anywhere on the page.

    A numeric date with a first number above 12 proves day-first, a second number above 12 proves
    month-first; an amount containing both separators shows which one is decimal; a currency code
    printed after an amount or next to a "Currency" caption names the currency. Contradicting
    evidence leaves the answer unknown rather than picking one.
    """
    day_first: set[bool] = set()
    decimal_comma: set[bool] = set()
    currencies: set[str] = set()
    for line in page_lines:
        text = unicodedata.normalize("NFKC", line.text)
        for a, b in _NUMERIC_DATE.findall(text):
            if int(a) > 12:
                day_first.add(True)
            elif int(b) > 12:
                day_first.add(False)
        for amount in re.findall(r"\d[\d.,]*[.,]\d+", text):
            if "," in amount and "." in amount:
                decimal_comma.add(amount.rfind(",") > amount.rfind("."))
        for code in re.findall(r"\b([A-Z]{3})\b", text):
            if code in MINOR_UNITS:
                currencies.add(code)
    return Locale(
        day_first=day_first.pop() if len(day_first) == 1 else None,
        decimal_comma=decimal_comma.pop() if len(decimal_comma) == 1 else None,
        currency=currencies.pop() if len(currencies) == 1 else None,
    )


def _money(text: str, locale: Locale) -> tuple[Money, bool] | None:
    match = _CODE_SUFFIX.match(text.strip())
    number, currency = (match[1], match[2]) if match else (text, locale.currency)
    if currency is None or currency not in MINOR_UNITS:
        return None
    parsed = parse_amount(number, locale.decimal_comma)
    readings: tuple[str, ...] = ()
    if isinstance(parsed, Parsed):
        readings = (parsed.value,)
    elif isinstance(parsed, Ambiguous):
        readings = parsed.readings
    # The currency's precision is evidence too: "1,234 USD" cannot be 1.234 dollars.
    valid: list[Money] = []
    for reading in readings:
        try:
            valid.append(normalize_money(reading, currency))
        except ContractError:
            continue
    if not valid:
        return None
    return valid[0], len(valid) > 1


def _quantity(text: str, locale: Locale) -> tuple[Quantity, bool] | None:
    parts = text.strip().rsplit(" ", 1)
    if len(parts) != 2 or parts[1].upper() not in UNITS:
        return None
    parsed = parse_amount(parts[0], locale.decimal_comma)
    if isinstance(parsed, Parsed):
        return Quantity(Decimal(parsed.value), parts[1].upper()), False
    if isinstance(parsed, Ambiguous):
        return Quantity(Decimal(parsed.readings[0]), parts[1].upper()), True
    return None


def typed_value(value_type: ValueType, text: str, locale: Locale) -> tuple[FieldValue, bool] | None:
    """Turn printed text into a typed value, returning it and whether the text was ambiguous."""
    match value_type:
        case ValueType.TEXT | ValueType.IDENTIFIER | ValueType.CODE:
            return text.strip(), False
        case ValueType.DATE:
            parsed = parse_date(text, locale.day_first)
            if isinstance(parsed, Parsed):
                return parsed.value, False
            if isinstance(parsed, Ambiguous):
                return parsed.readings[0], True
            return None
        case ValueType.MONEY:
            return _money(text, locale)
        case ValueType.QUANTITY:
            return _quantity(text, locale)
        case ValueType.PARTY:
            return Party(text.strip(), None), False
        case ValueType.LINE_ITEMS:
            return None


def _is_address(text: str) -> bool:
    return bool(text) and text.lstrip()[:1].isdigit() and "," in text


def header_parties(
    page: OcrPage, page_lines: Sequence[Line], first_captioned: int
) -> list[tuple[str, str | None]]:
    """The uncaptioned party blocks between the title and the first captioned line.

    Text in the left half of the page is the first block, text in the right half the second. When
    everything is on the left, blocks are split where an address line ends one.
    """
    left: list[str] = []
    right: list[str] = []
    for line in page_lines[1:first_captioned]:
        half = page.width / 2
        left_text = " ".join(
            page.tokens[i].text.strip() for i in line.tokens if page.tokens[i].rect[0] < half
        )
        right_text = " ".join(
            page.tokens[i].text.strip() for i in line.tokens if page.tokens[i].rect[0] >= half
        )
        if left_text:
            left.append(left_text)
        if right_text:
            right.append(right_text)
    columns = [left, right] if right else [left]
    blocks: list[tuple[str, str | None]] = []
    for column in columns:
        position = 0
        while position < len(column):
            name = column[position]
            address = column[position + 1] if position + 1 < len(column) else None
            if address is not None and _is_address(address) and not _is_address(name):
                blocks.append((name, address))
                position += 2
            else:
                blocks.append((name, None))
                position += 1
    return blocks


def _cell_text(page: OcrPage, tokens: Sequence[int], low: float, high: float) -> str | None:
    words = [page.tokens[i].text.strip() for i in tokens if low <= page.tokens[i].rect[0] < high]
    return " ".join(words) if words else None


def read_table(
    page: OcrPage,
    page_lines: Sequence[Line],
    header_at: int,
    captions: dict[str, str],
    locale: Locale,
) -> tuple[tuple[LineItem, ...], list[str], list[int]]:
    """Rows under a table header, cut into columns at the header's Qty, Unit price, Amount marks.

    Returns the items, the printed texts of their cells, and the indexes of the lines read as rows.
    Reading stops at the first line that carries a colon caption, another table header, or a row
    whose amount cannot be read. Rows are not checked for colon-less captions: "Inside delivery" is
    a row, not the Delivery caption.
    """
    header = page_lines[header_at]
    marks: dict[str, float] = {}
    for i in header.tokens:
        k = key(page.tokens[i].text)
        # "Unit price" is one token or two depending on the engine, so its mark is "unit".
        for name in ("qty", "unit", "amount"):
            if k.startswith(name) and name not in marks:
                marks[name] = page.tokens[i].rect[0]
    if len(marks) < 3:
        return (), [], []
    margin = 0.02 * page.width
    cuts = (marks["qty"] - margin, marks["unit"] - margin, marks["amount"] - margin)
    items: list[LineItem] = []
    printed: list[str] = []
    rows: list[int] = []
    for number in range(header_at + 1, len(page_lines)):
        line = page_lines[number]
        if _CAPTION_LIKE.search(line.text) or any(
            key(page.tokens[i].text).startswith(("qty", "amount")) for i in line.tokens
        ):
            break
        description = _cell_text(page, line.tokens, -1e9, cuts[0])
        quantity_text = _cell_text(page, line.tokens, cuts[0], cuts[1])
        price_text = _cell_text(page, line.tokens, cuts[1], cuts[2])
        amount_text = _cell_text(page, line.tokens, cuts[2], 1e9)
        amount = _money(amount_text, locale) if amount_text else None
        if description is None or amount is None:
            break
        price = _money(price_text, locale) if price_text else None
        quantity = _quantity(quantity_text, locale) if quantity_text else None
        items.append(
            LineItem(
                description,
                amount[0],
                quantity[0] if quantity else None,
                price[0] if price else None,
            )
        )
        printed.extend(t for t in (description, quantity_text, price_text, amount_text) if t)
        rows.append(number)
    return tuple(items), printed, rows


def extract(page: OcrPage) -> Extraction:
    """Read every field the baseline can find on a page.

    Args:
        page: The OCR page.

    Returns:
        The document type and one candidate per field read. A field with no candidate was not found;
        a field in ``unreadable`` was found but its text could not be parsed.
    """
    page_lines = lines(page)
    doc_type = classify(page_lines)
    result = Extraction(doc_type)
    if doc_type is None:
        return result
    specs = {spec.name: spec for spec in SCHEMAS[doc_type]}
    captions = training_captions(doc_type)
    locale = read_locale(page_lines)
    # Tables first, so their rows are never mistaken for captions ("Inside delivery").
    table_rows: set[int] = set()
    for number, line in enumerate(page_lines):
        title = key(line.text)
        for prefix, name in _TABLE_FIELDS.items():
            if title.startswith(prefix) and name in specs and name not in result.candidates:
                items, printed, rows = read_table(page, page_lines, number, captions, locale)
                table_rows.update(rows)
                if items:
                    result.candidates[name] = Candidate(name, items, tuple(printed))
    raw: dict[str, tuple[str, ...]] = {}
    # The header ends at the first line with any caption, known or not.
    first_captioned = next(
        (n for n, line in enumerate(page_lines) if n and is_caption_line(line.text, captions)),
        len(page_lines),
    )
    for number, line in enumerate(page_lines):
        # The title is never a caption line: "RATE CONFIRMATION" would read as the "Rate" field.
        if number == 0 or number in table_rows:
            continue
        found = caption_values(line.text, captions)
        for name, text in found.items():
            raw.setdefault(name, (text,))
            if specs[name].value_type is ValueType.PARTY and number + 1 < len(page_lines):
                following = page_lines[number + 1].text
                if _is_address(following) and not caption_values(following, captions):
                    raw[name] = (text, following)
    primary, secondary = HEADER_PARTIES[doc_type]
    for name, block in zip(
        (primary, secondary), header_parties(page, page_lines, first_captioned), strict=False
    ):
        raw.setdefault(name, tuple(x for x in block if x))
    for name, texts in raw.items():
        spec = specs[name]
        if spec.value_type is ValueType.PARTY:
            first, rest = texts[0], texts[1:]
            result.candidates[name] = Candidate(
                name, Party(first, rest[0] if rest else None), texts
            )
            continue
        typed = typed_value(spec.value_type, texts[0], locale)
        if typed is None:
            result.unreadable.add(name)
        else:
            result.candidates[name] = Candidate(name, typed[0], texts, typed[1])
    return result
