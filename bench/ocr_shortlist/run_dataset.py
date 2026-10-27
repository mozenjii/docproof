"""Command-line runner: score one OCR engine on the degraded synthetic evaluation set.

Usage::

    python -m bench.ocr_shortlist.run_dataset --engine rapidocr --dataset <set> --out <dir>
        [--levels 0 1 2 3] [--limit N] [--rescore]

Purpose: the first benchmark used clean renders and every engine read almost all of the text, so
it could not separate them. This runner reads the evaluation set under ``data/synthetic/ocr-eval``
instead: each document is rendered at four degradation levels (0 clean to 3 a bad fax), and every
page ships the exact words printed on it with their boxes and the schema field each belongs to.
For every page it records word recall, field-word recall, the located field-word rate (right text
in the right place, which grounding needs) and seconds per page, then summarises each level with
a bootstrap interval over documents.

Boundary: one engine per process, as in ``run_bench``, so each engine keeps its own virtual
environment; nothing here imports the ``docproof`` package. The runner does not generate or
degrade pages and does not compare engines. Raw detections are saved per page, and ``--rescore``
recomputes every metric from them without running the engine again. A page that fails is recorded
with its error, excluded from the means, counted in the summary, and makes the exit status
non-zero.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from bench.ocr_shortlist.engines import (
    ENGINES,
    Box,
    Detection,
    EngineUnavailableError,
    create_engine,
)
from bench.ocr_shortlist.pages import normalize_tokens
from bench.ocr_shortlist.run_bench import (
    detections_to_json,
    environment_summary,
    installed_versions,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNAVAILABLE = 2

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
METRICS = ("word_recall", "field_word_recall", "located_field_word_rate", "seconds")
"""Per-page values summarised for each level."""

type Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class TruthWord:
    """One printed word from a page's ``words.json``.

    Attributes:
        token: The word after :func:`normalize_tokens`; empty for punctuation-only words.
        box: ``(left, top, right, bottom)`` in page pixels.
        field: Schema field the word belongs to, or ``None`` for captions and titles.
    """

    token: str
    box: Rect
    field: str | None


@dataclass(frozen=True)
class PageSpec:
    """One page of the evaluation set: a document at one degradation level.

    Attributes:
        doc_id: Document identifier from the manifest.
        template: Layout template the document was drawn from.
        doc_type: Document type, such as ``invoice``.
        split: Dataset split directory.
        level: Degradation level, 0 (clean) to 3 (worst).
        image_path: The page image.
        words_path: The page's ground-truth words.
    """

    doc_id: str
    template: str
    doc_type: str
    split: str
    level: int
    image_path: Path
    words_path: Path

    @property
    def page_key(self) -> str:
        """File stem used for this page's raw detections."""
        return f"{self.doc_id}.L{self.level}"


@dataclass(frozen=True)
class FieldCounts:
    """Per-field counts on one page.

    Attributes:
        total: Field words with a non-empty token.
        read: Of those, words matched by multiset recall within the field.
        located: Of those, words read at the right place (see :func:`is_located`).
    """

    total: int
    read: int
    located: int


@dataclass(frozen=True)
class PageScore:
    """Scores for one page.

    Attributes:
        word_recall: Multiset recall over every printed word.
        field_word_recall: Multiset recall over words with a field.
        located_field_word_rate: Share of field words found inside a detection that covers them.
        words: Number of printed words with a non-empty token.
        field_words: Number of those with a field.
        missed_field_tokens: Field tokens not read anywhere, with multiplicity, sorted.
        unlocated_field_tokens: Field tokens not found at their own position, sorted.
        fields: Counts per field name.
    """

    word_recall: float | None
    field_word_recall: float | None
    located_field_word_rate: float | None
    words: int
    field_words: int
    missed_field_tokens: tuple[str, ...]
    unlocated_field_tokens: tuple[str, ...]
    fields: dict[str, FieldCounts] = field(default_factory=dict)


def load_manifest(dataset: Path, levels: list[int], limit: int | None) -> list[PageSpec]:
    """Read the manifest and list the pages to run, document by document.

    Args:
        dataset: Directory holding ``manifest.jsonl`` and the split directories.
        levels: Degradation levels to include.
        limit: Keep only the first ``limit`` documents, or all when ``None``.

    Returns:
        One spec per document and requested level, in manifest order.

    Raises:
        FileNotFoundError: If the manifest or any listed page or words file is missing.
        ValueError: If a requested level is not listed for a document.
    """
    lines = (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    documents = [json.loads(line) for line in lines if line.strip()]
    if limit is not None:
        documents = documents[:limit]
    specs: list[PageSpec] = []
    for doc in documents:
        for level in levels:
            if level not in doc["levels"]:
                raise ValueError(f"{doc['doc_id']} has no level {level}")
            stem = dataset / doc["split"] / f"{doc['doc_id']}.L{level}"
            spec = PageSpec(
                doc_id=doc["doc_id"],
                template=doc["template"],
                doc_type=doc["doc_type"],
                split=doc["split"],
                level=level,
                image_path=Path(f"{stem}.png"),
                words_path=Path(f"{stem}.words.json"),
            )
            for path in (spec.image_path, spec.words_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            specs.append(spec)
    return specs


def load_truth(words_path: Path) -> list[TruthWord]:
    """Load and normalise a page's ground-truth words.

    Args:
        words_path: A ``*.words.json`` file.

    Returns:
        Every printed word. A word is one whitespace-free string, so it normalises to at most one
        token; words that normalise to nothing keep an empty token and are ignored by scoring.

    Raises:
        ValueError: If a word normalises to more than one token.
    """
    words: list[TruthWord] = []
    for entry in json.loads(words_path.read_text(encoding="utf-8")):
        tokens = normalize_tokens(entry["text"])
        if len(tokens) > 1:
            raise ValueError(f"{words_path}: word {entry['text']!r} is more than one token")
        left, top, right, bottom = (float(v) for v in entry["box"])
        token = tokens[0] if tokens else ""
        words.append(TruthWord(token, (left, top, right, bottom), entry["field"]))
    return words


def multiset_recall(truth: list[str], predicted: Counter[str]) -> tuple[float | None, Counter[str]]:
    """Score how many ground-truth tokens appear among the predicted tokens.

    Args:
        truth: Normalised ground-truth tokens.
        predicted: Normalised predicted token counts.

    Returns:
        ``matched / total`` (``None`` when there is no truth) and the unmatched truth counts.
    """
    truth_counts = Counter(truth)
    missed = truth_counts - predicted
    total = truth_counts.total()
    return ((total - missed.total()) / total if total else None), missed


def bounding_rect(detection: Detection) -> Rect | None:
    """Return the axis-aligned bounding rectangle of a detection's box, if it has one."""
    if detection.box is None:
        return None
    xs = [x for x, _ in detection.box]
    ys = [y for _, y in detection.box]
    return (min(xs), min(ys), max(xs), max(ys))


def is_located(word: TruthWord, detections: list[tuple[set[str], Rect | None]]) -> bool:
    """Decide whether a field word was read at its own position.

    A word is located when some detection contains its token among the detection's tokens and
    the centre of the word's box lies inside that detection's axis-aligned bounding rectangle.

    Args:
        word: A field word with a non-empty token.
        detections: Each detection's token set and bounding rectangle.

    Returns:
        Whether such a detection exists.
    """
    left, top, right, bottom = word.box
    cx, cy = (left + right) / 2, (top + bottom) / 2
    for tokens, rect in detections:
        if rect is None or word.token not in tokens:
            continue
        x0, y0, x1, y1 = rect
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return True
    return False


def score_page(truth: list[TruthWord], detections: list[Detection]) -> PageScore:
    """Compute every per-page metric.

    Args:
        truth: The page's printed words.
        detections: Engine output for the page.

    Returns:
        The page's scores, with per-field counts and the field tokens lost.
    """
    predicted = Counter(t for d in detections for t in normalize_tokens(d.text))
    indexed = [(set(normalize_tokens(d.text)), bounding_rect(d)) for d in detections]
    scored = [w for w in truth if w.token]
    field_words = [w for w in scored if w.field is not None]

    word_recall, _ = multiset_recall([w.token for w in scored], predicted)
    field_recall, missed = multiset_recall([w.token for w in field_words], predicted)
    located_flags = [is_located(w, indexed) for w in field_words]
    located_rate = sum(located_flags) / len(field_words) if field_words else None

    fields: dict[str, FieldCounts] = {}
    for name in sorted({w.field for w in field_words if w.field is not None}):
        members = [i for i, w in enumerate(field_words) if w.field == name]
        _, field_missed = multiset_recall([field_words[i].token for i in members], predicted)
        fields[name] = FieldCounts(
            total=len(members),
            read=len(members) - field_missed.total(),
            located=sum(located_flags[i] for i in members),
        )
    unlocated = [w.token for w, ok in zip(field_words, located_flags, strict=True) if not ok]
    return PageScore(
        word_recall=word_recall,
        field_word_recall=field_recall,
        located_field_word_rate=located_rate,
        words=len(scored),
        field_words=len(field_words),
        missed_field_tokens=tuple(sorted(missed.elements())),
        unlocated_field_tokens=tuple(sorted(unlocated)),
        fields=fields,
    )


def bootstrap_interval(values: list[float]) -> tuple[float, float] | None:
    """Return a 95% percentile bootstrap interval for the mean.

    Uses :data:`BOOTSTRAP_RESAMPLES` resamples drawn with a generator seeded by
    :data:`BOOTSTRAP_SEED`, so the interval is reproducible.

    Args:
        values: One value per document.

    Returns:
        ``(low, high)``, or ``None`` when there are no values.
    """
    if not values:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(BOOTSTRAP_RESAMPLES))
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise page records per level.

    Each document contributes one page per level, so the bootstrap over pages of a level is a
    bootstrap over documents.

    Args:
        records: Page records as written to the result file.

    Returns:
        Per level: page and failure counts, and for each metric the mean and 95% interval over
        the pages that succeeded; plus per-field totals of read and located words.
    """
    summary: dict[str, Any] = {}
    for level in sorted({r["level"] for r in records}):
        at_level = [r for r in records if r["level"] == level]
        ok = [r for r in at_level if r["status"] == "ok"]
        entry: dict[str, Any] = {"pages": len(at_level), "failed": len(at_level) - len(ok)}
        for metric in METRICS:
            values = [r[metric] for r in ok if r[metric] is not None]
            interval = bootstrap_interval(values)
            entry[metric] = {
                "mean": round(statistics.fmean(values), 4) if values else None,
                "ci95": [round(v, 4) for v in interval] if interval else None,
                "n": len(values),
            }
        totals: dict[str, Counter[str]] = {}
        for record in ok:
            for name, counts in record["fields"].items():
                totals.setdefault(name, Counter()).update(counts)
        entry["fields"] = {name: dict(counts) for name, counts in sorted(totals.items())}
        summary[str(level)] = entry
    return summary


def detections_from_json(entries: list[dict[str, Any]]) -> list[Detection]:
    """Rebuild detections from the raw JSON written by ``run_bench.detections_to_json``.

    Raises:
        ValueError: If a saved box does not have exactly four points.
    """
    detections: list[Detection] = []
    for entry in entries:
        box: Box | None = None
        if entry["box"]:
            points = [(float(x), float(y)) for x, y in entry["box"]]
            if len(points) != 4:
                raise ValueError(f"expected a 4-point box, got {len(points)} points")
            box = (points[0], points[1], points[2], points[3])
        detections.append(Detection(entry["text"], box, entry["confidence"]))
    return detections


def page_record(spec: PageSpec) -> dict[str, Any]:
    """Return the identifying part of a page record."""
    return {
        "doc_id": spec.doc_id,
        "template": spec.template,
        "doc_type": spec.doc_type,
        "split": spec.split,
        "level": spec.level,
    }


def scored_record(
    spec: PageSpec, detections: list[Detection], seconds: float | None
) -> dict[str, Any]:
    """Score a page and build its full record.

    Args:
        spec: The page.
        detections: Engine output for the page.
        seconds: Timed inference seconds, or ``None`` if unknown.

    Returns:
        The page record with status ``ok``.
    """
    score = score_page(load_truth(spec.words_path), detections)
    record = page_record(spec)
    record.update(
        status="ok",
        word_recall=score.word_recall,
        field_word_recall=score.field_word_recall,
        located_field_word_rate=score.located_field_word_rate,
        seconds=None if seconds is None else round(seconds, 4),
        detections=len(detections),
        words=score.words,
        field_words=score.field_words,
        missed_field_tokens=list(score.missed_field_tokens),
        unlocated_field_tokens=list(score.unlocated_field_tokens),
        fields={name: asdict(counts) for name, counts in score.fields.items()},
    )
    return record


def failed_record(spec: PageSpec, exc: Exception) -> dict[str, Any]:
    """Build the record of a page that raised, keeping the error and traceback."""
    record = page_record(spec)
    record.update(
        status="failed",
        error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(),
        **dict.fromkeys(METRICS),
    )
    return record


def run_pages(engine_name: str, specs: list[PageSpec], raw_dir: Path) -> dict[str, Any]:
    """Construct the engine, warm it up and run every page once.

    Args:
        engine_name: A key of :data:`~bench.ocr_shortlist.engines.ENGINES`.
        specs: Pages to run.
        raw_dir: Directory for one raw detection file per page.

    Returns:
        Construction and warm-up seconds, engine settings and versions, and the page records.
        A page that raises is recorded as failed and the run continues.
    """
    start = time.perf_counter()
    engine = create_engine(engine_name)
    construct_seconds = time.perf_counter() - start
    with Image.open(specs[0].image_path) as first:
        warm_image = first.convert("RGB")
    start = time.perf_counter()
    engine.run(warm_image)
    warmup_seconds = time.perf_counter() - start

    records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        try:
            with Image.open(spec.image_path) as opened:
                image = opened.convert("RGB")
            start = time.perf_counter()
            detections = engine.run(image)
            seconds = time.perf_counter() - start
            raw = {"seconds": round(seconds, 4), "detections": detections_to_json(detections)}
            (raw_dir / f"{spec.page_key}.json").write_text(json.dumps(raw), encoding="utf-8")
            record = scored_record(spec, detections, seconds)
        except Exception as exc:
            record = failed_record(spec, exc)
            print(f"FAILED {spec.page_key}: {record['error']}", file=sys.stderr)
        records.append(record)
        print(
            f"[{index}/{len(specs)}] {spec.page_key} {record['status']} "
            f"recall={record['word_recall']} located={record['located_field_word_rate']} "
            f"s={record['seconds']}",
            flush=True,
        )
    return {
        "construct_seconds": round(construct_seconds, 3),
        "warmup_seconds": round(warmup_seconds, 3),
        "granularity": engine.granularity,
        "engine_settings": engine.settings,
        "versions": installed_versions(engine.packages),
        "pages": records,
    }


def rescore_pages(specs: list[PageSpec], raw_dir: Path) -> dict[str, Any]:
    """Score every page from its saved raw detections without running the engine.

    Args:
        specs: Pages to score.
        raw_dir: Directory written by an earlier run.

    Returns:
        The page records. A page without a raw file, or whose raw file cannot be read, is
        recorded as failed.
    """
    records: list[dict[str, Any]] = []
    for spec in specs:
        try:
            raw = json.loads((raw_dir / f"{spec.page_key}.json").read_text(encoding="utf-8"))
            detections = detections_from_json(raw["detections"])
            records.append(scored_record(spec, detections, raw["seconds"]))
        except Exception as exc:
            records.append(failed_record(spec, exc))
            print(f"FAILED {spec.page_key}: {records[-1]['error']}", file=sys.stderr)
    return {"pages": records}


def evaluate(args: argparse.Namespace) -> int:
    """Run or rescore one engine and write ``<out>/<engine>.json``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        0 when every page succeeded, 2 when the engine is unavailable, 1 on any failure.
    """
    raw_dir = args.out / "raw" / args.engine
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "engine": args.engine,
        "environment": environment_summary(),
        "settings": {
            "dataset": str(args.dataset),
            "levels": args.levels,
            "limit": args.limit,
            "rescore": args.rescore,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        },
    }
    status = EXIT_OK
    try:
        specs = load_manifest(args.dataset, args.levels, args.limit)
        if not specs:
            raise ValueError("no pages selected")
        run = (
            rescore_pages(specs, raw_dir)
            if args.rescore
            else run_pages(args.engine, specs, raw_dir)
        )
        result.update(run)
        result["summary"] = summarise(run["pages"])
        failed = sum(1 for r in run["pages"] if r["status"] != "ok")
        result["status"] = "ok" if failed == 0 else f"{failed} pages failed"
        if failed:
            status = EXIT_FAILED
    except EngineUnavailableError as exc:
        result["status"] = "unavailable"
        result["error"] = str(exc)
        status = EXIT_UNAVAILABLE
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr)
        status = EXIT_FAILED
    out_path = args.out / f"{args.engine}.json"
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    brief = {k: result.get(k) for k in ("engine", "status", "error")}
    print(json.dumps(brief))
    return status


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name, or ``None`` for ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", required=True, choices=sorted(ENGINES))
    parser.add_argument("--dataset", required=True, type=Path, help="evaluation set directory")
    parser.add_argument("--out", required=True, type=Path, help="directory for result JSON")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--limit", type=int, default=None, help="first N documents only")
    parser.add_argument(
        "--rescore", action="store_true", help="score saved raw detections; do not run the engine"
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments without the program name, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
