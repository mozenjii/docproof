r"""Score the end-to-end path on a synthetic evaluation set and print the results table.

Usage::

    python -m docproof.evaluate --dataset data/synthetic/ocr-eval --ocr paddleocr-mobile \
        --tolerance 0 --thresholds 0.5 0.9 0.95 --out runs/b3a

``--ocr perfect`` builds pages from the generator's own word boxes instead of recorded OCR, which
separates extraction errors from OCR errors. It covers level 0 only: on a skewed page the generator
records each word's axis-aligned bounding box, which loses the text angle the deskewing relies on,
so "perfect" pages at levels 1-3 would test a geometry no engine produces.

Every number comes with a 95% bootstrap interval over documents. A threshold sweep is reported
rather than one threshold, because the accept threshold (OPEN-8) is not decided and confidence is
not calibrated yet.

Deliberate boundary: file I/O and reporting only; extraction, routing and scoring are imported.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docproof.extract.baseline import extract
from docproof.labels import decode_labels
from docproof.ocr import OcrPage, load_ocr_page, page_from_words
from docproof.pipeline import route_extraction
from docproof.scoring import Counts, Interval, Metrics, ScoredDocument, bootstrap, score_document

RESAMPLES = 1000


@dataclass(frozen=True, slots=True)
class PageRef:
    """One page of the evaluation set."""

    doc_id: str
    split: str
    level: int
    folder: Path


def load_pages(dataset: Path) -> list[PageRef]:
    """Every (document, level) page listed in the dataset's manifest."""
    refs: list[PageRef] = []
    for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        for level in entry["levels"]:
            refs.append(PageRef(entry["doc_id"], entry["split"], level, dataset / entry["split"]))
    return refs


def read_page(ref: PageRef, ocr: str) -> OcrPage:
    """Load a page's recorded OCR, or build a perfect page from its word boxes.

    Raises:
        FileNotFoundError: If the recorded OCR for this page does not exist.
    """
    stem = f"{ref.doc_id}.L{ref.level}"
    if ocr == "perfect":
        words = json.loads((ref.folder / f"{stem}.words.json").read_text(encoding="utf-8"))
        return page_from_words(1654, 2339, [(w["text"], tuple(w["box"])) for w in words])
    return load_ocr_page(
        json.loads((ref.folder / f"{stem}.ocr.{ocr}.json").read_text(encoding="utf-8"))
    )


def _fmt(interval: Interval) -> str:
    if interval.estimate is None:
        return "n/a"
    if interval.low is None or interval.high is None:
        return f"{interval.estimate:.3f}"
    return f"{interval.estimate:.3f} [{interval.low:.3f}, {interval.high:.3f}]"


def summarise(per_doc: list[Counts]) -> dict[str, dict[str, float | None]]:
    """The four metrics with bootstrap intervals over the given documents."""
    pickers: dict[str, Callable[[Metrics], float | None]] = {
        "field_f1": lambda m: m.field_f1,
        "document_complete_rate": lambda m: m.document_complete_rate,
        "straight_through_rate": lambda m: m.straight_through_rate,
        "accepted_precision": lambda m: m.accepted_precision,
    }
    out: dict[str, dict[str, float | None]] = {}
    for name, pick in pickers.items():
        interval = bootstrap(per_doc, pick, resamples=RESAMPLES, seed=0)
        out[name] = {"estimate": interval.estimate, "low": interval.low, "high": interval.high}
    return out


def evaluate(
    dataset: Path, ocr: str, tolerance: int, thresholds: Sequence[float]
) -> dict[str, Any]:
    """Run every page through the pipeline at every threshold and score it.

    Returns:
        ``{"settings": ..., "groups": {"<split> L<level> @<threshold>": summary}}``.
    """
    groups: dict[tuple[str, int, float], list[Counts]] = defaultdict(list)
    for ref in load_pages(dataset):
        if ocr == "perfect" and ref.level != 0:
            continue
        page = read_page(ref, ocr)
        raw_labels = json.loads(
            (ref.folder / f"{ref.doc_id}.labels.json").read_text(encoding="utf-8")
        )
        doc_type, gold = decode_labels(raw_labels)
        extraction = extract(page)
        for threshold in thresholds:
            result = route_extraction(
                extraction, page, tolerance_minor=tolerance, threshold=threshold
            )
            predicted = result.predictions() if result.doc_type is doc_type else {}
            counts = score_document(ScoredDocument(ref.doc_id, doc_type, gold, predicted))
            groups[(ref.split, ref.level, threshold)].append(counts)
    return {
        "settings": {
            "dataset": str(dataset),
            "ocr": ocr,
            "tolerance_minor": tolerance,
            "thresholds": list(thresholds),
            "resamples": RESAMPLES,
        },
        "groups": {
            f"{split} L{level} @{threshold}": {"documents": len(counts), **summarise(counts)}
            for (split, level, threshold), counts in sorted(groups.items())
        },
    }


def table(results: dict[str, Any]) -> str:
    """Render results as a markdown table, one row per split, level and threshold."""
    rows = [
        "| group | docs | field F1 | doc complete | straight-through | accepted precision |",
        "|---|---|---|---|---|---|",
    ]
    for name, group in results["groups"].items():

        def cell(metric: str, group: dict[str, Any] = group) -> str:
            m = group[metric]
            return _fmt(Interval(m["estimate"], m["low"], m["high"], 0.95))

        rows.append(
            f"| {name} | {group['documents']} | {cell('field_f1')} | "
            f"{cell('document_complete_rate')} | {cell('straight_through_rate')} | "
            f"{cell('accepted_precision')} |"
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Score the pipeline on a synthetic evaluation set."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ocr", required=True, help="recorded engine name, or 'perfect'")
    parser.add_argument("--tolerance", type=int, required=True, help="minor units for V1-V3")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=Path, help="folder for results.json and results.md")
    args = parser.parse_args(argv)
    results = evaluate(args.dataset, args.ocr, args.tolerance, args.thresholds)
    rendered = table(results)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        (args.out / "results.md").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
