"""Command-line runner: time one OCR engine on the synthetic pages and score word recall.

Usage::

    python -m bench.ocr_shortlist.run_bench --engine rapidocr --out <dir>

Purpose: produce one JSON result per engine with cold start, per-page latency, word recall on
clean and degraded pages, output capabilities and installed versions, plus the raw detections
for inspection.

Boundary: one engine per process, so each engine can live in its own virtual environment and
cold start is measured in a fresh interpreter. The runner does not compare engines; that is done
by reading the JSON files side by side. Failures are written into the result file and signalled
by a non-zero exit status rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from PIL import Image

from bench.ocr_shortlist.engines import (
    ENGINES,
    Detection,
    EngineUnavailableError,
    OcrEngine,
    create_engine,
)
from bench.ocr_shortlist.pages import degrade, make_page_set, normalize_tokens

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNAVAILABLE = 2


@dataclass(frozen=True)
class RecallResult:
    """Multiset word recall of OCR output against ground truth.

    Attributes:
        recall: ``matched / total``; 0.0 when there is no ground truth.
        matched: Ground-truth tokens found, each OCR token used at most once.
        total: Number of ground-truth tokens.
        missed: Ground-truth tokens not found, with multiplicity, sorted.
    """

    recall: float
    matched: int
    total: int
    missed: tuple[str, ...]


@dataclass(frozen=True)
class PageResult:
    """Timing and recall for one page variant.

    Attributes:
        page_id: Page identifier.
        variant: ``"clean"`` or ``"degraded"``.
        seconds: Wall-clock time of every timed run.
        median_seconds: Median of ``seconds``.
        recall: Word recall of the last run's output.
        detections: Number of detections returned by the last run.
        all_boxes: Whether every detection carried a box.
        all_confidences: Whether every detection carried a confidence.
    """

    page_id: str
    variant: str
    seconds: tuple[float, ...]
    median_seconds: float
    recall: RecallResult
    detections: int
    all_boxes: bool
    all_confidences: bool


def word_recall(truth: list[str], predicted: list[str]) -> RecallResult:
    """Score how many ground-truth tokens appear in the OCR tokens.

    Both inputs must already be normalised with :func:`normalize_tokens`. Matching is by token
    count, so a word printed twice must be read twice to count twice.

    Args:
        truth: Ground-truth tokens.
        predicted: OCR output tokens.

    Returns:
        The recall with the matched count and the missed tokens.
    """
    truth_counts = Counter(truth)
    remaining = truth_counts - Counter(predicted)
    total = sum(truth_counts.values())
    missed = sum(remaining.values())
    recall = (total - missed) / total if total else 0.0
    return RecallResult(recall, total - missed, total, tuple(sorted(remaining.elements())))


def directory_mb(path: Path) -> float | None:
    """Return the total size of files under ``path`` in MiB, or ``None`` if it does not exist.

    Args:
        path: Directory to measure.

    Returns:
        Size in MiB rounded to 0.1, or ``None`` when the path is absent.
    """
    if not path.exists():
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            if not file_path.is_symlink():
                total += file_path.stat().st_size
    return round(total / 2**20, 1)


def installed_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    """Look up installed distribution versions.

    Args:
        packages: Distribution names.

    Returns:
        Name to version, with ``None`` for a distribution that is not installed.
    """
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def detections_to_json(detections: list[Detection]) -> list[dict[str, Any]]:
    """Convert detections into JSON-ready dictionaries.

    Args:
        detections: Engine output.

    Returns:
        One ``{"text", "box", "confidence"}`` dictionary per detection.
    """
    return [
        {
            "text": d.text,
            "box": [list(point) for point in d.box] if d.box else None,
            "confidence": d.confidence,
        }
        for d in detections
    ]


def time_page(
    engine: OcrEngine,
    image: Image.Image,
    runs: int,
) -> tuple[list[float], list[Detection]]:
    """Run an engine repeatedly on one image.

    Args:
        engine: A constructed engine.
        image: Page image.
        runs: Number of timed runs, at least 1.

    Returns:
        Per-run wall-clock seconds and the detections from the final run.
    """
    seconds: list[float] = []
    detections: list[Detection] = []
    for _ in range(runs):
        start = time.perf_counter()
        detections = engine.run(image)
        seconds.append(time.perf_counter() - start)
    return seconds, detections


def environment_summary() -> dict[str, Any]:
    """Describe the interpreter and machine the run used.

    Returns:
        Python version, platform string, processor and logical CPU count, and whether the
        interpreter is running inside a virtual environment.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "in_venv": sys.prefix != sys.base_prefix,
        "prefix": sys.prefix,
    }


def mean_of(results: list[PageResult], variant: str, field: str) -> float | None:
    """Average one numeric field over the pages of a variant.

    Args:
        results: All page results.
        variant: ``"clean"`` or ``"degraded"``.
        field: ``"median_seconds"`` or ``"recall"``.

    Returns:
        The mean rounded to four places, or ``None`` when no page of that variant exists.
    """
    values = [
        r.median_seconds if field == "median_seconds" else r.recall.recall
        for r in results
        if r.variant == variant
    ]
    return round(statistics.fmean(values), 4) if values else None


def benchmark(engine_name: str, out_dir: Path, page_count: int, runs: int, seed: int) -> int:
    """Benchmark one engine and write ``<out_dir>/<engine_name>.json``.

    Cold start is the wall-clock time to import the engine package and construct the engine in
    this fresh interpreter (Pillow and NumPy are already imported). If weights were not yet
    cached, that time includes their download; run twice to get a cached figure. The first
    inference after construction is timed separately and excluded from the per-page medians.

    Args:
        engine_name: A key of :data:`~bench.ocr_shortlist.engines.ENGINES`.
        out_dir: Directory for the result file and raw detections.
        page_count: Number of synthetic pages; each is also run degraded.
        runs: Timed runs per page variant.
        seed: Page generator seed.

    Returns:
        A process exit status: 0 on success, 2 if the engine is unavailable, 1 on any failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw" / engine_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "engine": engine_name,
        "environment": environment_summary(),
        "settings": {"pages": page_count, "runs": runs, "seed": seed},
    }
    result_path = out_dir / f"{engine_name}.json"
    status = EXIT_OK
    try:
        start = time.perf_counter()
        engine = create_engine(engine_name)
        result["cold_start_seconds"] = round(time.perf_counter() - start, 3)
        result["granularity"] = engine.granularity
        result["engine_settings"] = engine.settings
        result["versions"] = installed_versions(engine.packages)

        pages = make_page_set(page_count, seed)
        start = time.perf_counter()
        engine.run(pages[0].image)
        result["first_inference_seconds"] = round(time.perf_counter() - start, 3)

        page_results: list[PageResult] = []
        for page in pages:
            truth = normalize_tokens(" ".join(page.words))
            for variant, image in (("clean", page.image), ("degraded", degrade(page.image))):
                seconds, detections = time_page(engine, image, runs)
                predicted = normalize_tokens("\n".join(d.text for d in detections))
                page_results.append(
                    PageResult(
                        page_id=page.page_id,
                        variant=variant,
                        seconds=tuple(round(s, 4) for s in seconds),
                        median_seconds=round(statistics.median(seconds), 4),
                        recall=word_recall(truth, predicted),
                        detections=len(detections),
                        all_boxes=all(d.box is not None for d in detections),
                        all_confidences=all(d.confidence is not None for d in detections),
                    )
                )
                raw_path = raw_dir / f"{page.page_id}-{variant}.json"
                raw_path.write_text(
                    json.dumps(detections_to_json(detections), indent=1), encoding="utf-8"
                )

        result["pages"] = [asdict(r) for r in page_results]
        result["summary"] = {
            "clean_seconds_per_page": mean_of(page_results, "clean", "median_seconds"),
            "degraded_seconds_per_page": mean_of(page_results, "degraded", "median_seconds"),
            "clean_recall": mean_of(page_results, "clean", "recall"),
            "degraded_recall": mean_of(page_results, "degraded", "recall"),
            "boxes": all(r.all_boxes for r in page_results),
            "confidences": all(r.all_confidences for r in page_results),
        }
        result["model_dirs_mb"] = {str(p): directory_mb(p) for p in engine.model_dirs()}
        result["status"] = "ok"
    except EngineUnavailableError as exc:
        result["status"] = "unavailable"
        result["error"] = str(exc)
        status = EXIT_UNAVAILABLE
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        status = EXIT_FAILED
    if result["environment"]["in_venv"]:
        result["venv_mb"] = directory_mb(Path(sys.prefix))
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("engine", "status", "error", "summary")}))
    if status == EXIT_FAILED:
        print(result["traceback"], file=sys.stderr)
    return status


def save_pages(out_dir: Path, page_count: int, seed: int) -> None:
    """Write the clean and degraded page images and their ground truth for inspection.

    Args:
        out_dir: Destination directory; created if missing.
        page_count: Number of pages.
        seed: Page generator seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in make_page_set(page_count, seed):
        page.image.save(out_dir / f"{page.page_id}-clean.png")
        degrade(page.image).save(out_dir / f"{page.page_id}-degraded.png")
        (out_dir / f"{page.page_id}-truth.txt").write_text(" ".join(page.words), encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name, or ``None`` for ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", required=True, choices=sorted(ENGINES))
    parser.add_argument("--out", required=True, type=Path, help="directory for result JSON")
    parser.add_argument("--pages", type=int, default=3, help="synthetic pages (default 3)")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per page (default 3)")
    parser.add_argument("--seed", type=int, default=20261025, help="page generator seed")
    parser.add_argument(
        "--save-pages", action="store_true", help="also write page images under <out>/pages"
    )
    args = parser.parse_args(argv)
    if args.pages < 1 or args.runs < 1:
        parser.error("--pages and --runs must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments without the program name, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    if args.save_pages:
        save_pages(args.out / "pages", args.pages, args.seed)
    return benchmark(args.engine, args.out, args.pages, args.runs, args.seed)


if __name__ == "__main__":
    sys.exit(main())
