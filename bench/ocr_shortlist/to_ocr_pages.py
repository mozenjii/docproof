r"""Turn the benchmark's saved detections into OCR page files the pipeline can read.

Usage::

    python -m bench.ocr_shortlist.to_ocr_pages --engine paddleocr-mobile \
        --raw <eval>/raw/paddleocr-mobile --dataset data/synthetic/ocr-eval

For every ``<doc_id>.L<n>.json`` in the raw folder it writes ``<doc_id>.L<n>.ocr.<engine>.json``
next to the page image, in the format ``docproof.ocr.load_ocr_page`` reads. The engine therefore
runs once and every later pipeline run replays its output.

Boundary: a format conversion only. It reads the page size from the image header and copies text,
boxes and confidences unchanged; it does not filter, merge or correct anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def convert(engine: str, raw_dir: Path, dataset: Path) -> int:
    """Write one OCR page file per saved detection file.

    Args:
        engine: Engine name, used in the output file name.
        raw_dir: Folder of ``<doc_id>.L<n>.json`` detection files from ``run_dataset``.
        dataset: The evaluation set holding ``<split>/<doc_id>.L<n>.png``.

    Returns:
        Number of pages written.

    Raises:
        FileNotFoundError: If a detection file has no matching page image.
    """
    images = {p.stem: p for p in dataset.glob("*/*.png")}
    count = 0
    for raw_path in sorted(raw_dir.glob("*.json")):
        page_key = raw_path.stem
        image_path = images.get(page_key)
        if image_path is None:
            raise FileNotFoundError(f"no page image for {page_key}")
        with Image.open(image_path) as image:
            width, height = image.size
        detections = json.loads(raw_path.read_text(encoding="utf-8"))["detections"]
        tokens = [
            {"text": d["text"], "box": d["box"], "confidence": d["confidence"]}
            for d in detections
            if d["box"] is not None and d["confidence"] is not None and d["text"].strip()
        ]
        out = image_path.with_name(f"{page_key}.ocr.{engine}.json")
        out.write_text(
            json.dumps({"width": width, "height": height, "tokens": tokens}), encoding="utf-8"
        )
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Convert saved detections to OCR page files.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"wrote {convert(args.engine, args.raw, args.dataset)} OCR pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
