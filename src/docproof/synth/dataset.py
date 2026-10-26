"""Write a synthetic dataset to disk: page images, labels, word boxes and a manifest.

Run ``python -m docproof.synth.dataset --out data/synthetic --per-template 10``. The same arguments
always write the same files, so a dataset is described by its command line and need not be
committed.

Layout::

    <out>/manifest.jsonl                 one line per document
    <out>/<split>/<doc_id>.labels.json   every schema field's true value
    <out>/<split>/<doc_id>.L<n>.png      the page at degradation level n
    <out>/<split>/<doc_id>.L<n>.words.json  word boxes on that image

Deliberate boundary: this module only writes files; sampling and drawing live next door.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path

from docproof.labels import encode_labels
from docproof.synth.render import LEVELS, RenderedPage, degrade, render
from docproof.synth.sample import sample_document
from docproof.synth.templates import TEMPLATES, template


def _noise_seed(seed: int, doc_id: str, level: int) -> int:
    """A stable per-page seed. ``hash()`` would differ between processes for strings."""
    return zlib.crc32(f"{seed}:{doc_id}:{level}".encode())


def _words_json(page: RenderedPage) -> list[dict[str, object]]:
    return [
        {"text": w.text, "box": list(w.box), "field": w.field, "row": w.row} for w in page.words
    ]


def write_dataset(
    out: Path,
    per_template: int,
    seed: int,
    levels: Sequence[int],
    template_ids: Sequence[str] | None = None,
) -> int:
    """Generate and write documents for every template.

    Args:
        out: Output directory; created if missing.
        per_template: Documents per template.
        seed: Dataset seed.
        levels: Degradation levels to write, indexes into :data:`LEVELS`.
        template_ids: Only these templates; ``None`` means all of them.

    Returns:
        Number of documents written.

    Raises:
        ValueError: If ``per_template`` < 1, a level is out of range or a template is unknown.
    """
    if per_template < 1:
        raise ValueError("per_template must be at least 1")
    if not levels or any(not 0 <= level < len(LEVELS) for level in levels):
        raise ValueError(f"levels must be within 0..{len(LEVELS) - 1}")
    chosen = [template(t) for t in template_ids] if template_ids else list(TEMPLATES)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with (out / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as manifest:
        for tpl in chosen:
            folder = out / tpl.split.value
            folder.mkdir(exist_ok=True)
            for index in range(per_template):
                doc = sample_document(tpl, index, seed)
                labels = encode_labels(tpl.doc_type, doc.labels)
                (folder / f"{doc.doc_id}.labels.json").write_text(
                    json.dumps(labels, indent=1), encoding="utf-8"
                )
                clean = render(doc)
                for level in levels:
                    page = degrade(clean, LEVELS[level], seed=_noise_seed(seed, doc.doc_id, level))
                    page.image.save(folder / f"{doc.doc_id}.L{level}.png")
                    (folder / f"{doc.doc_id}.L{level}.words.json").write_text(
                        json.dumps(_words_json(page)), encoding="utf-8"
                    )
                entry = {
                    "doc_id": doc.doc_id,
                    "template": tpl.template_id,
                    "doc_type": tpl.doc_type.value,
                    "split": tpl.split.value,
                    "levels": list(levels),
                }
                manifest.write(json.dumps(entry) + "\n")
                count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Write a synthetic document dataset.")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--per-template", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--templates", nargs="+", help="template ids; default all")
    args = parser.parse_args(argv)
    try:
        count = write_dataset(args.out, args.per_template, args.seed, args.levels, args.templates)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {count} documents to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
