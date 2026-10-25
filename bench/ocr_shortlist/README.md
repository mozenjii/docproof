# OCR shortlist benchmark

A small, repeatable comparison of candidate OCR engines for docproof. It renders synthetic
invoices whose text is known exactly, runs one engine over them, and records how fast the engine
is and how much of the printed text it reads back. It is measurement tooling, not product code:
nothing in docproof imports it and it never sees a real document.

## What it measures

For one engine per run:

- **Cold start**: seconds to import the engine package and construct it in a fresh interpreter.
  If the model weights are not cached yet, this includes the download, so I run each engine
  twice and keep the second figure.
- **First inference**: the first call after construction, reported separately because some
  engines finish initialising lazily. It is excluded from the per-page timings.
- **Seconds per page**: the median of `--runs` timed runs per page, averaged over pages, for
  clean pages and for degraded pages.
- **Word recall**: the share of ground-truth words that appear in the engine's output, averaged
  over pages. Both sides are split on whitespace, case-folded and stripped of everything except
  ASCII letters and digits (`1,234.50` becomes `123450`). Matching counts multiplicity, so a word
  printed twice must be read twice. An engine that splits or merges a token differently from
  how it was printed (for example `INV-2026-1234` read as two pieces) loses that token.
- **Geometry and confidence**: whether every detection carried a box and a score, and whether
  the engine reports lines or words.
- **Footprint and versions**: the size of the virtual environment, the size of any separate
  model cache, and the installed versions of the engine's packages.

## Test pages

`pages.py` draws an A4 invoice at 200 DPI from a fixed seed: an invented vendor and customer,
invoice number, dates, a purchase order reference, five to eight freight line items with
quantities and prices, subtotal, VAT and total. Each page also has a degraded variant: rotated
1.5 degrees, Gaussian blur of radius 1, and saved as JPEG at quality 30. Arial is used when it
exists, then DejaVu Sans, then Pillow's bundled font.

## Running it

Each engine gets its own throwaway virtual environment so their dependencies never meet. I keep
them outside the repository (`VENVS` below is any scratch directory) and run from the repository
root, with `uv`:

```sh
uv venv --python 3.12 "$VENVS"/rapidocr
uv pip install --python "$VENVS"/rapidocr/Scripts/python.exe rapidocr onnxruntime pillow numpy
"$VENVS"/rapidocr/Scripts/python.exe -m bench.ocr_shortlist.run_bench \
    --engine rapidocr --out "$VENVS"/results
```

Other engines follow the same pattern:

| Engine name | Install into its environment |
|---|---|
| `rapidocr` | `rapidocr onnxruntime pillow numpy` |
| `paddleocr`, `paddleocr-mobile` | `paddlepaddle paddleocr pillow numpy` |
| `doctr` | `torch torchvision` from the PyTorch CPU index, then `python-doctr pillow numpy` |
| `tesseract` | `pytesseract pillow numpy`, plus the Tesseract system binary on `PATH` |

The PyTorch CPU index is `https://download.pytorch.org/whl/cpu`; it avoids pulling CUDA builds.

`paddleocr` uses PaddleOCR's default (server) models; `paddleocr-mobile` selects the PP-OCRv5
mobile detector and the English mobile recogniser. Both switch off document orientation,
unwarping and text-line orientation so every engine does the same detect-and-recognise work.

Model caches can be pointed away from the home directory with `PADDLE_PDX_CACHE_HOME` (PaddleOCR)
and `DOCTR_CACHE_DIR` (docTR). RapidOCR keeps its models inside its package directory.

Options: `--pages` (default 3), `--runs` (default 3), `--seed`, and `--save-pages` to write the
page images and ground truth under `<out>/pages` for inspection.

## Output

`<out>/<engine>.json` holds the environment, settings, cold start, first inference, per-page
timings and recall (including the missed tokens), a summary, versions and footprint.
`<out>/raw/<engine>/` holds every engine's detections for each page variant.

The exit status is 0 on success, 2 when the engine is not installed (the reason is in the JSON),
and 1 when anything else fails (the error and traceback are in the JSON and on stderr).

## Limits

Synthetic pages are clean digital renders with one font, so recall here is an upper bound on what
an engine achieves on real scans. Timings depend on the machine and on the thread count each
runtime picks by default; compare engines measured on the same machine in the same session.
The Tesseract adapter needs a system binary and has not been exercised on a machine that has one.
