"""Uniform adapters over the candidate OCR engines.

Purpose: hide each engine's API behind one call, ``run(image) -> list[Detection]``, where a
detection is ``(text, box, confidence)`` in page pixel coordinates.

Boundary: adapters only translate inputs and outputs. They do not score, time or post-process
text, and they do not tune engines beyond switching off page-level preprocessing that the other
engines do not perform. Each engine is imported inside its adapter's constructor, so a machine
or virtual environment without that engine can still use every other adapter; a missing engine
raises :class:`EngineUnavailableError` with the underlying reason.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
from PIL import Image

type Point = tuple[float, float]
type Box = tuple[Point, Point, Point, Point]


class Detection(NamedTuple):
    """One recognised text span.

    Attributes:
        text: Recognised text; a line or a word depending on the engine's granularity.
        box: Four corners (top-left, top-right, bottom-right, bottom-left) in pixels of the input
            image, or ``None`` when the engine gives no geometry.
        confidence: Engine-reported score in ``[0, 1]``, or ``None`` when the engine gives none.
    """

    text: str
    box: Box | None
    confidence: float | None


class EngineUnavailableError(RuntimeError):
    """Raised when an engine's package or system binary is not present."""


class OcrEngine(Protocol):
    """The contract every adapter satisfies."""

    name: str
    """Registry name of the engine."""

    granularity: str
    """``"line"`` or ``"word"``: what one :class:`Detection` covers."""

    packages: tuple[str, ...]
    """Distribution names whose installed versions identify this engine build."""

    settings: dict[str, str]
    """The effective configuration that differs from or pins down the engine's defaults."""

    def run(self, image: Image.Image) -> list[Detection]:
        """Recognise text on one RGB page image."""
        ...

    def model_dirs(self) -> list[Path]:
        """Directories that hold downloaded model weights, for size reporting."""
        ...


def _quad(points: Sequence[Sequence[float]]) -> Box:
    """Convert a 4-point polygon of any numeric type into a :data:`Box` of floats.

    Raises:
        ValueError: If the polygon does not have exactly four points.
    """
    if len(points) != 4:
        raise ValueError(f"expected a 4-point polygon, got {len(points)} points")
    p0, p1, p2, p3 = ((float(x), float(y)) for x, y in points)
    return (p0, p1, p2, p3)


def _rect(xmin: float, ymin: float, xmax: float, ymax: float) -> Box:
    """Convert an axis-aligned rectangle into a :data:`Box`."""
    return ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))


def _to_bgr(image: Image.Image) -> np.ndarray:
    """Return a contiguous BGR uint8 array, the layout OpenCV-based engines expect."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])


class RapidOcrEngine:
    """RapidOCR: PaddleOCR detection and recognition models executed by ONNX Runtime."""

    name = "rapidocr"
    granularity = "line"
    packages = ("rapidocr", "onnxruntime", "opencv-python-headless", "numpy")

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Import RapidOCR and build the pipeline.

        Args:
            params: RapidOCR configuration overrides keyed like ``"Det.model_type"``; ``None``
                keeps the package defaults.

        Raises:
            EngineUnavailableError: If the ``rapidocr`` package cannot be imported.
        """
        try:
            import rapidocr
        except ImportError as exc:
            raise EngineUnavailableError(f"rapidocr is not installed: {exc}") from exc

        self._package_dir = Path(rapidocr.__file__).parent
        self.settings = {key: str(value) for key, value in (params or {}).items()}
        self._engine = rapidocr.RapidOCR(params=params)

    def run(self, image: Image.Image) -> list[Detection]:
        """Recognise text lines.

        Args:
            image: RGB page image.

        Returns:
            One detection per text line, with a 4-point box and a recognition score.
        """
        result = self._engine(_to_bgr(image))
        if result.txts is None or result.boxes is None or result.scores is None:
            return []
        return [
            Detection(str(text), _quad(box), float(score))
            for text, box, score in zip(result.txts, result.boxes, result.scores, strict=True)
        ]

    def model_dirs(self) -> list[Path]:
        """Return the package-local model directory (it lives inside the virtual environment)."""
        return [self._package_dir / "models"]


class RapidOcrMobileEngine(RapidOcrEngine):
    """RapidOCR with PP-OCRv5 mobile models and no text-line orientation classifier.

    It uses the same models and stages as ``paddleocr-mobile``, so the pair isolates the runtime:
    ONNX Runtime against PaddlePaddle.
    """

    name = "rapidocr-mobile"

    def __init__(self) -> None:
        """Build the pipeline with the mobile-sized models; they download on first use.

        Raises:
            EngineUnavailableError: If the ``rapidocr`` package cannot be imported.
        """
        try:
            from rapidocr import LangRec, ModelType, OCRVersion
        except ImportError as exc:
            raise EngineUnavailableError(f"rapidocr is not installed: {exc}") from exc
        super().__init__(
            params={
                "Global.use_cls": False,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.lang_type": LangRec.EN,
            }
        )


class PaddleOcrEngine:
    """PaddleOCR 3.x on the PaddlePaddle CPU runtime.

    Document orientation classification, unwarping and text-line orientation are switched off so
    the pipeline does the same detect-then-recognise work as the other engines.

    oneDNN (MKL-DNN) is switched off only for paddlepaddle 3.3.x, where on Windows the oneDNN
    executor fails on the first prediction with ``NotImplementedError: ConvertPirAttribute2Runtime
    Attribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]``. Without oneDNN,
    inference runs on the plain CPU kernels and is several times slower, so paddlepaddle 3.2.x
    gives the representative figures.
    """

    onednn_broken_prefix = "3.3."
    """paddlepaddle versions with this prefix run with oneDNN disabled."""

    name = "paddleocr"
    granularity = "line"
    packages = ("paddleocr", "paddlepaddle", "paddlex", "numpy")

    def __init__(self, **model_options: str) -> None:
        """Import PaddleOCR and build the pipeline.

        Args:
            **model_options: Extra keyword arguments for ``PaddleOCR``, for example
                ``text_detection_model_name`` to select a lighter model.

        Raises:
            EngineUnavailableError: If ``paddleocr`` or ``paddle`` cannot be imported.
        """
        try:
            import paddle
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise EngineUnavailableError(f"paddleocr is not installed: {exc}") from exc
        language = {} if model_options else {"lang": "en"}
        onednn = not str(paddle.__version__).startswith(self.onednn_broken_prefix)
        self.settings = {"enable_mkldnn": str(onednn), **language, **model_options}
        self._engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=onednn,
            **language,
            **model_options,
        )

    def run(self, image: Image.Image) -> list[Detection]:
        """Recognise text lines.

        Args:
            image: RGB page image.

        Returns:
            One detection per text line, with a 4-point box and a recognition score.
        """
        detections: list[Detection] = []
        for page in self._engine.predict(_to_bgr(image)):
            texts, scores, polys = page["rec_texts"], page["rec_scores"], page["rec_polys"]
            detections.extend(
                Detection(str(text), _quad(poly), float(score))
                for text, score, poly in zip(texts, scores, polys, strict=True)
            )
        return detections

    def model_dirs(self) -> list[Path]:
        """Return the PaddleX model cache (``PADDLE_PDX_CACHE_HOME`` or ``~/.paddlex``)."""
        cache_home = os.environ.get("PADDLE_PDX_CACHE_HOME")
        root = Path(cache_home) if cache_home else Path.home() / ".paddlex"
        return [root / "official_models"]


class PaddleOcrMobileEngine(PaddleOcrEngine):
    """PaddleOCR 3.x with the PP-OCRv5 mobile detector and the English mobile recogniser."""

    name = "paddleocr-mobile"

    def __init__(self) -> None:
        """Build the pipeline with the mobile-sized models.

        Raises:
            EngineUnavailableError: If ``paddleocr`` or ``paddle`` cannot be imported.
        """
        super().__init__(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        )


class DoctrEngine:
    """docTR with its default detection and recognition architectures on the PyTorch backend."""

    name = "doctr"
    granularity = "word"
    packages = ("python-doctr", "torch", "torchvision", "numpy")

    def __init__(self) -> None:
        """Import docTR and load the pretrained default predictor.

        Raises:
            EngineUnavailableError: If ``doctr`` or its backend cannot be imported.
        """
        try:
            from doctr.models import ocr_predictor
        except ImportError as exc:
            raise EngineUnavailableError(f"python-doctr is not installed: {exc}") from exc
        self._predictor = ocr_predictor(pretrained=True)
        self.settings = {
            "det_arch": type(self._predictor.det_predictor.model).__name__,
            "reco_arch": type(self._predictor.reco_predictor.model).__name__,
        }

    def run(self, image: Image.Image) -> list[Detection]:
        """Recognise words.

        Args:
            image: RGB page image.

        Returns:
            One detection per word, with an axis-aligned box and a recognition confidence.
        """
        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
        height, width = rgb.shape[:2]
        document = self._predictor([rgb])
        detections: list[Detection] = []
        for block in document.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    (xmin, ymin), (xmax, ymax) = word.geometry
                    box = _rect(xmin * width, ymin * height, xmax * width, ymax * height)
                    detections.append(Detection(word.value, box, float(word.confidence)))
        return detections

    def model_dirs(self) -> list[Path]:
        """Return the docTR weight cache (``DOCTR_CACHE_DIR`` or ``~/.cache/doctr``)."""
        cache_dir = os.environ.get("DOCTR_CACHE_DIR")
        return [Path(cache_dir) if cache_dir else Path.home() / ".cache" / "doctr"]


class TesseractEngine:
    """Tesseract through ``pytesseract``; needs the ``tesseract`` system binary on PATH."""

    name = "tesseract"
    granularity = "word"
    packages = ("pytesseract",)

    def __init__(self) -> None:
        """Check for the binary first, then import the Python wrapper.

        Raises:
            EngineUnavailableError: If the binary is not on PATH or ``pytesseract`` is missing.
        """
        if shutil.which("tesseract") is None:
            raise EngineUnavailableError("tesseract system binary not found on PATH")
        try:
            import pytesseract
        except ImportError as exc:
            raise EngineUnavailableError(f"pytesseract is not installed: {exc}") from exc
        self._pytesseract = pytesseract
        self.settings = {"tesseract_version": str(pytesseract.get_tesseract_version())}

    def run(self, image: Image.Image) -> list[Detection]:
        """Recognise words.

        Args:
            image: RGB page image.

        Returns:
            One detection per non-empty word. Tesseract reports confidence in 0-100; it is
            rescaled to ``[0, 1]``.
        """
        data = self._pytesseract.image_to_data(image, output_type=self._pytesseract.Output.DICT)
        detections: list[Detection] = []
        for text, left, top, width, height, conf in zip(
            data["text"],
            data["left"],
            data["top"],
            data["width"],
            data["height"],
            data["conf"],
            strict=True,
        ):
            if not str(text).strip():
                continue
            box = _rect(float(left), float(top), float(left + width), float(top + height))
            detections.append(Detection(str(text), box, float(conf) / 100))
        return detections

    def model_dirs(self) -> list[Path]:
        """Tesseract language data lives with the system install; nothing is reported."""
        return []


ENGINES: dict[str, Callable[[], OcrEngine]] = {
    RapidOcrEngine.name: RapidOcrEngine,
    RapidOcrMobileEngine.name: RapidOcrMobileEngine,
    PaddleOcrEngine.name: PaddleOcrEngine,
    PaddleOcrMobileEngine.name: PaddleOcrMobileEngine,
    DoctrEngine.name: DoctrEngine,
    TesseractEngine.name: TesseractEngine,
}
"""Registry of engine names to constructors. Constructing an engine imports it."""


def create_engine(name: str) -> OcrEngine:
    """Import and construct an engine by registry name.

    Args:
        name: A key of :data:`ENGINES`.

    Returns:
        A ready engine. Model weights may be downloaded on first construction.

    Raises:
        KeyError: If ``name`` is not registered.
        EngineUnavailableError: If the engine's package or binary is missing.
    """
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    return ENGINES[name]()
