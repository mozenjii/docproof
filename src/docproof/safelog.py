"""Logging that cannot carry a document's contents by accident.

Invoices hold names, addresses and bank details, and logs are copied, shipped and kept far longer
than documents. So the rules here are structural rather than a matter of care:

- A log call takes an event name and keyword fields. The event name must look like
  ``extract.field_routed``, so free text cannot hide in it.
- Field values must be exact ``int``/``float``/``bool``, ``None``, a member of one of docproof's
  own enums, or :class:`Safe` text. A plain ``str``, a subclass of a number and an enum made up on
  the spot are all refused, by the type checker where it can see them and always at runtime.
- :class:`Safe` text may not contain line breaks or other control characters, so a value cannot
  forge a second log record.
- A field whose name matches a document field or a money-ish word (``total``, ``iban``,
  ``amount``…) must be :class:`Safe`, which in practice means it came from :func:`Redactor.redact`.
  That closes the easy mistake of logging ``float(total)`` as a number.
- :meth:`Redactor.redact` renders an extracted value by its field's sensitivity: public values only
  if they look like a code (otherwise ``[unvalidated]``), business values as a keyed hash, personal
  and financial values as a bare marker.
- Errors are logged by type and code only. Messages are dropped, because ours quote the offending
  value, and refusals here never echo the text they refused.

Accepted limit: a contributor can still convert a sensitive value to a number and log it under an
innocent field name. The rules make that a deliberate act that shows up in review, not an accident.

Deliberate boundary: this module is the only one in ``docproof`` allowed to import :mod:`logging`; a
source-policy test enforces it. Formatting, handlers and shipping are the application's choice.
"""

from __future__ import annotations

import hashlib
import hmac
import logging as _logging
import re
import secrets
from dataclasses import dataclass
from enum import Enum

from docproof.labels import encode_value
from docproof.schema import SCHEMAS, FieldSpec, FieldValue, Sensitivity

_EVENT = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")
_FIELD_KEY = re.compile(r"[a-z][a-z0-9_]*")
_PUBLIC_TEXT = re.compile(r"[A-Za-z0-9._-]{1,32}")
_CONTROL = re.compile("[\x00-\x1f\x7f\x85\u2028\u2029]")
_MIN_KEY_BYTES = 16

SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {spec.name for specs in SCHEMAS.values() for spec in specs}
    | {"amount", "account", "iban", "price", "balance", "value", "name", "address"}
)
"""Log field names that must carry :class:`Safe` text, never a bare number or enum."""


@dataclass(frozen=True, slots=True)
class Safe:
    """A string the caller asserts contains nothing from a document.

    Use it for ids, template names and counts rendered as text. Wrapping is the point: it makes
    the decision visible in review.

    Attributes:
        text: The string. Control characters, including line breaks, are refused.
    """

    text: str

    def __post_init__(self) -> None:
        """Refuse text that could break a log line in two."""
        if _CONTROL.search(self.text):
            raise ValueError("Safe text may not contain line breaks or control characters")


type LogValue = int | float | bool | Enum | Safe | None
"""What a log field may hold. ``str`` is deliberately absent."""


class Redactor:
    """Renders extracted values for logs according to their field's sensitivity.

    Args:
        key: HMAC key for business values, at least 16 bytes. Defaults to a fresh random key per
            process, so hashes correlate within a run and mean nothing outside it.
    """

    def __init__(self, key: bytes | None = None) -> None:
        """Create a redactor with the given or a fresh random key."""
        if key is not None and len(key) < _MIN_KEY_BYTES:
            raise ValueError(f"the redaction key must be at least {_MIN_KEY_BYTES} bytes")
        self._key = key if key is not None else secrets.token_bytes(32)

    def redact(self, spec: FieldSpec, value: FieldValue | None) -> Safe:
        """Render a value for a log.

        Args:
            spec: The field's specification; its sensitivity decides the rendering.
            value: The extracted value, or ``None``.

        Returns:
            ``[none]`` for ``None``; for public fields the value if it looks like a short code and
            ``[unvalidated]`` otherwise (an extractor can put anything in a public field);
            ``hmac:<12 hex>`` for business fields; ``[personal]`` / ``[financial]`` otherwise.
        """
        if value is None:
            return Safe("[none]")
        match spec.sensitivity:
            case Sensitivity.PUBLIC:
                text = str(encode_value(spec.value_type, value))
                return Safe(text if _PUBLIC_TEXT.fullmatch(text) else "[unvalidated]")
            case Sensitivity.BUSINESS:
                canonical = repr(encode_value(spec.value_type, value)).encode()
                digest = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
                return Safe(f"hmac:{digest[:12]}")
            case Sensitivity.PERSONAL | Sensitivity.FINANCIAL:
                return Safe(f"[{spec.sensitivity.value}]")


def _render(value: LogValue) -> str:
    """Render one checked log value."""
    match value:
        case Safe():
            return value.text
        case Enum():
            return str(value.value)
        case _:
            return str(value)


def _allowed(value: object) -> bool:
    """Whether a value may be logged as it is."""
    if value is None or isinstance(value, Safe):
        return True
    if type(value) in (int, float, bool):  # exact types: a subclass can override __str__
        return True
    return isinstance(value, Enum) and type(value).__module__.startswith("docproof.")


def _check_fields(fields: dict[str, LogValue]) -> None:
    """Refuse field names and values that could carry document text.

    Messages name the problem, never the refused text, so a refusal cannot leak what it refused.
    """
    for key, value in fields.items():
        if not _FIELD_KEY.fullmatch(key):
            raise ValueError("a log field name is not lower_snake_case")
        if not _allowed(value):
            raise TypeError(
                f"log field {key!r} got {type(value).__name__}; wrap text in Safe or redact it"
            )
        if key in SENSITIVE_FIELD_NAMES and not isinstance(value, Safe | None):
            raise TypeError(f"log field {key!r} names document data; pass Redactor.redact(...)")


class SafeLogger:
    """A structured logger whose fields cannot be raw strings.

    Args:
        name: Logger name: a dotted module path, usually ``__name__``.
    """

    def __init__(self, name: str) -> None:
        """Wrap the standard logger called ``name``."""
        if not _EVENT.fullmatch(name):
            raise ValueError("a logger name must be a dotted lower_snake_case module path")
        self._logger = _logging.getLogger(name)

    def _emit(self, level: int, event: str, fields: dict[str, LogValue]) -> None:
        if not _EVENT.fullmatch(event):
            raise ValueError("an event name must be dotted lower_snake_case, not free text")
        _check_fields(fields)
        if not self._logger.isEnabledFor(level):
            return
        rendered = " ".join(f"{k}={_render(v)}" for k, v in sorted(fields.items()))
        self._logger.log(level, "%s %s", event, rendered)

    def debug(self, event: str, **fields: LogValue) -> None:
        """Log at DEBUG. See the module docstring for what fields may hold."""
        self._emit(_logging.DEBUG, event, fields)

    def info(self, event: str, **fields: LogValue) -> None:
        """Log at INFO."""
        self._emit(_logging.INFO, event, fields)

    def warning(self, event: str, **fields: LogValue) -> None:
        """Log at WARNING."""
        self._emit(_logging.WARNING, event, fields)

    def error(self, event: str, error: BaseException, **fields: LogValue) -> None:
        """Log an error by its type and code, never its message.

        Args:
            event: Event name.
            error: The exception. Its ``code`` attribute is logged when it is a lower_snake_case
                string; its message, cause and traceback are not, because they can quote document
                values.
            **fields: Extra fields.
        """
        code = getattr(error, "code", None)
        extra: dict[str, LogValue] = {"error_type": Safe(type(error).__name__)}
        if isinstance(code, str) and _FIELD_KEY.fullmatch(code):
            extra["error_code"] = Safe(code)
        self._emit(_logging.ERROR, event, {**fields, **extra})
