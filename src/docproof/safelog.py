"""Logging that cannot carry a document's contents by accident.

Invoices hold names, addresses and bank details, and logs are copied, shipped and kept far longer
than documents. So the rules here are structural rather than a matter of care:

- A log call takes an event name and keyword fields. The event name must look like
  ``extract.field_routed``, so free text cannot hide in it.
- Field values must be numbers, booleans, enums, ``None`` or :class:`Safe` strings. A plain ``str``
  is refused, by the type checker and again at runtime. Text gets into a log only by being wrapped
  in :class:`Safe` on purpose, or by going through :func:`redact`.
- :func:`redact` renders an extracted value by its field's sensitivity: public values as they are,
  business values as a keyed hash (equal values match within a run, but the hash cannot be reversed
  by guessing totals offline), personal and financial values as a bare marker.
- Errors are logged by type and code only. Messages are dropped, because ours quote the offending
  value.

Deliberate boundary: this module is the only one in ``docproof`` allowed to import :mod:`logging`; a
test enforces it. Formatting, handlers and log shipping are the application's choice.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
from dataclasses import dataclass
from enum import Enum

from docproof.labels import encode_value
from docproof.schema import FieldSpec, FieldValue, Sensitivity

_EVENT = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")
_FIELD_KEY = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True, slots=True)
class Safe:
    """A string the caller asserts contains nothing from a document.

    Use it for ids, template names and counts rendered as text. Wrapping is the point: it makes
    the decision visible in review.

    Attributes:
        text: The string.
    """

    text: str


type LogValue = int | float | bool | Enum | Safe | None
"""What a log field may hold. ``str`` is deliberately absent."""


class Redactor:
    """Renders extracted values for logs according to their field's sensitivity.

    Args:
        key: HMAC key for business values. Defaults to a fresh random key per process, so hashes
            correlate within a run and mean nothing outside it.
    """

    def __init__(self, key: bytes | None = None) -> None:
        """Create a redactor with the given or a fresh random key."""
        self._key = key if key is not None else secrets.token_bytes(32)

    def redact(self, spec: FieldSpec, value: FieldValue | None) -> Safe:
        """Render a value for a log.

        Args:
            spec: The field's specification; its sensitivity decides the rendering.
            value: The extracted value, or ``None``.

        Returns:
            A :class:`Safe` rendering: the value (public), ``hmac:<12 hex>`` (business), or
            ``[personal]`` / ``[financial]``. ``None`` is ``[none]`` in every class, since
            absence is not sensitive.
        """
        if value is None:
            return Safe("[none]")
        match spec.sensitivity:
            case Sensitivity.PUBLIC:
                return Safe(str(encode_value(spec.value_type, value)))
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


def _check_fields(fields: dict[str, LogValue]) -> None:
    """Refuse field names and values that could carry document text."""
    for key, value in fields.items():
        if not _FIELD_KEY.fullmatch(key):
            raise ValueError(f"log field name {key!r} must be lower_snake_case")
        # bool is an int, and both are fine; str is the type this whole module exists to stop.
        if not isinstance(value, int | float | Enum | Safe) and value is not None:
            raise TypeError(
                f"log field {key!r} got {type(value).__name__}; wrap text in Safe or redact it"
            )


class SafeLogger:
    """A structured logger whose fields cannot be raw strings.

    Args:
        name: Logger name, usually the module's ``__name__``.
    """

    def __init__(self, name: str) -> None:
        """Wrap the standard logger called ``name``."""
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, fields: dict[str, LogValue]) -> None:
        if not _EVENT.fullmatch(event):
            raise ValueError(f"event {event!r} must be dotted lower_snake_case, not free text")
        _check_fields(fields)
        if not self._logger.isEnabledFor(level):
            return
        rendered = " ".join(f"{k}={_render(v)}" for k, v in sorted(fields.items()))
        self._logger.log(level, "%s %s", event, rendered)

    def debug(self, event: str, **fields: LogValue) -> None:
        """Log at DEBUG. See the module docstring for what fields may hold."""
        self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: LogValue) -> None:
        """Log at INFO."""
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, **fields: LogValue) -> None:
        """Log at WARNING."""
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, error: BaseException, **fields: LogValue) -> None:
        """Log an error by its type and code, never its message.

        Args:
            event: Event name.
            error: The exception. Its ``code`` attribute is logged when it is a string; its
                message and traceback are not, because they can quote document values.
            **fields: Extra fields.
        """
        code = getattr(error, "code", None)
        extra: dict[str, LogValue] = {"error_type": Safe(type(error).__name__)}
        if isinstance(code, str) and _FIELD_KEY.fullmatch(code):
            extra["error_code"] = Safe(code)
        self._emit(logging.ERROR, event, {**fields, **extra})
