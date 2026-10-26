"""Decide each field's status from its signals: accept it, flag it for a human, or abstain.

This is where the precision promise is kept or broken, so it fails closed. A field is accepted
only when every signal agrees: the value was found on the page, no validator failed or came
back unknown, the calibrated confidence clears the threshold, and nothing about the rendering is
ambiguous. Every reason that applies is reported, not just the first, so a reviewer sees the
whole picture and the flag-reason statistics are honest.

Deliberate boundary: routing does not compute any signal. Grounding, validators and confidence
are inputs. The threshold is an input too, because it is chosen on validation data for a target
precision that is still an open product decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from docproof.errors import ContractError
from docproof.schema import Check, Reason, Status


class Proposal(StrEnum):
    """What the extractor proposed for a field."""

    VALUE = "value"
    NOT_PRESENT = "not_present"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """One validator's result as routing sees it.

    Attributes:
        result: The validator's answer.
        advisory: When true, an ``UNKNOWN`` result does not block acceptance. A ``FAILED`` one
            always does.
    """

    result: Check
    advisory: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    """A field's routed outcome.

    Attributes:
        status: Accepted, flagged or abstained.
        reasons: Why it was flagged, sorted; empty unless flagged.
    """

    status: Status
    reasons: tuple[Reason, ...] = ()


def _unit_interval(name: str, value: float, code: str) -> None:
    """Raise unless ``value`` is a real number in [0, 1]."""
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ContractError(code, f"{name} must be in [0, 1], got {value}")


def _flagged(reasons: set[Reason]) -> Decision:
    """Build a flagged decision with reasons in a stable order."""
    return Decision(Status.FLAGGED, tuple(sorted(reasons)))


def route(
    *,
    required: bool,
    proposal: Proposal,
    grounded: bool | None,
    checks: Sequence[CheckOutcome],
    confidence: float | None,
    threshold: float,
    ambiguous: bool,
    conflicting: bool,
    in_vocabulary: bool,
) -> Decision:
    """Route one field.

    Args:
        required: Whether the document is incomplete without this field.
        proposal: What the extractor proposed.
        grounded: Whether the value was located on the page; ``None`` means not checked, which
            counts as not grounded.
        checks: Results of the validators that read this field.
        confidence: Calibrated probability the value is correct, or ``None`` if not computed.
        threshold: Minimum confidence to accept.
        ambiguous: The printed rendering has more than one reading.
        conflicting: Candidate values disagree after normalisation.
        in_vocabulary: For closed-set fields, the value is a member; ``True`` for open fields.

    Returns:
        The decision.

    Raises:
        ContractError: ``invalid_threshold`` or ``invalid_confidence`` if either is outside [0, 1].
    """
    _unit_interval("threshold", threshold, "invalid_threshold")
    if confidence is not None:
        _unit_interval("confidence", confidence, "invalid_confidence")
    confident = confidence is not None and confidence >= threshold

    if proposal is Proposal.NONE:
        return Decision(Status.ABSTAINED)
    if conflicting:
        return _flagged({Reason.CONFLICTING_CANDIDATES})
    if proposal is Proposal.NOT_PRESENT:
        if required:
            return _flagged({Reason.REQUIRED_NOT_FOUND})
        return Decision(Status.ACCEPTED) if confident else _flagged({Reason.LOW_CONFIDENCE})

    reasons: set[Reason] = set()
    if grounded is not True:
        reasons.add(Reason.UNGROUNDED)
    if any(c.result is Check.FAILED for c in checks):
        reasons.add(Reason.VALIDATOR_FAILED)
    if any(c.result is Check.UNKNOWN and not c.advisory for c in checks):
        reasons.add(Reason.VALIDATOR_UNKNOWN)
    if not confident:
        reasons.add(Reason.LOW_CONFIDENCE)
    if ambiguous:
        reasons.add(Reason.AMBIGUOUS_RENDERING)
    if not in_vocabulary:
        reasons.add(Reason.OUT_OF_VOCABULARY)
    return _flagged(reasons) if reasons else Decision(Status.ACCEPTED)
