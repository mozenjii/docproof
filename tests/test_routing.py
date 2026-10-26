"""Tests for field routing: the named rules, and an exhaustive sweep proving it fails closed."""

import itertools
from collections.abc import Sequence

import pytest

from docproof.errors import ContractError
from docproof.routing import CheckOutcome, Decision, Proposal, route
from docproof.schema import Check, Reason, Status


def routed(
    *,
    required: bool = True,
    proposal: Proposal = Proposal.VALUE,
    grounded: bool | None = True,
    checks: Sequence[CheckOutcome] = (CheckOutcome(Check.PASSED),),
    confidence: float | None = 0.995,
    threshold: float = 0.99,
    ambiguous: bool = False,
    conflicting: bool = False,
    in_vocabulary: bool = True,
) -> Decision:
    """Route with every signal good unless a test overrides it."""
    return route(
        required=required,
        proposal=proposal,
        grounded=grounded,
        checks=checks,
        confidence=confidence,
        threshold=threshold,
        ambiguous=ambiguous,
        conflicting=conflicting,
        in_vocabulary=in_vocabulary,
    )


def test_all_good_signals_accept() -> None:
    assert routed() == Decision(Status.ACCEPTED)


def test_confidence_equal_to_threshold_accepts() -> None:
    assert routed(confidence=0.99).status is Status.ACCEPTED


def test_no_proposal_abstains_whatever_else_is_true() -> None:
    assert routed(proposal=Proposal.NONE, conflicting=True, grounded=False) == Decision(
        Status.ABSTAINED
    )


def test_conflict_is_reported_alone() -> None:
    assert routed(conflicting=True, grounded=False) == Decision(
        Status.FLAGGED, (Reason.CONFLICTING_CANDIDATES,)
    )


def test_not_present_on_a_required_field_is_always_reviewed() -> None:
    assert routed(proposal=Proposal.NOT_PRESENT, confidence=1.0) == Decision(
        Status.FLAGGED, (Reason.REQUIRED_NOT_FOUND,)
    )


def test_not_present_on_an_optional_field_ignores_grounding_and_checks() -> None:
    decision = routed(
        proposal=Proposal.NOT_PRESENT,
        required=False,
        grounded=None,
        checks=[CheckOutcome(Check.FAILED)],
        ambiguous=True,
    )
    assert decision == Decision(Status.ACCEPTED)
    assert routed(proposal=Proposal.NOT_PRESENT, required=False, confidence=None) == Decision(
        Status.FLAGGED, (Reason.LOW_CONFIDENCE,)
    )


def test_every_applicable_reason_is_collected_and_sorted() -> None:
    decision = routed(
        grounded=None,
        checks=[CheckOutcome(Check.FAILED), CheckOutcome(Check.UNKNOWN)],
        confidence=None,
        ambiguous=True,
        in_vocabulary=False,
    )
    assert decision.status is Status.FLAGGED
    assert decision.reasons == tuple(
        sorted(
            {
                Reason.UNGROUNDED,
                Reason.VALIDATOR_FAILED,
                Reason.VALIDATOR_UNKNOWN,
                Reason.LOW_CONFIDENCE,
                Reason.AMBIGUOUS_RENDERING,
                Reason.OUT_OF_VOCABULARY,
            }
        )
    )


def test_advisory_unknown_does_not_block_but_advisory_failure_does() -> None:
    assert routed(checks=[CheckOutcome(Check.UNKNOWN, advisory=True)]).status is Status.ACCEPTED
    assert routed(checks=[CheckOutcome(Check.FAILED, advisory=True)]).reasons == (
        Reason.VALIDATOR_FAILED,
    )


def test_one_unknown_among_passes_still_flags() -> None:
    checks = [CheckOutcome(Check.PASSED), CheckOutcome(Check.UNKNOWN)]
    assert routed(checks=checks).reasons == (Reason.VALIDATOR_UNKNOWN,)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
def test_threshold_and_confidence_must_be_probabilities(bad: float) -> None:
    with pytest.raises(ContractError) as info:
        routed(threshold=bad)
    assert info.value.code == "invalid_threshold"
    with pytest.raises(ContractError) as info:
        routed(confidence=bad)
    assert info.value.code == "invalid_confidence"


def test_exhaustive_sweep_accepts_a_value_only_when_every_signal_is_good() -> None:
    check_sets = [
        [],
        [CheckOutcome(Check.PASSED)],
        [CheckOutcome(Check.FAILED)],
        [CheckOutcome(Check.UNKNOWN)],
        [CheckOutcome(Check.UNKNOWN, advisory=True)],
        [CheckOutcome(Check.FAILED, advisory=True)],
    ]
    space = itertools.product(
        [True, False],
        list(Proposal),
        [True, False, None],
        range(len(check_sets)),
        [None, 0.0, 0.5, 0.99, 1.0],
        [True, False],
        [True, False],
        [True, False],
    )
    seen = 0
    for required, proposal, grounded, ci, confidence, ambiguous, conflicting, vocab in space:
        checks = check_sets[ci]
        d = route(
            required=required,
            proposal=proposal,
            grounded=grounded,
            checks=checks,
            confidence=confidence,
            threshold=0.99,
            ambiguous=ambiguous,
            conflicting=conflicting,
            in_vocabulary=vocab,
        )
        seen += 1
        assert (d.status is Status.FLAGGED) == bool(d.reasons)
        if d.status is not Status.ACCEPTED:
            continue
        assert confidence is not None
        assert confidence >= 0.99
        assert not conflicting
        if proposal is Proposal.VALUE:
            assert grounded is True
            assert not ambiguous
            assert vocab
            assert all(c.result is Check.PASSED or c.advisory for c in checks)
            assert all(c.result is not Check.FAILED for c in checks)
        else:
            assert proposal is Proposal.NOT_PRESENT
            assert not required
    assert seen == 2 * 3 * 3 * 6 * 5 * 2 * 2 * 2
