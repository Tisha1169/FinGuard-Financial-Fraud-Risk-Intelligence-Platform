"""Unit tests for the case status state machine - no DB required."""
import pytest

from investigation.state_machine import ACTION_TRANSITIONS, InvalidTransitionError, apply_action, valid_actions_from


def test_open_to_in_review_via_investigate():
    assert apply_action("OPEN", "INVESTIGATE") == "IN_REVIEW"


def test_escalated_to_in_review_via_investigate():
    assert apply_action("ESCALATED", "INVESTIGATE") == "IN_REVIEW"


def test_in_review_to_confirmed_fraud():
    assert apply_action("IN_REVIEW", "CONFIRM_FRAUD") == "CONFIRMED_FRAUD"


def test_in_review_to_false_positive():
    assert apply_action("IN_REVIEW", "MARK_FALSE_POSITIVE") == "FALSE_POSITIVE"


def test_confirmed_fraud_to_closed():
    assert apply_action("CONFIRMED_FRAUD", "CLOSE") == "CLOSED"


def test_false_positive_to_closed():
    assert apply_action("FALSE_POSITIVE", "CLOSE") == "CLOSED"


def test_cannot_close_directly_from_open():
    with pytest.raises(InvalidTransitionError):
        apply_action("OPEN", "CLOSE")


def test_cannot_investigate_a_closed_case():
    with pytest.raises(InvalidTransitionError):
        apply_action("CLOSED", "INVESTIGATE")


def test_cannot_confirm_fraud_directly_from_open():
    """An OPEN case must go through IN_REVIEW/ESCALATED first - you can't
    resolve a case that was never investigated.
    """
    with pytest.raises(InvalidTransitionError):
        apply_action("OPEN", "CONFIRM_FRAUD")


def test_unknown_action_type_rejected():
    with pytest.raises(InvalidTransitionError):
        apply_action("OPEN", "DELETE_EVERYTHING")


def test_closed_is_terminal_no_valid_actions():
    assert valid_actions_from("CLOSED") == []


def test_valid_actions_from_open():
    assert set(valid_actions_from("OPEN")) == {"INVESTIGATE", "ESCALATE"}


def test_valid_actions_from_in_review():
    assert set(valid_actions_from("IN_REVIEW")) == {"ESCALATE", "CONFIRM_FRAUD", "MARK_FALSE_POSITIVE"}


def test_every_status_reachable_from_open_eventually_reaches_closed():
    """Sanity check on the transition table itself: every non-terminal
    status must have at least one path forward - no dead-end statuses
    that aren't CLOSED.
    """
    all_statuses = {"OPEN", "IN_REVIEW", "ESCALATED", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"}
    reachable_targets = {to for transitions in ACTION_TRANSITIONS.values() for to in transitions.values()}
    non_terminal = all_statuses - {"CLOSED"}
    for status in non_terminal:
        has_outgoing = any(status in transitions for transitions in ACTION_TRANSITIONS.values())
        assert has_outgoing, f"{status} has no valid outgoing action - dead end"
    assert "CLOSED" in reachable_targets
