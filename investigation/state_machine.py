"""Case status state machine - shared by the API's action endpoint and
its own test suite, so the transition rules have exactly one definition.

    OPEN ----------> IN_REVIEW ----------> ESCALATED
      \\                 |    \\                |
       \\                |     \\-------------->|
        \\               v                      v
         \\--------> CONFIRMED_FRAUD or FALSE_POSITIVE ----> CLOSED

ASSIGN doesn't change status (it only sets assigned_investigator) and is
handled by a separate endpoint (POST /cases/{id}/assign) rather than
through this state machine.
"""

ACTION_TRANSITIONS = {
    # action_type: {from_status: to_status}
    "INVESTIGATE": {"OPEN": "IN_REVIEW", "ESCALATED": "IN_REVIEW"},
    "ESCALATE": {"OPEN": "ESCALATED", "IN_REVIEW": "ESCALATED"},
    "CONFIRM_FRAUD": {"IN_REVIEW": "CONFIRMED_FRAUD", "ESCALATED": "CONFIRMED_FRAUD"},
    "MARK_FALSE_POSITIVE": {"IN_REVIEW": "FALSE_POSITIVE", "ESCALATED": "FALSE_POSITIVE"},
    "CLOSE": {"CONFIRMED_FRAUD": "CLOSED", "FALSE_POSITIVE": "CLOSED"},
}

TERMINAL_STATUSES = {"CLOSED"}
RESOLUTION_STATUSES = {"CONFIRMED_FRAUD", "FALSE_POSITIVE"}


class InvalidTransitionError(ValueError):
    pass


def apply_action(current_status: str, action_type: str) -> str:
    """Returns the new status for `action_type` applied to `current_status`,
    or raises InvalidTransitionError if that action isn't valid from the
    current status.
    """
    if action_type not in ACTION_TRANSITIONS:
        raise InvalidTransitionError(f"unknown action_type: {action_type}")
    transitions = ACTION_TRANSITIONS[action_type]
    if current_status not in transitions:
        raise InvalidTransitionError(
            f"action {action_type} is not valid from status {current_status} "
            f"(valid from: {sorted(transitions.keys())})"
        )
    return transitions[current_status]


def valid_actions_from(current_status: str) -> list[str]:
    """Which actions can legally be applied given the current status -
    useful for an API/UI to know what buttons to show.
    """
    return [action for action, transitions in ACTION_TRANSITIONS.items() if current_status in transitions]
