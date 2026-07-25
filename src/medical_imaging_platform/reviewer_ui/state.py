"""Safe Streamlit session-state handling."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

DEFAULT_STATE: dict[str, Any] = {
    "last_api_request": None,
    "last_response": None,
    "selected_evidence_type": "longitudinal",
    "review_decision": "insufficient_information",
    "review_notes": "",
    "export_status": None,
}


def initialise_state(state: MutableMapping[str, Any]) -> None:
    for key, value in DEFAULT_STATE.items():
        state.setdefault(key, value)


def reset_state(state: MutableMapping[str, Any]) -> None:
    for key, value in DEFAULT_STATE.items():
        state[key] = value


def remember_response(
    state: MutableMapping[str, Any], request_summary: dict[str, Any], response: dict[str, Any]
) -> None:
    safe_request = {key: value for key, value in request_summary.items() if key not in {"values"}}
    state["last_api_request"] = safe_request
    state["last_response"] = response
