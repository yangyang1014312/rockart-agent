"""Shared state schema for the LangGraph rock art agent."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


RouteName = Literal[
    "load_memory",
    "decide_intent",
    "detect_instances",
    "retry_detection",
    "human_review",
    "analyze_detection",
    "answer_from_memory",
    "direct_answer",
    "save_memory",
    "final_response",
    "handle_error",
]


class RockArtAgentState(TypedDict, total=False):
    image_path: str
    user_query: str
    api_url: str
    score_thr: float
    include_masks: bool
    low_confidence_thr: float
    max_retries: int
    max_detections_before_review: int
    retry_count: int
    detection_result: dict[str, Any] | None
    memory_context: list[dict[str, Any]]
    memory_loaded_count: int
    analysis: str | None
    final_answer: str | None
    errors: list[str]
    intent: str | None
    memory_used: bool
    tool_calls: list[dict[str, Any]]
    decision_trace: list[str]
    needs_human_review: bool
    human_review_reason: str | None
    human_feedback: str | dict[str, Any] | None
    human_approved: bool
    memory_path: str
    use_llm: bool
    llm_model: str | None
    llm_base_url: str | None
    llm_temperature: float
    next_step: str | None


DEFAULT_SCORE_THR = 0.3
DEFAULT_LOW_CONFIDENCE_THR = 0.45
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_DETECTIONS_BEFORE_REVIEW = 50


def with_defaults(state: RockArtAgentState) -> RockArtAgentState:
    next_state: RockArtAgentState = dict(state)
    next_state.setdefault("score_thr", DEFAULT_SCORE_THR)
    next_state.setdefault("include_masks", False)
    next_state.setdefault("low_confidence_thr", DEFAULT_LOW_CONFIDENCE_THR)
    next_state.setdefault("max_retries", DEFAULT_MAX_RETRIES)
    next_state.setdefault("max_detections_before_review", DEFAULT_MAX_DETECTIONS_BEFORE_REVIEW)
    next_state.setdefault("retry_count", 0)
    next_state.setdefault("detection_result", None)
    next_state.setdefault("memory_context", [])
    next_state.setdefault("memory_loaded_count", 0)
    next_state.setdefault("analysis", None)
    next_state.setdefault("final_answer", None)
    next_state.setdefault("errors", [])
    next_state.setdefault("intent", None)
    next_state.setdefault("memory_used", False)
    next_state.setdefault("tool_calls", [])
    next_state.setdefault("decision_trace", [])
    next_state.setdefault("needs_human_review", False)
    next_state.setdefault("human_review_reason", None)
    next_state.setdefault("human_feedback", None)
    next_state.setdefault("human_approved", False)
    next_state.setdefault("use_llm", False)
    next_state.setdefault("llm_model", None)
    next_state.setdefault("llm_base_url", None)
    next_state.setdefault("llm_temperature", 0.2)
    next_state.setdefault("next_step", None)
    return next_state


def add_error(state: RockArtAgentState, message: str) -> list[str]:
    return [*state.get("errors", []), message]


def add_trace(state: RockArtAgentState, message: str) -> list[str]:
    return [*state.get("decision_trace", []), message]


def add_tool_call(state: RockArtAgentState, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("tool_calls", []), {"name": name, "args": args}]
