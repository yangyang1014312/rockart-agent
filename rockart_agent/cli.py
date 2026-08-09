"""Command-line entrypoint for the LangGraph rock art agent."""

from __future__ import annotations

import argparse
import json

from rockart_agent.graph import run_agent
from rockart_agent.memory import DEFAULT_MEMORY_PATH
from rockart_agent.tools import DEFAULT_API_URL


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph rock art analysis agent.")
    parser.add_argument("image", nargs="?", default=None, help="Optional path to the image to analyze.")
    parser.add_argument("--query", default="", help="User question or task context.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the FastAPI service.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection confidence threshold.")
    parser.add_argument("--low-confidence-thr", type=float, default=0.45, help="Human review threshold.")
    parser.add_argument("--include-masks", action="store_true", help="Request COCO RLE masks.")
    parser.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH, help="SQLite memory path.")
    parser.add_argument("--human-approved", action="store_true", help="Pre-approve results that need review.")
    parser.add_argument("--use-llm", action="store_true", help="Use the configured LLM for final analysis.")
    parser.add_argument("--llm-model", default=None, help="LLM model name. Falls back to ROCKART_LLM_MODEL.")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--trace", action="store_true", help="Print a compact agent trace.")
    parser.add_argument("--json", action="store_true", help="Print full graph state.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    initial_state = {
        "user_query": args.query,
        "api_url": args.api_url,
        "score_thr": args.score_thr,
        "low_confidence_thr": args.low_confidence_thr,
        "include_masks": args.include_masks,
        "memory_path": args.memory_path,
        "human_feedback": {"approved": True} if args.human_approved else None,
        "use_llm": args.use_llm,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "llm_temperature": args.llm_temperature,
    }
    if args.image:
        initial_state["image_path"] = args.image

    state = run_agent(
        initial_state
    )

    if args.trace:
        print(json.dumps(build_trace_payload(state), ensure_ascii=False, indent=2))
    elif args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        print(state.get("final_answer", "No final answer produced."))


def build_trace_payload(state: dict) -> dict:
    return {
        "intent": state.get("intent"),
        "tool_calls": state.get("tool_calls", []),
        "memory_used": state.get("memory_used", False),
        "memory_loaded_count": state.get("memory_loaded_count", 0),
        "needs_human_review": state.get("needs_human_review", False),
        "human_review_reason": state.get("human_review_reason"),
        "decision_trace": state.get("decision_trace", []),
        "errors": state.get("errors", []),
        "final_answer": state.get("final_answer", "No final answer produced."),
    }


if __name__ == "__main__":
    main()
