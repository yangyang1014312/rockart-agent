"""LangGraph orchestration for the rock art analysis agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    END = "__end__"
    StateGraph = None
    LANGGRAPH_IMPORT_ERROR = exc
else:
    LANGGRAPH_IMPORT_ERROR = None

try:
    from langgraph.types import interrupt
except ImportError:  # pragma: no cover - depends on LangGraph version
    interrupt = None

from rockart_agent.llm import (
    LLMNotConfiguredError,
    generate_detection_analysis,
    generate_direct_answer,
    generate_memory_answer,
    llm_is_configured,
)
from rockart_agent.memory import DEFAULT_MEMORY_PATH, RockArtMemory
from rockart_agent.state import RockArtAgentState, add_error, add_tool_call, add_trace, with_defaults
from rockart_agent.tools import DEFAULT_API_URL, analyze_rock_art_image


MEMORY_KEYWORDS = {
    "刚才",
    "上次",
    "上一次",
    "最近",
    "历史",
    "之前",
    "记忆",
    "记录",
    "追问",
    "总结",
    "比较",
    "last",
    "previous",
    "recent",
    "history",
    "memory",
    "remember",
}

IMAGE_TOOL_KEYWORDS = {
    "分析",
    "检测",
    "识别",
    "图像",
    "图片",
    "这张",
    "岩画",
    "bbox",
    "mask",
    "置信度",
    "阈值",
    "重新",
    "analyze",
    "detect",
    "identify",
    "image",
    "picture",
}

DIRECT_KEYWORDS = {
    "怎么用",
    "如何",
    "说明",
    "流程",
    "能力",
    "帮助",
    "什么时候",
    "什么情况下",
    "help",
    "what can you do",
    "how to",
    "when",
}


def validate_input(state: RockArtAgentState) -> RockArtAgentState:
    state = with_defaults(state)
    return {
        **state,
        "api_url": state.get("api_url") or DEFAULT_API_URL,
        "memory_path": state.get("memory_path") or DEFAULT_MEMORY_PATH,
        "decision_trace": add_trace(state, "validate_input: initialized defaults and runtime config."),
        "next_step": "load_memory",
    }


def load_memory(state: RockArtAgentState) -> RockArtAgentState:
    memory = RockArtMemory(state.get("memory_path") or DEFAULT_MEMORY_PATH)
    memory_context = memory.load_recent(limit=5)
    return {
        **state,
        "memory_context": memory_context,
        "memory_loaded_count": len(memory_context),
        "decision_trace": add_trace(state, f"load_memory: loaded {len(memory_context)} recent case(s)."),
        "next_step": "decide_intent",
    }


def decide_intent(state: RockArtAgentState) -> RockArtAgentState:
    query = state.get("user_query", "").strip()
    query_lower = query.lower()
    image_path = state.get("image_path")
    has_image = bool(image_path)

    if has_image and looks_like_image_tool_request(query_lower):
        intent = "image_analysis"
        next_step = "detect_instances"
        reason = "image path is present and the query asks for image/detection analysis."
    elif has_image and not looks_like_memory_query(query_lower):
        intent = "image_analysis"
        next_step = "detect_instances"
        reason = "image path is present, so defaulting to image analysis."
    elif looks_like_memory_query(query_lower):
        intent = "memory_query"
        next_step = "answer_from_memory"
        reason = "query refers to recent, previous, or remembered results."
    elif looks_like_direct_question(query_lower):
        intent = "direct_answer"
        next_step = "direct_answer"
        reason = "query asks about usage or capabilities, no vision tool needed."
    elif looks_like_image_tool_request(query_lower):
        intent = "image_analysis"
        next_step = "handle_error"
        reason = "query asks for image analysis but no image path was provided."
    elif has_image:
        intent = "image_analysis"
        next_step = "detect_instances"
        reason = "fallback with image path: run the vision tool."
    else:
        intent = "direct_answer"
        next_step = "direct_answer"
        reason = "no image path and no memory cue, answer directly."

    return {
        **state,
        "intent": intent,
        "errors": add_error(state, "Image analysis requires an image path.") if next_step == "handle_error" else state.get("errors", []),
        "decision_trace": add_trace(state, f"decide_intent: {intent}; {reason}"),
        "next_step": next_step,
    }


def detect_instances(state: RockArtAgentState) -> RockArtAgentState:
    image_path = state.get("image_path")
    if not image_path:
        return {
            **state,
            "errors": add_error(state, "Image analysis requires image_path."),
            "decision_trace": add_trace(state, "detect_instances: blocked because image_path is missing."),
            "next_step": "handle_error",
        }
    if not Path(image_path).is_file():
        return {
            **state,
            "errors": add_error(state, f"Image not found: {image_path}"),
            "decision_trace": add_trace(state, f"detect_instances: image not found: {image_path}"),
            "next_step": "handle_error",
        }

    tool_args = {
        "image_path": image_path,
        "api_url": state.get("api_url") or DEFAULT_API_URL,
        "score_thr": float(state.get("score_thr", 0.3)),
        "include_masks": bool(state.get("include_masks", False)),
    }
    state_with_call = {**state, "tool_calls": add_tool_call(state, "analyze_rock_art_image", tool_args)}

    try:
        result = analyze_rock_art_image(**tool_args)
    except Exception as exc:
        retry_count = int(state.get("retry_count", 0)) + 1
        return {
            **state_with_call,
            "retry_count": retry_count,
            "errors": add_error(state, f"Detection API failed on attempt {retry_count}: {exc}"),
            "decision_trace": add_trace(state, f"detect_instances: tool failed on attempt {retry_count}."),
            "next_step": "retry_detection",
        }

    needs_review, reason = evaluate_review_need(state, result)
    return {
        **state_with_call,
        "detection_result": result,
        "needs_human_review": needs_review,
        "human_review_reason": reason,
        "decision_trace": add_trace(
            state,
            "detect_instances: tool succeeded; "
            + ("human review required." if needs_review else "routing to analysis."),
        ),
        "next_step": "human_review" if needs_review else "analyze_detection",
    }


def retry_detection(state: RockArtAgentState) -> RockArtAgentState:
    if int(state.get("retry_count", 0)) < int(state.get("max_retries", 3)):
        return {
            **state,
            "decision_trace": add_trace(state, "retry_detection: retries remain, running tool again."),
            "next_step": "detect_instances",
        }
    return {
        **state,
        "decision_trace": add_trace(state, "retry_detection: retry budget exhausted."),
        "next_step": "handle_error",
        "final_answer": "Detection failed after retrying. Please check the FastAPI service and model server logs.",
    }


def human_review(state: RockArtAgentState) -> RockArtAgentState:
    if not state.get("needs_human_review"):
        return {**state, "next_step": "analyze_detection"}

    prompt = {
        "message": "Human confirmation is needed before final analysis.",
        "reason": state.get("human_review_reason"),
        "image_path": state.get("image_path"),
        "detection_summary": (state.get("detection_result") or {}).get("summary"),
        "question": "Approve this detection result for downstream analysis?",
    }

    feedback = state.get("human_feedback")
    if feedback is None and interrupt is not None:
        feedback = interrupt(prompt)

    if feedback is None:
        return {
            **state,
            "decision_trace": add_trace(state, "human_review: no approval available, stopping before analysis."),
            "final_answer": (
                "Human confirmation required: "
                f"{state.get('human_review_reason') or 'review requested'}"
            ),
            "next_step": "final_response",
        }

    approved = parse_human_approval(feedback)
    return {
        **state,
        "human_feedback": feedback,
        "human_approved": approved,
        "decision_trace": add_trace(
            state,
            "human_review: approved." if approved else "human_review: rejected.",
        ),
        "next_step": "analyze_detection" if approved else "final_response",
        "final_answer": None if approved else "Analysis stopped because human review did not approve the result.",
    }


def analyze_detection(state: RockArtAgentState) -> RockArtAgentState:
    result = state.get("detection_result") or {}
    class_counts = result.get("class_counts", {})
    memory_context = state.get("memory_context", [])

    if not result or result.get("status") != "success":
        return {
            **state,
            "errors": add_error(state, "No successful detection result is available for analysis."),
            "decision_trace": add_trace(state, "analyze_detection: missing successful detection result."),
            "next_step": "handle_error",
        }

    if should_use_llm(state):
        try:
            analysis = generate_detection_analysis(
                user_query=state.get("user_query", ""),
                detection_result=result,
                memory_context=memory_context,
                model=state.get("llm_model"),
                base_url=state.get("llm_base_url"),
                temperature=float(state.get("llm_temperature", 0.2)),
            )
            if analysis:
                return {
                    **state,
                    "memory_used": bool(memory_context),
                    "analysis": analysis,
                    "decision_trace": add_trace(state, "analyze_detection: generated answer with LLM."),
                    "next_step": "save_memory",
                }
        except LLMNotConfiguredError as exc:
            state = {**state, "errors": add_error(state, f"LLM is not configured: {exc}")}
        except Exception as exc:
            state = {**state, "errors": add_error(state, f"LLM analysis failed, using fallback: {exc}")}

    if not class_counts:
        analysis = "No confident rock art instances were detected. Consider lowering the threshold or requesting manual review."
    else:
        classes = ", ".join(f"{name} x{count}" for name, count in class_counts.items())
        memory_note = ""
        if memory_context:
            memory_note = f" Recent memory contains {len(memory_context)} prior case(s) for comparison."
        analysis = f"Detected rock art categories: {classes}. {result.get('summary', '')}{memory_note}"

    return {
        **state,
        "memory_used": bool(memory_context),
        "analysis": analysis,
        "decision_trace": add_trace(state, "analyze_detection: generated deterministic fallback answer."),
        "next_step": "save_memory",
    }


def answer_from_memory(state: RockArtAgentState) -> RockArtAgentState:
    memory_context = state.get("memory_context", [])
    if not memory_context:
        return {
            **state,
            "memory_used": False,
            "decision_trace": add_trace(state, "answer_from_memory: no memory records available."),
            "final_answer": "目前还没有可用的历史分析记录。请先分析一张岩画图像，再追问历史结果。",
            "next_step": "final_response",
        }

    if should_use_llm(state):
        try:
            answer = generate_memory_answer(
                user_query=state.get("user_query", ""),
                memory_context=memory_context,
                model=state.get("llm_model"),
                base_url=state.get("llm_base_url"),
                temperature=float(state.get("llm_temperature", 0.2)),
            )
            if answer:
                return {
                    **state,
                    "memory_used": True,
                    "analysis": answer,
                    "decision_trace": add_trace(state, "answer_from_memory: answered from memory with LLM."),
                    "next_step": "final_response",
                }
        except LLMNotConfiguredError as exc:
            state = {**state, "errors": add_error(state, f"LLM is not configured: {exc}")}
        except Exception as exc:
            state = {**state, "errors": add_error(state, f"Memory QA LLM failed, using fallback: {exc}")}

    answer = build_memory_fallback_answer(state)
    return {
        **state,
        "memory_used": True,
        "analysis": answer,
        "decision_trace": add_trace(state, "answer_from_memory: answered from memory with deterministic fallback."),
        "next_step": "final_response",
    }


def direct_answer(state: RockArtAgentState) -> RockArtAgentState:
    if should_use_llm(state):
        try:
            answer = generate_direct_answer(
                user_query=state.get("user_query", ""),
                model=state.get("llm_model"),
                base_url=state.get("llm_base_url"),
                temperature=float(state.get("llm_temperature", 0.2)),
            )
            if answer:
                return {
                    **state,
                    "analysis": answer,
                    "decision_trace": add_trace(state, "direct_answer: answered with LLM, no tool call."),
                    "final_answer": answer,
                    "next_step": "final_response",
                }
        except LLMNotConfiguredError as exc:
            state = {**state, "errors": add_error(state, f"LLM is not configured: {exc}")}
        except Exception as exc:
            state = {**state, "errors": add_error(state, f"Direct LLM answer failed, using fallback: {exc}")}

    answer = (
        "我可以根据你的问题选择路径：如果你提供图像并要求分析/检测，我会调用实例分割 tool；"
        "如果你追问上次、最近或历史结果，我会只读取 memory；如果你问使用方式，我会直接回答。"
        "使用 `--json` 可以查看 intent、tool_calls、memory_used 和 decision_trace。"
    )
    return {
        **state,
        "decision_trace": add_trace(state, "direct_answer: answered without tool call."),
        "final_answer": answer,
        "next_step": "final_response",
    }


def should_use_llm(state: RockArtAgentState) -> bool:
    if bool(state.get("use_llm", False)):
        return True
    return llm_is_configured(model=state.get("llm_model"))


def save_memory(state: RockArtAgentState) -> RockArtAgentState:
    result = state.get("detection_result") or {}
    if result.get("status") == "success":
        memory = RockArtMemory(state.get("memory_path") or DEFAULT_MEMORY_PATH)
        memory.save_case(
            image_path=state.get("image_path", ""),
            user_query=state.get("user_query", ""),
            detection_result=result,
            analysis=state.get("analysis") or "",
            human_feedback=state.get("human_feedback"),
        )
        trace = add_trace(state, "save_memory: saved detection case.")
    else:
        trace = add_trace(state, "save_memory: skipped because there is no successful detection result.")
    return {**state, "decision_trace": trace, "next_step": "final_response"}


def final_response(state: RockArtAgentState) -> RockArtAgentState:
    if state.get("final_answer"):
        return state

    result = state.get("detection_result") or {}
    final_answer = state.get("analysis") or result.get("summary") or "Analysis complete."
    return {**state, "final_answer": final_answer}


def handle_error(state: RockArtAgentState) -> RockArtAgentState:
    errors = state.get("errors", [])
    message = errors[-1] if errors else "Unknown agent error."
    return {
        **state,
        "decision_trace": add_trace(state, f"handle_error: {message}"),
        "final_answer": message,
    }


def evaluate_review_need(
    state: RockArtAgentState,
    result: dict[str, Any],
) -> tuple[bool, str | None]:
    if result.get("status") != "success":
        return True, "Detection did not return success."
    if int(result.get("count", 0)) == 0:
        return True, "No instances were detected."
    if float(result.get("max_score", 0.0)) < float(state.get("low_confidence_thr", 0.45)):
        return True, "Top detection confidence is below the human review threshold."
    if int(result.get("count", 0)) > int(state.get("max_detections_before_review", 50)):
        return True, "Detection count is unusually high."
    return False, None


def parse_human_approval(feedback: str | dict[str, Any]) -> bool:
    if isinstance(feedback, dict):
        value = feedback.get("approved", feedback.get("approve", feedback.get("ok", False)))
        return bool(value)
    return feedback.strip().lower() in {"y", "yes", "true", "approve", "approved", "ok", "通过", "确认"}


def looks_like_memory_query(query_lower: str) -> bool:
    return any(keyword in query_lower for keyword in MEMORY_KEYWORDS)


def looks_like_image_tool_request(query_lower: str) -> bool:
    return any(keyword in query_lower for keyword in IMAGE_TOOL_KEYWORDS)


def looks_like_direct_question(query_lower: str) -> bool:
    return any(keyword in query_lower for keyword in DIRECT_KEYWORDS)


def build_memory_fallback_answer(state: RockArtAgentState) -> str:
    memories = state.get("memory_context", [])
    if not memories:
        return "目前还没有可用的历史分析记录。"

    latest = memories[0]
    class_counts = latest.get("class_counts", {})
    class_text = ", ".join(f"{name} x{count}" for name, count in class_counts.items()) or "无明确类别"
    return (
        f"最近一次记录来自 {latest.get('image_path')}，检测类别为：{class_text}。"
        f"摘要：{latest.get('summary', '')}"
    )


def route_next(state: RockArtAgentState) -> str:
    return state.get("next_step") or "handle_error"


def build_graph() -> Any:
    if StateGraph is None:
        raise RuntimeError(
            "LangGraph is not installed. Install dependencies with `pip install -r requirements-agent.txt`."
        ) from LANGGRAPH_IMPORT_ERROR

    graph = StateGraph(RockArtAgentState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("load_memory", load_memory)
    graph.add_node("decide_intent", decide_intent)
    graph.add_node("detect_instances", detect_instances)
    graph.add_node("retry_detection", retry_detection)
    graph.add_node("human_review", human_review)
    graph.add_node("analyze_detection", analyze_detection)
    graph.add_node("answer_from_memory", answer_from_memory)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("save_memory", save_memory)
    graph.add_node("final_response", final_response)
    graph.add_node("handle_error", handle_error)

    graph.set_entry_point("validate_input")
    graph.add_edge("validate_input", "load_memory")
    graph.add_edge("load_memory", "decide_intent")
    graph.add_conditional_edges(
        "decide_intent",
        route_next,
        {
            "detect_instances": "detect_instances",
            "answer_from_memory": "answer_from_memory",
            "direct_answer": "direct_answer",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "detect_instances",
        route_next,
        {
            "retry_detection": "retry_detection",
            "human_review": "human_review",
            "analyze_detection": "analyze_detection",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "retry_detection",
        route_next,
        {"detect_instances": "detect_instances", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "human_review",
        route_next,
        {"analyze_detection": "analyze_detection", "final_response": "final_response"},
    )
    graph.add_conditional_edges(
        "analyze_detection",
        route_next,
        {"save_memory": "save_memory", "handle_error": "handle_error"},
    )
    graph.add_edge("answer_from_memory", "final_response")
    graph.add_edge("direct_answer", "final_response")
    graph.add_edge("save_memory", "final_response")
    graph.add_edge("final_response", END)
    graph.add_edge("handle_error", END)
    return graph.compile()


def run_agent(initial_state: RockArtAgentState) -> RockArtAgentState:
    compiled_graph = build_graph()
    return compiled_graph.invoke(initial_state)
