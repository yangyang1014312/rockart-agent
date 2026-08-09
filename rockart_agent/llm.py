"""Optional OpenAI-compatible LLM clients for the rock art agent."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency guard
    OpenAI = None


DEFAULT_LLM_TEMPERATURE = 0.2


class LLMNotConfiguredError(RuntimeError):
    pass


def llm_is_configured(api_key: str | None = None, model: str | None = None) -> bool:
    return bool(_resolve_api_key(api_key)) and bool(_resolve_model(model))


def generate_detection_analysis(
    user_query: str,
    detection_result: dict[str, Any],
    memory_context: list[dict[str, Any]] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
) -> str:
    """Generate an evidence-grounded natural language analysis from detection output."""

    prompt_payload = {
        "user_query": user_query,
        "detection_result": {
            "image_path": detection_result.get("image_path"),
            "score_thr": detection_result.get("score_thr"),
            "count": detection_result.get("count"),
            "class_counts": detection_result.get("class_counts"),
            "max_score": detection_result.get("max_score"),
            "detections": detection_result.get("detections", []),
            "summary": detection_result.get("summary"),
        },
        "memory_context": memory_context or [],
    }

    return _chat_completion(
        system_prompt=(
            "You are a cautious rock-art image analysis assistant. "
            "Use only the structured instance segmentation output and memory provided. "
            "Clearly separate model observations from interpretation. "
            "Do not claim cultural period, site identity, ritual meaning, or chronology "
            "unless the provided data explicitly supports it. Answer in Chinese."
        ),
        user_prompt=(
            "Answer the user's question based on the detection result below. "
            "Give a structured, verifiable analysis.\n\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        ),
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def generate_memory_answer(
    user_query: str,
    memory_context: list[dict[str, Any]],
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
) -> str:
    """Answer a follow-up question from stored memory without calling the vision tool."""

    return _chat_completion(
        system_prompt=(
            "You answer questions about prior rock-art analysis cases. "
            "Use only the provided memory records. If the memory is insufficient, say so. "
            "Do not invent detections or archaeological conclusions. Answer in Chinese."
        ),
        user_prompt=(
            "Answer the user follow-up using only these memory records.\n\n"
            f"User question: {user_query}\n\n"
            f"Memory records:\n{json.dumps(memory_context, ensure_ascii=False, indent=2)}"
        ),
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def generate_direct_answer(
    user_query: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
) -> str:
    """Answer a general user question without calling vision tools or memory."""

    return _chat_completion(
        system_prompt=(
            "You are the conversational layer of a rock-art analysis agent. "
            "For general questions, answer normally and helpfully. "
            "When relevant, explain that image analysis requires the instance segmentation tool, "
            "and memory follow-ups use stored prior analysis. Answer in Chinese."
        ),
        user_prompt=user_query,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def _chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
) -> str:
    if OpenAI is None:
        raise LLMNotConfiguredError("The `openai` package is not installed.")

    resolved_api_key = _resolve_api_key(api_key)
    resolved_model = _resolve_model(model)
    resolved_base_url = _resolve_base_url(base_url)

    if not resolved_api_key:
        raise LLMNotConfiguredError("No LLM API key is configured.")
    if not resolved_model:
        raise LLMNotConfiguredError("ROCKART_LLM_MODEL or OPENAI_MODEL is not configured.")

    client_kwargs = {"api_key": resolved_api_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=resolved_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    message = response.choices[0].message.content
    return message.strip() if message else ""


def _resolve_api_key(api_key: str | None = None) -> str | None:
    return (
        api_key
        or os.getenv("ROCKART_LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def _resolve_model(model: str | None = None) -> str | None:
    return model or os.getenv("ROCKART_LLM_MODEL") or os.getenv("OPENAI_MODEL")


def _resolve_base_url(base_url: str | None = None) -> str | None:
    resolved_base_url = base_url or os.getenv("ROCKART_LLM_BASE_URL") or None
    if resolved_base_url is None and os.getenv("DEEPSEEK_API_KEY"):
        resolved_base_url = "https://api.deepseek.com"
    return resolved_base_url
