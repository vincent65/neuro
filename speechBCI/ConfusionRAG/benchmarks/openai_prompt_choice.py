"""OpenAI prompt-choice helper for closed-set confusion candidate selection."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class PromptChoiceResult:
    """Structured result for one prompt-based candidate choice."""

    chosen_candidate: str
    confidence: Optional[float]
    used_fallback: bool
    fallback_reason: str
    raw_response_text: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_candidate": self.chosen_candidate,
            "confidence": self.confidence,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "raw_response_text": self.raw_response_text,
            "error": self.error,
        }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, out))


def _norm_text(value: str) -> str:
    out = value.strip().lower()
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r'^[\s"\'.,:;!?()\[\]{}-]+', "", out)
    out = re.sub(r'[\s"\'.,:;!?()\[\]{}-]+$', "", out)
    return out


def _default_candidate(candidates: Sequence[str], candidate_weights: Sequence[float]) -> str:
    if not candidates:
        return ""
    if candidate_weights and len(candidate_weights) == len(candidates):
        best_idx = max(range(len(candidates)), key=lambda i: float(candidate_weights[i]))
        return candidates[best_idx]
    return candidates[0]


def _extract_text_from_response(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output_items = getattr(response, "output", None)
    if not isinstance(output_items, list):
        return ""
    chunks: List[str] = []
    for item in output_items:
        contents = getattr(item, "content", None) or []
        for content in contents:
            if getattr(content, "type", "") == "output_text":
                chunk = getattr(content, "text", "")
                if isinstance(chunk, str) and chunk:
                    chunks.append(chunk)
    return "\n".join(chunks).strip()


def _extract_json_object(text: str) -> str:
    if not text:
        return ""
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
        clean = clean.strip()
    if clean.startswith("{") and clean.endswith("}"):
        return clean
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        return clean[start:end + 1]
    return ""


def _resolve_candidate_from_text(raw_text: str, candidates: Sequence[str]) -> str:
    if not raw_text:
        return ""
    norm_to_candidate: Dict[str, str] = {}
    for cand in candidates:
        norm = _norm_text(str(cand))
        if norm and norm not in norm_to_candidate:
            norm_to_candidate[norm] = str(cand)

    raw_norm = _norm_text(raw_text)
    if raw_norm in norm_to_candidate:
        return norm_to_candidate[raw_norm]

    for norm, cand in norm_to_candidate.items():
        if norm and re.search(rf"\b{re.escape(norm)}\b", raw_norm):
            return cand
    return ""


def _parse_choice_payload(raw_text: str, candidates: Sequence[str]) -> Dict[str, Any]:
    """
    Parse model output with robust fallbacks:
    1) strict JSON object
    2) extracted JSON substring
    3) free-text candidate mention
    """
    json_blob = _extract_json_object(raw_text)
    payload: Dict[str, Any] = {}
    if json_blob:
        try:
            parsed = json.loads(json_blob)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = {}

    chosen = payload.get("chosen_candidate")
    if not isinstance(chosen, str) or not chosen.strip():
        # Allow index-style outputs if model provides them.
        idx = payload.get("chosen_index")
        if isinstance(idx, int) and 0 <= idx < len(candidates):
            chosen = str(candidates[idx])
        elif isinstance(idx, str) and idx.isdigit():
            i = int(idx)
            if 0 <= i < len(candidates):
                chosen = str(candidates[i])

    resolved = _resolve_candidate_from_text(str(chosen) if chosen else "", candidates)
    if not resolved:
        resolved = _resolve_candidate_from_text(raw_text, candidates)

    return {
        "chosen_candidate": resolved,
        "confidence": _safe_float(payload.get("confidence")),
    }


def _build_messages(
    masked_sentence: str,
    candidates: Sequence[str],
    candidate_weights: Sequence[float],
    evidence_docs: Sequence[str],
) -> List[Dict[str, str]]:
    candidate_payload = []
    for idx, cand in enumerate(candidates):
        prior = float(candidate_weights[idx]) if idx < len(candidate_weights) else None
        candidate_payload.append({"candidate": cand, "prior_weight": prior})

    evidence_payload = [str(d) for d in evidence_docs if str(d).strip()]
    user_payload = {
        "task": "Choose the best candidate for the [MASK] token.",
        "constraints": [
            "Return strict JSON only.",
            "chosen_candidate must be exactly one of the listed candidates.",
            "Do not invent new words.",
        ],
        "masked_sentence": masked_sentence,
        "candidates": candidate_payload,
        "evidence_docs": evidence_payload,
        "output_schema": {
            "chosen_candidate": "string, required, must match one candidate exactly",
            "chosen_index": "integer index into candidates list, optional",
            "confidence": "number in [0,1], optional",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a constrained ASR post-decoding assistant. "
                "You must choose exactly one candidate from the provided set."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=True, indent=2),
        },
    ]


def choose_candidate_with_openai(
    masked_sentence: str,
    candidates: Sequence[str],
    candidate_weights: Optional[Sequence[float]] = None,
    evidence_docs: Optional[Sequence[str]] = None,
    model: str = "gpt-5",
    temperature: Optional[float] = None,
    max_output_tokens: int = 128,
    reasoning_effort: str = "minimal",
    timeout_s: float = 30.0,
    max_retries: int = 2,
    api_key: Optional[str] = None,
) -> PromptChoiceResult:
    """
    Choose one candidate using OpenAI Responses API with strict fallback safety.
    """
    cand_list = [str(c) for c in candidates]
    weight_list = [float(w) for w in (candidate_weights or [])]
    docs_list = [str(d) for d in (evidence_docs or [])]

    fallback = _default_candidate(cand_list, weight_list)
    if not cand_list:
        return PromptChoiceResult(
            chosen_candidate="",
            confidence=None,
            used_fallback=True,
            fallback_reason="no_candidates",
            raw_response_text="",
            error="No candidates supplied.",
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        return PromptChoiceResult(
            chosen_candidate=fallback,
            confidence=None,
            used_fallback=True,
            fallback_reason="openai_not_installed",
            raw_response_text="",
            error=str(exc),
        )

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        return PromptChoiceResult(
            chosen_candidate=fallback,
            confidence=None,
            used_fallback=True,
            fallback_reason="missing_api_key",
            raw_response_text="",
            error="OPENAI_API_KEY is not set.",
        )

    client = OpenAI(api_key=resolved_key, timeout=timeout_s)
    messages = _build_messages(masked_sentence, cand_list, weight_list, docs_list)

    last_error = ""
    last_text = ""
    current_max_output_tokens = max(32, int(max_output_tokens))
    for attempt in range(max(1, max_retries + 1)):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "input": messages,
                "max_output_tokens": current_max_output_tokens,
            }
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if reasoning_effort:
                request_kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.create(**request_kwargs)
            last_text = _extract_text_from_response(response)
            incomplete = getattr(response, "incomplete_details", None)
            incomplete_reason = getattr(incomplete, "reason", None) if incomplete else None
            if not last_text and incomplete_reason == "max_output_tokens":
                last_error = "Response truncated at max_output_tokens before final text."
                current_max_output_tokens = min(current_max_output_tokens * 2, 2048)
                if attempt < max_retries:
                    continue
            parsed = _parse_choice_payload(last_text, cand_list)
            chosen = parsed["chosen_candidate"]
            if isinstance(chosen, str) and chosen in cand_list:
                return PromptChoiceResult(
                    chosen_candidate=chosen,
                    confidence=parsed["confidence"],
                    used_fallback=False,
                    fallback_reason="",
                    raw_response_text=last_text,
                    error=None,
                )
            last_error = "Invalid or out-of-set chosen_candidate in model response."
        except Exception as exc:  # pragma: no cover - API exception shape varies
            last_error = str(exc)

        if attempt < max_retries:
            time.sleep(0.25 * (2 ** attempt))

    return PromptChoiceResult(
        chosen_candidate=fallback,
        confidence=None,
        used_fallback=True,
        fallback_reason="invalid_or_failed_response",
        raw_response_text=last_text,
        error=last_error or "Unknown OpenAI response error.",
    )
