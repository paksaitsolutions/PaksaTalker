"""
Conversational Abilities API
Multi-turn conversation with context management, memory, and topic coherence
"""

from fastapi import APIRouter, Form, HTTPException
from typing import Dict, Any, List, Optional
import uuid
import time

from models.qwen_omni import get_qwen_model

try:
    from models.prompt_engine import prompt_engine, SafetyLevel
    HAS_PROMPT_ENGINE = True
except Exception:
    HAS_PROMPT_ENGINE = False
    prompt_engine = None  # type: ignore
    SafetyLevel = None  # type: ignore

router = APIRouter(prefix="/conversation", tags=["conversation"])


# In-memory conversation store (replace with DB/redis in prod)
_sessions: Dict[str, Dict[str, Any]] = {}


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: 4 chars per token (very approximate)
    return max(1, int(len(text) / 4))


def _trim_history(messages: List[Dict[str, str]], max_tokens: int, max_messages: int) -> List[Dict[str, str]]:
    # First trim by messages
    trimmed = messages[-max_messages:]
    # Then trim by tokens budget
    total = 0
    reversed_msgs = list(reversed(trimmed))
    kept: List[Dict[str, str]] = []
    for m in reversed_msgs:
        t = _estimate_tokens(m.get("content", ""))
        if total + t > max_tokens:
            break
        kept.append(m)
        total += t
    return list(reversed(kept))


def _extract_topic_seed(text: str) -> str:
    # Very simple topic seed: first 8 words
    words = text.strip().split()
    return " ".join(words[:8]) if words else "general"


@router.post("/start")
async def start_conversation(
    persona: str = Form("professional"),
    safety_level: str = Form("moderate"),
    max_messages: int = Form(12),
    max_context_tokens: int = Form(2000)
):
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "created_at": time.time(),
        "persona": persona,
        "safety_level": safety_level,
        "max_messages": max(4, min(max_messages, 64)),
        "max_context_tokens": max(500, min(max_context_tokens, 8000)),
        "topic": None,
        "messages": []  # list of {role, content}
    }
    return {"success": True, "session_id": session_id}


@router.post("/message")
async def send_message(
    session_id: str = Form(...),
    text: str = Form(""),
    image_b64: Optional[str] = Form(None),
    audio_b64: Optional[str] = Form(None)
):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = _sessions[session_id]

    # Safety gate for user text
    if HAS_PROMPT_ENGINE and text:
        sl = SafetyLevel(sess["safety_level"]) if isinstance(sess.get("safety_level"), str) else sess.get("safety_level")
        if not prompt_engine._passes_safety_filter(text, sl):
            moderated = prompt_engine.apply_safety_moderation(text, sl)
            return {
                "success": True,
                "role": "assistant",
                "response": moderated,
                "message": "Safety moderation applied",
                "session_id": session_id
            }

    # Establish topic on first user turn
    if not sess.get("topic") and text:
        sess["topic"] = _extract_topic_seed(text)

    # Append user message
    if text:
        sess["messages"].append({"role": "user", "content": text})

    # Build context with window management
    history = _trim_history(sess["messages"], sess["max_context_tokens"], sess["max_messages"])

    # Add a lightweight system header for persona + topic coherence
    system_parts = [f"Persona: {sess['persona']}"]
    if sess.get("topic"):
        system_parts.append(f"Topic: {sess['topic']}")
        system_parts.append("Maintain coherence with prior discussion unless user explicitly changes topic.")
    system_prompt = " | ".join(system_parts)
    history_with_system = [{"role": "system", "content": system_prompt}] + history

    # Route to model
    model = get_qwen_model()
    result = model.multimodal_chat(
        text=text or None,
        image=image_b64,
        audio=audio_b64,
        context=history_with_system
    )

    response_text = result.get("response", "")

    # Follow-up guidance: if user asked a question, the assistant can suggest a next step
    if text and text.strip().endswith(("?", ":")):
        response_text += "\n\nWould you like me to elaborate or provide examples?"

    # Store assistant turn
    sess["messages"].append({"role": "assistant", "content": response_text})

    return {
        "success": True,
        "session_id": session_id,
        "response": response_text,
        "conversation": history_with_system + [{"role": "assistant", "content": response_text}],
        "limits": {
            "max_messages": sess["max_messages"],
            "max_context_tokens": sess["max_context_tokens"]
        }
    }


@router.get("/{session_id}")
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = _sessions[session_id]
    return {
        "success": True,
        "session_id": session_id,
        "persona": s["persona"],
        "safety_level": s["safety_level"],
        "topic": s.get("topic"),
        "messages": s["messages"],
        "limits": {"max_messages": s["max_messages"], "max_context_tokens": s["max_context_tokens"]}
    }


@router.post("/reset")
async def reset_session(session_id: str = Form(...)):
    if session_id in _sessions:
        _sessions[session_id]["messages"] = []
        _sessions[session_id]["topic"] = None
    return {"success": True, "session_id": session_id}


@router.post("/config")
async def update_config(
    session_id: str = Form(...),
    max_messages: Optional[int] = Form(None),
    max_context_tokens: Optional[int] = Form(None),
    persona: Optional[str] = Form(None),
    safety_level: Optional[str] = Form(None)
):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = _sessions[session_id]
    if max_messages is not None:
        s["max_messages"] = max(4, min(int(max_messages), 64))
    if max_context_tokens is not None:
        s["max_context_tokens"] = max(500, min(int(max_context_tokens), 8000))
    if persona is not None:
        s["persona"] = persona
    if safety_level is not None:
        s["safety_level"] = safety_level
    return {"success": True, "session_id": session_id, "limits": {"max_messages": s["max_messages"], "max_context_tokens": s["max_context_tokens"]}}

