"""
Style and Emotion Adaptation Endpoints
"""

from fastapi import APIRouter, Form, HTTPException
from typing import Optional, Dict, Any

from models.prompt_engine import prompt_engine

router = APIRouter(prefix="/style", tags=["style-emotion"])


@router.post("/adapt-text")
async def adapt_text(
    text: str = Form(...),
    formality: str = Form("neutral"),  # casual | neutral | formal
    domain: Optional[str] = Form(None),
    personality: Optional[str] = Form(None),
    emotion: Optional[str] = Form(None)
):
    try:
        adapted = prompt_engine.adapt_text_style(
            text=text,
            formality=formality,
            domain=domain,
            personality=personality,
            emotion=emotion
        )
        embedding = prompt_engine.emotion_embedding(emotion or 'neutral')
        return {
            "success": True,
            "adapted_text": adapted,
            "emotion_embedding": embedding,
            "controls": {
                "formality": formality,
                "domain": domain,
                "personality": personality,
                "emotion": emotion or 'neutral'
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emotion-embedding")
async def get_emotion_embedding(emotion: str = Form("neutral")):
    try:
        embedding = prompt_engine.emotion_embedding(emotion)
        return {"success": True, "emotion": emotion, "embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

