"""
Expression analysis endpoints: unify MediaPipe, 3DDFA, OpenSeeFace, and mini-XCEPTION.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import tempfile

from models.expression.engine import detect_capabilities, estimate_from_path

router = APIRouter(prefix="/expressions", tags=["expressions"])


@router.get("/capabilities")
async def expressions_capabilities() -> Dict[str, Any]:
    caps = detect_capabilities()
    return {"success": True, "engines": caps}


@router.post("/estimate")
async def estimate_expressions(
    image: UploadFile = File(...),
    engine: str = Form("auto")
):
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix='_' + (image.filename or 'img') ) as tf:
            tf.write(await image.read())
            tf.flush()
            result = estimate_from_path(tf.name, engine)
        return {"success": True, "engine": result.engine, "result": result.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

