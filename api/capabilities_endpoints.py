"""
Capabilities endpoint to inform frontend which features/models are available.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Any
from pathlib import Path
import asyncio
import subprocess
import importlib

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


class CapabilityModels(BaseModel):
    sadtalker: bool = Field(True)
    wav2lip2: bool = Field(False)
    emage: bool = Field(False)
    qwen: bool = Field(False)
    mediapipe: bool = Field(False)
    threeddfa: bool = Field(False)
    openseeface: bool = Field(False)
    mini_xception: bool = Field(False)
    sadtalker_weights: bool = Field(False)


class CapabilityFFmpeg(BaseModel):
    available: bool
    filters: Dict[str, bool]


class CapabilitiesResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any]


def _emage_available() -> bool:
    try:
        import os
        # Prefer env override to avoid false negatives
        override = os.environ.get('PAKSA_EMAGE_ROOT') or os.environ.get('EMAGE_ROOT')
        repo = Path(override) if override else Path('EMAGE')
        if not repo.exists():
            return False
        models_dir = repo / 'models'
        checkpoints_dir = repo / 'checkpoints'
        has_code = (models_dir / 'gesture_decoder.py').exists() or (models_dir / 'gesture_decoder/__init__.py').exists()
        has_ckpt = (checkpoints_dir / 'emage_best.pth').exists()
        return has_code and has_ckpt
    except Exception:
        return False


def _wav2lip2_available() -> bool:
    try:
        importlib.import_module('models.wav2lip2_aoti')
        weights = Path('wav2lip2-aoti') / 'weights' / 'wav2lip2_fp8.pt'
        return weights.exists()
    except Exception:
        return False


def _sadtalker_available() -> bool:
    try:
        importlib.import_module('models.sadtalker_full')
        return True
    except Exception:
        return False


def _sadtalker_weights_available() -> bool:
    try:
        import os
        from os import getenv
        p = getenv('SADTALKER_WEIGHTS')
        if p and Path(p).exists():
            return True
        ck = Path('models') / 'sadtalker' / 'checkpoints'
        candidates = [
            ck / 'epoch_20.pth',
            ck / 'facevid2vid_00189-model.pth.tar',
            ck / 'mapping_00109-model.pth.tar'
        ]
        return any(c.exists() for c in candidates)
    except Exception:
        return False


def _qwen_available() -> bool:
    try:
        importlib.import_module('models.qwen_omni')
        return True
    except Exception:
        return False


def _ffmpeg_filters() -> CapabilityFFmpeg:
    try:
        proc = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True, timeout=10)
        ok = proc.returncode == 0
        text = proc.stdout if ok else ''
    except Exception:
        ok = False
        text = ''
    names = {
        'lenscorrection', 'colortemperature', 'chromashift', 'tmix', 'gblur',
        'vignette', 'unsharp', 'hqdn3d', 'chromakey', 'overlay', 'scale', 'fps'
    }
    filters = {n: (n in text) for n in names} if ok else {n: False for n in names}
    return CapabilityFFmpeg(available=ok, filters=filters)


@router.get("")
async def get_capabilities() -> CapabilitiesResponse:
    # Kick asset ensure in background to reduce first-use errors/latency
    try:
        from models.emotion.model_loader import ensure_model_downloaded, ensure_emage_weights, ensure_openseeface_models
        async def _ensure():
            try:
                await asyncio.to_thread(ensure_model_downloaded)
            except Exception:
                pass
            try:
                await asyncio.to_thread(ensure_emage_weights)
            except Exception:
                pass
            try:
                await asyncio.to_thread(ensure_openseeface_models)
            except Exception:
                pass
        try:
            asyncio.create_task(_ensure())
        except Exception:
            pass
    except Exception:
        pass
    from models.expression.engine import detect_capabilities
    expr_caps = detect_capabilities()
    models = CapabilityModels(
        sadtalker=_sadtalker_available(),
        wav2lip2=_wav2lip2_available(),
        emage=_emage_available(),
        qwen=_qwen_available(),
        mediapipe=expr_caps.get('mediapipe', False),
        threeddfa=expr_caps.get('threeddfa', False),
        openseeface=expr_caps.get('openseeface', False),
        mini_xception=expr_caps.get('mini_xception', False),
        sadtalker_weights=_sadtalker_weights_available(),
    )
    ffmpeg = _ffmpeg_filters()
    data = {
        'models': models.dict(),
        'ffmpeg': ffmpeg.dict(),
    }
    return CapabilitiesResponse(success=True, data=data)
