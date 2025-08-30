"""
Capabilities endpoint to inform frontend which features/models are available.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Any
from pathlib import Path
import subprocess
import importlib

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


class CapabilityModels(BaseModel):
    sadtalker: bool = Field(True)
    wav2lip2: bool = Field(False)
    emage: bool = Field(False)
    qwen: bool = Field(False)


class CapabilityFFmpeg(BaseModel):
    available: bool
    filters: Dict[str, bool]


class CapabilitiesResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any]


def _emage_available() -> bool:
    try:
        importlib.import_module('EMAGE.models.gesture_decoder')
        return True
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
    models = CapabilityModels(
        sadtalker=_sadtalker_available(),
        wav2lip2=_wav2lip2_available(),
        emage=_emage_available(),
        qwen=_qwen_available(),
    )
    ffmpeg = _ffmpeg_filters()
    data = {
        'models': models.dict(),
        'ffmpeg': ffmpeg.dict(),
    }
    return CapabilitiesResponse(success=True, data=data)

