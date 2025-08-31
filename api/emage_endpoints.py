"""
EMAGE validation endpoints
"""
from fastapi import APIRouter
from typing import Dict, Any
from pathlib import Path

router = APIRouter(prefix="/emage", tags=["emage"])


@router.get("/status")
async def emage_status() -> Dict[str, Any]:
    repo = Path('EMAGE')
    available = repo.exists()
    details: Dict[str, Any] = {"present": available}
    if available:
        models_dir = repo / 'models'
        checkpoints_dir = repo / 'checkpoints'
        details.update({
            "models_dir": str(models_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "gesture_decoder": (models_dir / 'gesture_decoder.py').exists() or (models_dir / 'gesture_decoder' / '__init__.py').exists(),
            "weights_present": (checkpoints_dir / 'emage_best.pth').exists()
        })
    return {"success": True, "emage": details}

