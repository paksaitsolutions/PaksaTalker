from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel

# Define the StyleSettings model
class StyleSettings(BaseModel):
    styleType: str
    intensity: int
    culturalInfluence: str = "universal"
    mannerisms: List[str] = []

# Create router without prefix (prefix is already added in main.py)
router = APIRouter(tags=["styles"])

# In-memory storage for presets
PRESETS = {
    "professional": {
        "styleType": "professional",
        "intensity": 7,
        "culturalInfluence": "western",
        "mannerisms": ["formal", "concise"]
    },
    "casual": {
        "styleType": "casual",
        "intensity": 5,
        "culturalInfluence": "universal",
        "mannerisms": ["friendly", "relaxed"]
    }
}

@router.get("/style-presets", response_model=Dict[str, Any])
@router.get("/style/presets", response_model=Dict[str, Any], include_in_schema=False)
async def get_style_presets():
    """Get all available style presets
    
    This endpoint is available at both /api/v1/style-presets and /api/v1/style/presets
    for backward compatibility.
    """
    return PRESETS

@router.post("/presets")
async def save_style_preset(preset: StyleSettings):
    """Save a new style preset"""
    preset_name = preset.styleType.lower()
    PRESETS[preset_name] = preset.dict()
    return {"status": "success", "preset_name": preset_name}

@router.post("/transfer")
async def transfer_style(style_settings: StyleSettings):
    """Apply style transfer to content"""
    # This is a placeholder - implement actual style transfer logic here
    return {
        "status": "success",
        "message": "Style transfer initiated",
        "style_applied": style_settings.dict()
    }
