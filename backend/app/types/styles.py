from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class StyleType(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    ENTHUSIASTIC = "enthusiastic"

class StyleSettings(BaseModel):
    styleType: StyleType
    intensity: int  # 1-10 scale
    culturalInfluence: Optional[str] = None
    mannerisms: Optional[List[str]] = None
