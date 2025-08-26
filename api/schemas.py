"""Pydantic models for API request/response schemas."""
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime

# Enums
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoFormat(str, Enum):
    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"

class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"

# Request Models
class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50, example="johndoe")
    email: str = Field(..., regex=r"^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    password: str = Field(..., min_length=8, example="securepassword123")
    full_name: Optional[str] = Field(None, example="John Doe")

class UserLogin(BaseModel):
    """Schema for user login."""
    email: str = Field(..., example="user@example.com")
    password: str = Field(..., example="yourpassword")

class VideoGenerationRequest(BaseModel):
    """Schema for video generation request."""
    text: Optional[str] = Field(
        None,
        description="Text to convert to speech (if audio is not provided)",
        example="Hello, this is a test message."
    )
    emotion: EmotionType = Field(
        EmotionType.NEUTRAL,
        description="Emotion to express in the generated video"
    )
    format: VideoFormat = Field(
        VideoFormat.MP4,
        description="Output video format"
    )
    resolution: str = Field(
        "512x512",
        regex=r"^\d+x\d+$",
        description="Output resolution in WxH format"
    )

class TextToSpeechRequest(BaseModel):
    """Schema for text-to-speech request."""
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field("default", description="Voice ID to use for synthesis")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed (0.5-2.0)")

# Response Models
class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str
    user: Dict[str, Any]

class TaskStatusResponse(BaseModel):
    """Background task status response."""
    task_id: str
    status: TaskStatus
    progress: Optional[float] = Field(None, ge=0, le=100)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class VideoInfo(BaseModel):
    """Video metadata information."""
    id: str
    filename: str
    size: int
    duration: float
    format: str
    created_at: datetime
    url: HttpUrl

class APIStatus(BaseModel):
    """API health and status information."""
    status: str
    version: str
    models: Dict[str, bool]
    database: bool
    cache: bool
    uptime: float
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    code: int
    details: Optional[Dict[str, Any]] = None

# Example responses
example_responses = {
    200: {"description": "Successful operation"},
    400: {"model": ErrorResponse, "description": "Bad request"},
    401: {"model": ErrorResponse, "description": "Unauthorized"},
    403: {"model": ErrorResponse, "description": "Forbidden"},
    404: {"model": ErrorResponse, "description": "Not found"},
    422: {"model": ErrorResponse, "description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}
