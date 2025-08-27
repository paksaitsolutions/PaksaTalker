"""Pydantic models for API request/response schemas."""
from typing import List, Optional, Dict, Any, Union, Tuple, Literal
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl, conlist, confloat, conint
from datetime import datetime

# Import base schemas to avoid circular imports
from .base import APIStatus, ErrorResponse, TokenResponse, example_responses

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
    email: str = Field(..., pattern=r"^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
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
        pattern=r"^\d+x\d+$",
        description="Output resolution in WxH format"
    )

class TextToSpeechRequest(BaseModel):
    """Schema for text-to-speech request."""
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field("default", description="Voice ID to use for synthesis")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed (0.5-2.0)")

class SpeakerRegistrationRequest(BaseModel):
    """Schema for speaker registration request."""
    audio_file: str = Field(..., description="Base64 encoded audio file")
    speaker_id: str = Field(..., description="Unique identifier for the speaker")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata about the speaker"
    )

class SpeakerIdentificationRequest(BaseModel):
    """Schema for speaker identification request."""
    audio_file: str = Field(..., description="Base64 encoded audio file for identification")
    threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0-1) for positive identification"
    )

class ModelAdaptationRequest(BaseModel):
    """Schema for model adaptation request."""
    speaker_id: str = Field(..., description="ID of the speaker to adapt to")
    audio_dir: str = Field(..., description="Directory containing speaker's audio files")
    model_type: str = Field(
        "sadtalker",
        description="Type of model to adapt (sadtalker, wav2lip, etc.)"
    )
    epochs: int = Field(10, ge=1, description="Number of training epochs")
    learning_rate: float = Field(1e-4, gt=0, description="Learning rate for adaptation")
    batch_size: int = Field(4, ge=1, description="Batch size for training")


class AdaptationStatusResponse(BaseModel):
    """Schema for model adaptation status response."""
    task_id: str = Field(..., description="Unique identifier for the adaptation task")
    status: str = Field(..., description="Current status of the adaptation task")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress of the adaptation (0.0 to 1.0)")
    message: Optional[str] = Field(None, description="Additional status message")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the task was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the status was last updated")

class AnimationStyleBase(BaseModel):
    """Base schema for animation styles."""
    name: str = Field(..., description="Name of the animation style")
    description: str = Field(..., description="Description of the style")
    parameters: Dict[str, Any] = Field(
        ...,
        description="Animation parameters specific to this style"
    )

class AnimationStyleCreate(AnimationStyleBase):
    """Schema for creating a new animation style."""
    speaker_id: Optional[str] = Field(
        None,
        description="Speaker ID this style is associated with (if any)"
    )
    is_global: bool = Field(
        False,
        description="Whether this is a global style (available to all speakers)"
    )

class AnimationStyleUpdate(BaseModel):
    """Schema for updating an existing animation style."""
    name: Optional[str] = Field(None, description="Updated name of the style")
    description: Optional[str] = Field(None, description="Updated description")
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated animation parameters"
    )

class VoiceModelBase(BaseModel):
    """Base schema for voice models."""
    speaker_name: str = Field(..., description="Name of the speaker")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the voice model"
    )

class VoiceCreateRequest(VoiceModelBase):
    """Schema for creating a new voice model."""
    audio_files: conlist(Union[str, bytes]) = Field(
        ...,
        min_length=1,
        description="List of audio files or audio data to create the voice model from"
    )
    reference_text: Optional[str] = Field(
        None,
        description="Optional reference text for better voice cloning quality"
    )
    voice_id: Optional[str] = Field(
        None,
        description="Optional custom ID for the voice model"
    )

# Response Models
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

class SpeakerInfo(BaseModel):
    """Schema for speaker information response."""
    speaker_id: str
    created_at: datetime
    last_updated: datetime
    metadata: Optional[Dict[str, Any]] = None
    num_recordings: int = 0
    embedding_shape: Optional[tuple] = None

class AnimationStyle(AnimationStyleBase):
    """Schema for animation style responses."""
    style_id: str = Field(..., description="Unique identifier for the style")
    speaker_id: Optional[str] = Field(
        None,
        description="Speaker ID this style is associated with (if any)"
    )
    is_global: bool = Field(
        False,
        description="Whether this is a global style (available to all speakers)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the style was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the style was last updated"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "style_id": "style_123",
                "name": "Expressive",
                "description": "Exaggerated facial expressions and movements",
                "parameters": {
                    "intensity": 1.3,
                    "smoothness": 0.7,
                    "expressiveness": 1.0,
                    "motion_scale": 1.2,
                    "head_movement": 0.8,
                    "eye_blink_rate": 0.7,
                    "lip_sync_strength": 1.1
                },
                "is_global": True,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z"
            }
        }

class VoiceResponse(VoiceModelBase):
    """Schema for voice model responses."""
    voice_id: str = Field(..., description="Unique identifier for the voice model")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the voice model was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the voice model was last updated"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "voice_id": "voice_123",
                "speaker_name": "John Doe",
                "metadata": {
                    "language": "en-US",
                    "accent": "American"
                },
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z"
            }
        }

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
