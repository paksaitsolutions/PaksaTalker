"""PaksaTalker API schemas package."""
from .base import APIStatus, ErrorResponse, TokenResponse, example_responses

# Import models directly to avoid circular imports
from .schemas import (
    AnimationStyle,
    AnimationStyleCreate,
    AnimationStyleUpdate,
    VideoInfo,
    TaskStatusResponse
)

# Re-export all models
__all__ = [
    'APIStatus',
    'ErrorResponse',
    'TokenResponse',
    'example_responses',
    'AnimationStyle',
    'AnimationStyleCreate',
    'AnimationStyleUpdate',
    'VideoInfo',
    'TaskStatusResponse'
]
