"""Base schemas and shared types for the API."""
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


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


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str
    user: Dict[str, Any]


# Example responses
example_responses = {
    200: {"description": "Successful operation"},
    400: {"model": ErrorResponse, "description": "Bad request"},
    401: {"model": ErrorResponse, "description": "Unauthorized"},
    403: {"model": ErrorResponse, "description": "Forbidden"},
    404: {"model": ErrorResponse, "description": "Not found"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}
