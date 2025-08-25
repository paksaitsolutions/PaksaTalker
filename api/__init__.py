""PaksaTalker API package."""
from fastapi import APIRouter
from . import routes

# Create main API router
router = APIRouter()

# Include all route modules
router.include_router(routes.router)

__all__ = ['router']
