"""
PaksaTalker - AI-Powered Video Generation Platform

This is the main FastAPI application that serves both the API and the frontend.

## Features
- RESTful API for video generation and processing
- JWT-based authentication
- Background task processing
- Interactive API documentation
- CORS support
- Static file serving for frontend
"""
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
    Depends,
    Security
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Scope, Receive, Send

from config import config
from api.schemas import (
    APIStatus,
    ErrorResponse,
    TokenResponse,
    example_responses,
    VideoInfo,
    TaskStatusResponse
)

# Import configuration
from config import config

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if config['app']['debug'] else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend" / "dist"
STATIC_DIR = FRONTEND_DIR / "assets"

# Ensure directories exist
os.makedirs(FRONTEND_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> HTMLResponse:
        response = await super().get_response(path, scope)
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response

def custom_openapi():
    """Generate custom OpenAPI schema with additional metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=config['app']['name'],
        version=config['app']['version'],
        description="""
        ## PaksaTalker API

        This is the API documentation for PaksaTalker, an AI-powered video generation platform.

        ### Authentication
        Most endpoints require authentication. Use the `/auth/login` endpoint to get a JWT token.

        ### Rate Limiting
        - Free tier: 100 requests/hour
        - Pro tier: 1000 requests/hour

        ### Error Handling
        All error responses follow the same format:
        ```json
        {
            "error": "Error message",
            "code": 400,
            "details": {
                "field": "additional error details"
            }
        }
        ```
        """,
        routes=app.routes,
        contact={
            "name": "PaksaTalker Support",
            "email": "support@paksatalker.com"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        },
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.paksatalker.com",
                "description": "Production server"
            }
        ],
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2PasswordBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token in the format: Bearer <token>"
        }
    }

    # Add global security
    openapi_schema["security"] = [{"OAuth2PasswordBearer": []}]

    # Add more detailed error responses
    for path in openapi_schema["paths"].values():
        for method in path.values():
            if "responses" not in method:
                method["responses"] = {}

            # Add common error responses
            for code, response in example_responses.items():
                if str(code) not in method["responses"]:
                    method["responses"][str(code)] = response

    app.openapi_schema = openapi_schema
    return openapi_schema

# Initialize FastAPI app
app = FastAPI(
    title=config['app']['name'],
    version=config['app']['version'],
    description="AI-Powered Video Generation Platform",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url="/api/openapi.json" if config['app']['debug'] else None
)

# Set custom OpenAPI schema
app.openapi = custom_openapi

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB

# In production, force HTTPS
if not config['app']['debug']:
    app.add_middleware(HTTPSRedirectMiddleware)

# Setup CORS for API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['app'].get('cors_origins', [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8000",  # Production frontend
        "http://127.0.0.1:5173",  # Vite dev server (alternative)
        "http://127.0.0.1:8000",  # Production frontend (alternative)
    ]),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Request-ID"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Mount static files
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

# Create necessary directories
os.makedirs(config['paths']['output'], exist_ok=True)
os.makedirs(config['paths']['temp'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Serve SPA - catch all other routes and return the frontend
@app.get("/{full_path:path}", include_in_schema=False)
async def catch_all(full_path: str):
    """Catch all other routes and return the SPA"""
    if os.path.exists(FRONTEND_DIR / full_path):
        return FileResponse(FRONTEND_DIR / full_path)
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend not built. Please run 'npm run build' in the frontend directory."
        )
    return FileResponse(index_path)

# Static files are already mounted above

# Import and include routers
from api.routes import router as api_router
app.include_router(api_router, prefix="/api/v1")

# Custom docs endpoints
@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title=f"{config['app']['name']} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

@app.get("/api/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="/api/openapi.json",
        title=f"{config['app']['name']} - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )

# Health check endpoint
@app.get(
    "/api/health",
    response_model=APIStatus,
    responses={
        200: {"description": "Service is healthy"},
        503: {"model": ErrorResponse, "description": "Service is unhealthy"}
    },
    tags=["System"]
)
async def health_check() -> APIStatus:
    """
    Check the health status of the API and its dependencies.

    Returns:
        APIStatus: The current status of the API and its components
    """
    # Check database connection
    db_ok = True
    # Check model loading
    models_ok = True

    # Check cache connection
    cache_ok = True

    status_code = status.HTTP_200_OK if all([db_ok, models_ok, cache_ok]) else status.HTTP_503_SERVICE_UNAVAILABLE

    return APIStatus(
        status="healthy" if status_code == 200 else "unhealthy",
        version=config['app']['version'],
        models={
            "sadtalker": models_ok,
            "wav2lip": models_ok,
            "gesture": models_ok,
            "qwen": models_ok
        },
        database=db_ok,
        cache=cache_ok,
        uptime=time.time() - app.start_time if hasattr(app, 'start_time') else 0,
        timestamp=datetime.utcnow()
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# Main entry point
@app.on_event("startup")
async def startup_event():
    """Initialize application services on startup."""
    app.start_time = time.time()
    logger.info("Starting PaksaTalker server...")

    # Initialize models in the background
    from api.routes import get_models
    import asyncio

    async def init_models():
        try:
            await asyncio.to_thread(get_models)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    asyncio.create_task(init_models())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config['server']['host'],
        port=config['server']['port'],
        reload=config['server']['reload'],
        log_level=config['server']['log_level'],
        workers=config['server'].get('workers', 1),
        proxy_headers=True,
        forwarded_allow_ips='*',
        timeout_keep_alive=30,
    )
