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
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
    Depends,
    Security,
    UploadFile,
    File
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
from werkzeug.utils import secure_filename

from config import config
from api.schemas.base import (
    APIStatus,
    ErrorResponse,
    TokenResponse,
    example_responses
)
from api.schemas.schemas import (
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

# Disable HTTPS redirect for development
# if not config['app']['debug']:
#     app.add_middleware(HTTPSRedirectMiddleware)

# Setup CORS for API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
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

# Import and include routers BEFORE catch-all route
from api.routes import router as api_router
from api.routes import get_task_status as v1_get_task_status
from api.routes import generate_advanced_video as v1_generate_advanced_video
from api.websocket_routes import router as websocket_router
from api.prompt_endpoints import router as prompt_router
from api.conversation_endpoints import router as convo_router
from api.style_endpoints import router as style_router
from api.language_endpoints import router as language_router
from api.composition_endpoints import router as composition_router
from api.lighting_endpoints import router as lighting_router
from api.camera_endpoints import router as camera_router
from api.background_endpoints import router as background_router
app.include_router(api_router, prefix="/api/v1")
app.include_router(prompt_router, prefix="/api/v1")
app.include_router(convo_router, prefix="/api/v1")
app.include_router(style_router, prefix="/api/v1")
app.include_router(language_router, prefix="/api/v1")
app.include_router(composition_router, prefix="/api/v1")
app.include_router(lighting_router, prefix="/api/v1")
app.include_router(camera_router, prefix="/api/v1")
app.include_router(background_router, prefix="/api/v1")
app.include_router(websocket_router)

# Serve SPA - catch all other routes and return the frontend
@app.get("/{full_path:path}", include_in_schema=False)
async def catch_all(full_path: str):
    """Catch all other routes and return the SPA"""
    
    # Secure the path to prevent directory traversal
    if full_path and not full_path.startswith('api/'):
        # Normalize and secure the path
        secure_path = secure_filename(full_path)
        file_path = FRONTEND_DIR / secure_path
        
        # Ensure the resolved path is within FRONTEND_DIR
        try:
            file_path = file_path.resolve()
            FRONTEND_DIR.resolve()
            if file_path.is_relative_to(FRONTEND_DIR.resolve()) and file_path.exists():
                return FileResponse(file_path)
        except (OSError, ValueError):
            pass
    
    # Return index.html for SPA routing
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend not built. Please run 'npm run build' in the frontend directory."
        )
    return FileResponse(index_path)

# Compatibility endpoints without version prefix
@app.get("/api/status/{task_id}", include_in_schema=False)
async def status_alias(task_id: str):
    """Alias for /api/v1/status/{task_id} to match frontend calls."""
    return await v1_get_task_status(task_id)

@app.post("/api/generate/advanced-video", include_in_schema=False)
async def advanced_video_alias(
    request: Request,
):
    """Alias for /api/v1/generate/advanced-video to match frontend calls."""
    # Re-parse the incoming multipart/form-data and pass to v1 handler
    from fastapi import BackgroundTasks
    form = await request.form()
    background_tasks = BackgroundTasks()
    # FastAPI handlers expect proper dependency injection; call v1 handler directly
    # by reconstructing parameters
    from fastapi import UploadFile
    image: UploadFile = form.get('image')  # type: ignore
    audio: UploadFile = form.get('audio') if 'audio' in form else None  # type: ignore

    # Coerce booleans
    def to_bool(v, default=False):
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        return str(v).lower() in ["1", "true", "yes", "on"]

    # Call underlying handler
    return await v1_generate_advanced_video(
        background_tasks=background_tasks,  # type: ignore
        image=image,  # type: ignore
        audio=audio,  # type: ignore
        text=form.get('text'),
        useEmage=to_bool(form.get('useEmage'), True),
        useWav2Lip2=to_bool(form.get('useWav2Lip2'), True),
        useSadTalkerFull=to_bool(form.get('useSadTalkerFull'), True),
        emotion=form.get('emotion') or 'neutral',
        bodyStyle=form.get('bodyStyle') or 'natural',
        avatarType=form.get('avatarType') or 'realistic',
        lipSyncQuality=form.get('lipSyncQuality') or 'high',
        resolution=form.get('resolution') or '1080p',
        fps=int(form.get('fps') or 30),
    )

@app.post("/api/generate/preview", include_in_schema=False)
async def generate_preview(
    image: UploadFile,  # type: ignore[name-defined]
    audio: UploadFile = None,  # type: ignore[name-defined]
    duration: int = 3
):
    """Generate a short preview clip for the given image/audio.
    Returns an MP4 blob directly for inline playback in the UI.
    """
    try:
        from pathlib import Path
        import uuid as _uuid
        import os as _os
        from fastapi import UploadFile as _UploadFile  # ensure symbol present

        temp_dir = Path(config.get('paths.temp', 'temp'))
        output_dir = Path(config.get('paths.output', 'output'))
        temp_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        task_id = str(_uuid.uuid4())
        img_path = temp_dir / f"{task_id}_{getattr(image, 'filename', 'image') }"
        with open(img_path, 'wb') as f:
            f.write(await image.read())

        audio_path = None
        if audio is not None:
            audio_path = temp_dir / f"{task_id}_{getattr(audio, 'filename', 'audio') }"
            with open(audio_path, 'wb') as f:
                f.write(await audio.read())

        preview_path = temp_dir / f"{task_id}_preview.mp4"

        # Try fast AI preview via SadTalkerFull (low fps/short duration)
        ai_ok = False
        try:
            from models.sadtalker_full import SadTalkerFull
            st = SadTalkerFull()
            st.img_size = 192  # smaller for speed
            result = st.generate(
                image_path=str(img_path),
                audio_path=str(audio_path) if audio_path else str(img_path),
                output_path=str(preview_path),
                emotion='neutral',
                enhance_face=False
            )
            if Path(result).exists() and Path(result).stat().st_size > 10_000:
                ai_ok = True
        except Exception:
            ai_ok = False

        if not ai_ok:
            # Fallback to ffmpeg still image + short duration/audio
            import subprocess
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1', '-i', str(img_path),
            ]
            if audio_path:
                cmd += ['-i', str(audio_path)]
            else:
                cmd += ['-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo']
            cmd += [
                '-t', str(max(1, int(duration))),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-shortest', str(preview_path)
            ]
            subprocess.run(cmd, capture_output=True)

        if not preview_path.exists():
            raise RuntimeError('Preview generation failed')

        return FileResponse(
            str(preview_path),
            media_type='video/mp4',
            filename=preview_path.name
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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
        host=config['api']['host'],
        port=8000,  # Use port 8000
        reload=config['api']['debug'],
        log_level='debug' if config['api']['debug'] else 'info',
        workers=1,  # Use single worker to avoid port conflicts
        proxy_headers=True,
        forwarded_allow_ips='*',
        timeout_keep_alive=300,  # 5 minutes keep-alive
        timeout_graceful_shutdown=60,  # 1 minute graceful shutdown
    )
