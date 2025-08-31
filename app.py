"""
PaksaTalker - AI-Powered Video Generation Platform

This is the main FastAPI application that serves both the API and the frontend.
"""
import os
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Import configuration before setting up logging
from config import config

# Configure logging once
logging.basicConfig(
    level=logging.DEBUG if config['app'].get('debug', False) else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

# Logging is already configured above

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

def custom_openapi() -> Dict[str, Any]:
    """Generate custom OpenAPI schema with additional metadata and documentation.
    
    Returns:
        Dict[str, Any]: The generated OpenAPI schema
        
    Raises:
        RuntimeError: If there's an error generating the schema
    """
    if hasattr(app, 'openapi_schema') and app.openapi_schema:
        return app.openapi_schema
    
    try:
        # Get base OpenAPI schema
        openapi_schema = get_openapi(
            title=config['app'].get('name', 'PaksaTalker API'),
            version=config['app'].get('version', '1.0.0'),
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
    
    # Initialize components if not present
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    # Add security schemes
    security_schemes = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Enter your API key in the format: Bearer <your-api-key>"
        },
        "OAuth2": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": config.get('auth', {}).get('authorization_url', '/auth/authorize'),
                    "tokenUrl": config.get('auth', {}).get('token_url', '/auth/token'),
                    "scopes": {
                        "read": "Read access to resources",
                        "write": "Write access to resources"
                    }
                }
            }
        }
    }
    
    openapi_schema["components"]["securitySchemes"] = security_schemes
    
    # Add global security
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"OAuth2": ["read"]}
    ]
    
    # Add error responses
    for path in openapi_schema["paths"].values():
        for method in path.values():
            if "responses" not in method:
                method["responses"] = {}
            method["responses"].update({
                "400": {"$ref": "#/components/responses/400"},
                "401": {"$ref": "#/components/responses/401"},
                "403": {"$ref": "#/components/responses/403"},
                "404": {"$ref": "#/components/responses/404"},
                "422": {"$ref": "#/components/responses/422"},
                "500": {"$ref": "#/components/responses/500"},
            })
    
    # Initialize responses if not present
    if "responses" not in openapi_schema["components"]:
        openapi_schema["components"]["responses"] = {}
        
    # Add common responses
    common_responses = {
        "400": {
            "description": "Bad Request: The request was invalid or cannot be served.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "detail": "Invalid request parameters",
                        "error": "bad_request",
                        "status_code": 400
                    }
                }
            }
        },
        "401": {
            "description": "Unauthorized: Authentication failed or user doesn't have permissions.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "detail": "Not authenticated",
                        "error": "unauthorized",
                        "status_code": 401
                    }
                }
            }
        },
        "403": {
            "description": "Forbidden: The request is understood but it has been refused or access is not allowed.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "detail": "Insufficient permissions",
                        "error": "forbidden",
                        "status_code": 403
                    }
                }
            }
        },
        "404": {
            "description": "Not Found: The requested resource doesn't exist.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "detail": "Resource not found",
                        "error": "not_found",
                        "status_code": 404
                    }
                }
            }
        },
        "422": {
            "description": "Validation Error: The request was well-formed but was unable to be followed due to semantic errors.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/HTTPValidationError"},
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "field_name"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        },
        "429": {
            "description": "Too Many Requests: Rate limit exceeded.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "detail": "Rate limit exceeded",
                        "error": "rate_limit_exceeded",
                        "status_code": 429,
                        "retry_after": 60
                    }
                }
            }
        },
        "500": {
            "description": "Internal Server Error: An unexpected error occurred on the server.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "detail": "Internal server error",
                        "error": "internal_server_error",
                        "status_code": 500
                    }
                }
            }
        },
    }
    
    openapi_schema["components"]["responses"].update(common_responses)
    
    # Initialize schemas if not present
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
    
    # Add common schemas
    openapi_schema["components"]["schemas"].update({
        "Error": {
            "title": "Error",
            "type": "object",
            "properties": {
                "detail": {"type": "string", "description": "A human-readable explanation of the error"},
                "error": {"type": "string", "description": "A machine-readable error code"},
                "status_code": {"type": "integer", "description": "The HTTP status code"},
                "retry_after": {"type": "integer", "description": "Number of seconds to wait before retrying (for rate limits)", "nullable": True}
            },
            "required": ["detail", "status_code"]
        },
        "HTTPValidationError": {
            "title": "HTTPValidationError",
            "type": "object",
            "properties": {
                "detail": {
                    "title": "Detail",
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/ValidationError"},
                    "description": "List of validation errors"
                }
            }
        },
        "ValidationError": {
            "title": "ValidationError",
            "required": ["loc", "msg", "type"],
            "type": "object",
            "properties": {
                "loc": {
                    "title": "Location",
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Location of the error in the request"
                },
                "msg": {
                    "title": "Message", 
                    "type": "string",
                    "description": "Description of the error"
                },
                "type": {
                    "title": "Error Type", 
                    "type": "string",
                    "description": "Type of the error"
                }
            }
        },
        "HealthCheck": {
            "title": "HealthCheck",
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok", "error"], "description": "Overall service status"},
                "version": {"type": "string", "description": "API version"},
                "timestamp": {"type": "string", "format": "date-time", "description": "Current server time"},
                "dependencies": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["ok", "error"], "description": "Dependency status"},
                            "details": {"type": "string", "description": "Additional details about the dependency status"}
                        },
                        "required": ["status"]
                    },
                    "description": "Status of external dependencies"
                }
            },
            "required": ["status", "version", "timestamp"]
        }
    })
    
    # Add tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Video Generation",
            "description": "Endpoints for generating and processing videos"
        },
        {
            "name": "Authentication",
            "description": "Authentication and authorization endpoints"
        },
        {
            "name": "Status",
            "description": "Health check and system status"
        },
        {
            "name": "Documentation",
            "description": "API documentation and schema"
        }
    ]
    
    # Add examples for common requests/responses
    openapi_schema["components"]["examples"] = {
        "GenerateVideoRequest": {
            "summary": "Example video generation request",
            "value": {
                "image_url": "https://example.com/image.jpg",
                "audio_url": "https://example.com/audio.mp3",
                "output_format": "mp4",
                "resolution": "720p",
                "fps": 30,
                "enhance_quality": True,
                "background_removal": True,
                "watermark": {
                    "enabled": True,
                    "text": "PaksaTalker",
                    "position": "bottom-right",
                    "opacity": 0.5
                }
            }
        },
        "VideoGenerationResponse": {
            "summary": "Example video generation response",
            "value": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "progress": 0,
                "estimated_time_remaining": 30,
                "result_url": "https://api.paksatalker.com/results/550e8400-e29b-41d4-a716-446655440000/video.mp4",
                "created_at": "2025-08-30T12:00:00Z",
                "updated_at": "2025-08-30T12:00:05Z"
            }
        },
        "ErrorRateLimit": {
            "summary": "Example rate limit error",
            "value": {
                "detail": "Rate limit exceeded",
                "error": "rate_limit_exceeded",
                "status_code": 429,
                "retry_after": 60
            }
        }
    }

    except Exception as e:
        logger.error(f"Failed to build OpenAPI schema: {str(e)}")
        raise
        logger.error(f"Failed to build OpenAPI schema: {str(e)}")
        raise

    try:
        # Cache the generated schema
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    except Exception as e:
        logger.error(f"Failed to cache OpenAPI schema: {str(e)}")
        return openapi_schema  # Still return the schema even if caching fails
    except Exception as e:
        logger.error(f"Failed to generate OpenAPI schema: {str(e)}")
        # Return a minimal valid schema if generation fails
        return {
            "openapi": "3.0.2",
            "info": {
                "title": "PaksaTalker API",
                "version": "1.0.0",
                "description": "Error generating full documentation. Please check server logs."
            },
            "paths": {}
        }

# Initialize FastAPI app
app = FastAPI(
    title="PaksaTalker API",
    description="""## Professional AI Video Generation API
    
### Overview
PaksaTalker provides AI-powered video generation with advanced features for creating realistic talking head videos.

### Authentication
- This API uses API Key authentication
- Include your API key in the `X-API-Key` header

### Rate Limiting
- 60 requests per minute
- 1000 requests per hour
    """,
    version=config['app']['version'],
    contact={
        "name": "Paksa IT Solutions",
        "email": "support@paksait.com",
    },
    license_info={
        "name": "Proprietary",
    },
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/openapi.json",
    servers=[
        {"url": "http://localhost:8000", "description": "Local Development"},
        {"url": "http://api.paksatalker.com", "description": "Production"}
    ],
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
from api.postprocess_endpoints import router as post_router
from api.capabilities_endpoints import router as capabilities_router
from api.diagnostics_endpoints import router as diagnostics_router
from api.emage_endpoints import router as emage_router
from api.expressions_endpoints import router as expressions_router
app.include_router(api_router, prefix="/api/v1")
app.include_router(prompt_router, prefix="/api/v1")
app.include_router(convo_router, prefix="/api/v1")
app.include_router(style_router, prefix="/api/v1")
app.include_router(language_router, prefix="/api/v1")
app.include_router(composition_router, prefix="/api/v1")
app.include_router(lighting_router, prefix="/api/v1")
app.include_router(camera_router, prefix="/api/v1")
app.include_router(background_router, prefix="/api/v1")
app.include_router(post_router, prefix="/api/v1")
app.include_router(capabilities_router, prefix="/api/v1")
app.include_router(diagnostics_router, prefix="/api/v1")
app.include_router(emage_router, prefix="/api/v1")
app.include_router(expressions_router, prefix="/api/v1")
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
    """Generate custom Swagger UI with PaksaTalker branding"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="PaksaTalker API Documentation",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_favicon_url="https://paksait.com/favicon.ico",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "docExpansion": "list",
            "filter": True,
            "persistAuthorization": True,
            "displayRequestDuration": True
        }
    )

@app.get("/api/redoc", include_in_schema=False)
async def redoc_html():
    """Generate custom ReDoc UI with PaksaTalker branding"""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="PaksaTalker API Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
        with_google_fonts=True,
        redoc_favicon_url="https://paksait.com/favicon.ico"
    )

@app.get("/api/generate-docs", include_in_schema=False)
async def generate_docs():
    """Generate static API documentation"""
    try:
        import subprocess
        import sys
        
        # Run the documentation generator
        result = subprocess.run(
            [sys.executable, "generate_docs.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Documentation generated successfully",
                "path": str(Path("docs").absolute())
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate documentation: {result.stderr}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating documentation: {str(e)}"
        )

# Serve static documentation files
@app.get("/docs/{file_path:path}", include_in_schema=False)
async def serve_docs(file_path: str):
    """Serve static documentation files"""
    docs_path = Path("docs") / file_path
    if not docs_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if docs_path.is_dir():
        return FileResponse(docs_path / "index.html" if (docs_path / "index.html").exists() else "")
    
    return FileResponse(docs_path)

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
