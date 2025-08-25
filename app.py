""PaksaTalker - AI-Powered Video Generation Platform"""
import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
from config import config

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if config['app.debug'] else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config['app.name'],
    version=config['app.version'],
    description="AI-Powered Video Generation Platform",
    docs_url="/docs" if config['app.debug'] else None,
    redoc_url="/redoc" if config['app.debug'] else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=config['paths.static']), name="static")

# Import and include routers
from api.routes import router as api_router
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": config['app.version'],
        "debug": config['app.debug']
    }

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
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=config['api.host'],
        port=config['api.port'],
        reload=config['app.debug'],
        workers=config['api.workers']
    )
