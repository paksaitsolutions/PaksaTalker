#!/usr/bin/env python3
"""
Simple working PaksaTalker server
Bypasses complex AI model loading for immediate functionality
"""
import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend" / "dist"
STATIC_DIR = FRONTEND_DIR / "assets"

# Create app
app = FastAPI(
    title="PaksaTalker",
    description="AI-Powered Video Generation Platform",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend assets
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

# API endpoints
@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "Server is running"}

@app.get("/api/v1/health")
async def health_v1():
    return {"status": "ok", "message": "Server is running", "models": "ready"}

@app.post("/api/v1/generate/video")
async def generate_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(None),
    text: str = Form(None),
    resolution: str = Form("720p"),
    fps: int = Form(30),
    expressionIntensity: float = Form(0.8),
    gestureLevel: str = Form("medium"),
    voiceModel: str = Form("en-US-JennyNeural"),
    background: str = Form("blur"),
    enhanceFace: bool = Form(True),
    stabilization: bool = Form(True)
):
    """Generate video from uploaded files"""
    
    # Validate inputs
    if not image:
        raise HTTPException(status_code=400, detail="Image file is required")
    
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either audio file or text is required")
    
    # Create a mock task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    logger.info(f"Video generation request: {task_id}")
    logger.info(f"Image: {image.filename}, Audio: {audio.filename if audio else 'None'}")
    logger.info(f"Text: {text[:50] if text else 'None'}...")
    logger.info(f"Settings: {resolution}, {fps}fps, {gestureLevel} gestures")
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "Video generation started",
        "status": "processing"
    }

@app.post("/api/v1/generate/video-from-prompt")
async def generate_video_from_prompt(
    prompt: str = Form(...),
    voice: str = Form("en-US-JennyNeural"),
    resolution: str = Form("720p"),
    fps: int = Form(30),
    gestureLevel: str = Form("medium")
):
    """Generate video from text prompt"""
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Create a mock task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    logger.info(f"Prompt-based generation: {task_id}")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Voice: {voice}, Resolution: {resolution}")
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "Video generation from prompt started",
        "status": "processing"
    }

@app.post("/api/generate/advanced-video")
async def generate_advanced_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(None),
    text: str = Form(None),
    useEmage: bool = Form(False),
    useWav2Lip2: bool = Form(False),
    useSadTalkerFull: bool = Form(False),
    emotion: str = Form("neutral"),
    bodyStyle: str = Form("natural"),
    avatarType: str = Form("realistic"),
    lipSyncQuality: str = Form("high"),
    resolution: str = Form("720p"),
    fps: int = Form(30)
):
    """Generate video with advanced AI models"""
    
    if not image:
        raise HTTPException(status_code=400, detail="Image file is required")
    
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either audio file or text is required")
    
    # Create a mock task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    logger.info(f"Advanced video generation: {task_id}")
    logger.info(f"Models: EMAGE={useEmage}, Wav2Lip2={useWav2Lip2}, SadTalker={useSadTalkerFull}")
    logger.info(f"Settings: {emotion} emotion, {bodyStyle} style, {avatarType} avatar")
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "Advanced video generation started",
        "status": "processing"
    }

@app.get("/api/status/{task_id}")
@app.get("/api/v1/status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    
    # Mock progress simulation
    import time
    import hashlib
    
    # Use task_id to create deterministic progress
    hash_val = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
    elapsed = int(time.time()) % 120  # 2-minute cycle
    
    if elapsed < 100:
        # Still processing
        progress = min(90, elapsed)
        return {
            "success": True,
            "data": {
                "status": "processing",
                "progress": progress,
                "stage": f"Processing... {progress}% complete"
            }
        }
    else:
        # Completed
        return {
            "success": True,
            "data": {
                "status": "completed",
                "progress": 100,
                "stage": "Video generation completed",
                "video_url": f"/api/v1/download/{task_id}.mp4"
            }
        }

@app.get("/api/v1/download/{filename}")
async def download_file(filename: str):
    """Download generated video"""
    
    # For demo, return a placeholder response
    return JSONResponse({
        "message": "Video download would be available here",
        "filename": filename,
        "note": "This is a demo server. Actual video generation requires AI models to be loaded."
    })

# Serve SPA
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve the React SPA"""
    
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # Try to serve the requested file
    if full_path and not full_path.startswith("api/"):
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
    
    # Return index.html for SPA routing
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    # Frontend not built
    return JSONResponse({
        "error": "Frontend not built",
        "message": "Please run 'npm run build' in the frontend directory",
        "frontend_dir": str(FRONTEND_DIR)
    }, status_code=500)

if __name__ == "__main__":
    logger.info("Starting simple PaksaTalker server...")
    logger.info("This is a demo server with mock AI functionality")
    logger.info("Server will be available at: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )