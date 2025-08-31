import os
import uuid
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import logging
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
from pydantic import BaseModel
from typing import Optional, Dict
import aiofiles
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Processing API", version="1.0.0")

# API v1 Router
from fastapi import APIRouter
from .routers import styles, status

# Main API router
api_router = APIRouter(tags=["api"])

# Include routers
api_router.include_router(router=styles.router)
api_router.include_router(router=status.router, prefix="/status")

# Include the API router with /api/v1 prefix
app.include_router(api_router, prefix="/api/v1")

@api_router.post("/generate/video")
async def generate_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Handle video generation request"""
    try:
        # Generate a unique ID for this video
        video_id = str(uuid.uuid4())
        video_path = get_video_path(video_id)
        
        # Save the uploaded file
        await save_upload_file(file, video_path)
        
        # Start processing in background
        background_tasks.add_task(process_video_task, video_id)
        
        # Update status
        video_status[video_id] = {
            "id": video_id,
            "status": "processing",
            "progress": 0,
            "download_url": None,
            "error": None
        }
        
        return {"success": True, "task_id": video_id}
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing video: {str(e)}"
        )

# The router is already included above with the /api/v1 prefix

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Video Processing API is running"}

@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to list all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, "name", ""),
                "endpoint": getattr(route, "endpoint", "").__name__
            })
    return {"routes": routes}


# CORS middleware configuration with enhanced security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173"   # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # For file downloads
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    # Add CORS headers to response
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    logger.info(f"Response status: {response.status_code}")
    logger.info(f"Response headers: {dict(response.headers)}")
    return response

# Configuration
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MAX_FILE_SIZE = 1024 * 1024 * 500  # 500MB
ALLOWED_EXTENSIONS = {"mp4", "webm", "ogg"}

# Ensure upload and processed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# In-memory storage for video status (in production, use a database)
video_status: Dict[str, Dict] = {}

class VideoStatus(BaseModel):
    id: str
    status: str  # 'uploading', 'processing', 'completed', 'failed'
    progress: int  # 0-100
    download_url: Optional[str] = None
    error: Optional[str] = None

def get_video_path(video_id: str) -> Path:
    """Get the path to store the uploaded video."""
    return Path(UPLOAD_FOLDER) / f"{video_id}"

def get_processed_path(video_id: str) -> Path:
    """Get the path to store the processed video."""
    return Path(PROCESSED_FOLDER) / f"{video_id}.mp4"

async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file to the specified path."""
    try:
        async with aiofiles.open(destination, 'wb') as buffer:
            while content := await upload_file.read(1024 * 1024):  # 1MB chunks
                await buffer.write(content)
    except Exception as e:
        logger.error(f"Error saving file {upload_file.filename}: {e}")
        if destination.exists():
            destination.unlink()
        raise

async def process_video_task(video_id: str):
    """Background task to process the video."""
    try:
        video_status[video_id] = {
            "status": "processing",
            "progress": 0,
            "error": None,
            "download_url": None
        }
        
        input_path = get_video_path(video_id)
        output_path = get_processed_path(video_id)
        
        # Simulate processing (replace with actual video processing logic)
        for i in range(1, 11):
            await asyncio.sleep(2)  # Simulate processing time
            progress = i * 10
            video_status[video_id]["progress"] = progress
            logger.info(f"Processing {video_id}: {progress}% complete")
        
        # In a real application, you would process the video here
        # For now, we'll just copy the file as a placeholder
        shutil.copy2(input_path, output_path)
        
        # Update status to completed
        video_status[video_id].update({
            "status": "completed",
            "progress": 100,
            "download_url": f"/api/videos/{video_id}/download"
        })
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        if video_id in video_status:
            video_status[video_id].update({
                "status": "failed",
                "error": str(e)
            })

@app.post("/api/videos/upload", response_model=VideoStatus)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    logger.info(f"Received upload request for file: {file.filename}")
    logger.info(f"Content type: {file.content_type}")
    logger.info(f"Headers: {file.headers}")
    """Upload a video file for processing."""
    # Check file size
    file.file.seek(0, 2)  # Move to the end of the file
    file_size = file.file.tell()
    file.file.seek(0)  # Reset file pointer
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {MAX_FILE_SIZE / (1024 * 1024)}MB"
        )
    
    # Check file extension
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate a unique ID for this video
    video_id = str(uuid.uuid4())
    file_path = get_video_path(video_id)
    
    try:
        # Save the uploaded file
        await save_upload_file(file, file_path)
        
        # Start background processing
        background_tasks.add_task(process_video_task, video_id)
        
        # Initial status
        video_status[video_id] = {
            "id": video_id,
            "status": "uploading",
            "progress": 100,  # Upload is complete
            "download_url": None
        }
        
        return video_status[video_id]
        
    except Exception as e:
        logger.error(f"Error handling upload: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing file"
        )

@app.post("/api/videos/{video_id}/process")
async def start_processing(video_id: str):
    """Start processing a video (already handled in upload)."""
    if video_id not in video_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {"status": "processing_started", "video_id": video_id}

@app.get("/api/videos/{video_id}/status", response_model=VideoStatus)
async def get_status(video_id: str):
    """Get the status of a video processing job."""
    if video_id not in video_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video_status[video_id]

@app.get("/api/videos/{video_id}/download")
async def download_video(video_id: str):
    """Download the processed video."""
    if video_id not in video_status or video_status[video_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Video not available for download")
    
    output_path = get_processed_path(video_id)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"processed_{video_id}.mp4"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
