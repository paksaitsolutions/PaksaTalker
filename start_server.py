#!/usr/bin/env python3
"""
Simple startup script for PaksaTalker server
Handles dependency issues gracefully
"""
import sys
import os
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import fastapi
    except ImportError:
        missing_deps.append('fastapi')
    
    try:
        import uvicorn
    except ImportError:
        missing_deps.append('uvicorn')
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install them with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def start_minimal_server():
    """Start a minimal server without AI models"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pathlib import Path
    import uvicorn
    
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
    
    # Serve frontend
    frontend_dir = Path(__file__).parent / "frontend" / "dist"
    if frontend_dir.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")
        
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint not found")
            
            file_path = frontend_dir / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            
            # Return index.html for SPA routing
            index_path = frontend_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            
            raise HTTPException(status_code=404, detail="Frontend not built")
    
    # Basic API endpoints
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "message": "Server is running"}
    
    @app.get("/api/v1/health")
    async def health_v1():
        return {"status": "ok", "message": "Server is running", "models": "loading"}
    
    @app.post("/api/v1/generate/video")
    async def generate_video():
        return JSONResponse(
            status_code=503,
            content={"error": "AI models are still loading. Please try again in a few minutes."}
        )
    
    @app.post("/api/v1/generate/video-from-prompt")
    async def generate_video_from_prompt():
        return JSONResponse(
            status_code=503,
            content={"error": "AI models are still loading. Please try again in a few minutes."}
        )
    
    logger.info("Starting minimal PaksaTalker server...")
    logger.info("Server will be available at: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

def main():
    """Main entry point"""
    logger.info("Starting PaksaTalker server...")
    
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Try to import the full app
        from app import app
        import uvicorn
        
        logger.info("Full application loaded successfully")
        logger.info("Server will be available at: http://localhost:8000")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        logger.warning(f"Could not start full application: {e}")
        logger.info("Starting minimal server instead...")
        start_minimal_server()

if __name__ == "__main__":
    main()