import sys
import os
import logging

# Set up logging to a file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='debug.log',
    filemode='w'
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

def log_environment():
    """Log environment information for debugging."""
    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version)
    logger.info("Current working directory: %s", os.getcwd())
    logger.info("Python path: %s", sys.path)
    
    # Log environment variables
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if 'PYTHON' in key.upper() or 'PATH' in key.upper():
            logger.info("  %s: %s", key, value)

def run_fastapi():
    """Run the FastAPI application with detailed logging."""
    try:
        logger.info("Starting FastAPI application...")
        
        # Import FastAPI and create app
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI()
        
        @app.get("/")
        async def read_root():
            logger.info("Root endpoint called")
            return {"message": "PulsaTalker API is working!"}
        
        logger.info("Starting uvicorn server...")
        uvicorn.run(
            "__main__:app",
            host="0.0.0.0",
            port=8000,
            log_level="debug",
            reload=True
        )
        
    except Exception as e:
        logger.exception("Error in run_fastapi: %s", str(e))
        return 1
    return 0

if __name__ == "__main__":
    log_environment()
    sys.exit(run_fastapi())
