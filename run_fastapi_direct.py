import sys
import os
import logging

# Set up logging to a file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='fastapi_debug.log',
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

if __name__ == "__main__":
    log_environment()
    
    try:
        logger.info("Importing FastAPI...")
        from fastapi import FastAPI
        import uvicorn
        
        logger.info("Creating FastAPI app...")
        app = FastAPI()
        
        @app.get("/")
        async def root():
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
        logger.exception("Error occurred:")
