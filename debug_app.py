import uvicorn
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Log environment information
logger.info("Python executable: %s", sys.executable)
logger.info("Python path: %s", sys.path)
logger.info("Current working directory: %s", os.getcwd())

# Create FastAPI app
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def read_root():
    logger.info("Root endpoint called")
    return {"message": "PulsaTalker API is working!"}

if __name__ == "__main__":
    logger.info("Starting debug FastAPI server...")
    try:
        uvicorn.run(
            "debug_app:app",
            host="0.0.0.0",
            port=8000,
            log_level="debug",
            reload=True
        )
    except Exception as e:
        logger.exception("Failed to start FastAPI server: %s", str(e))
