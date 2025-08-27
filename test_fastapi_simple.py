import uvicorn
from fastapi import FastAPI
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "PulsaTalker API is working!"}

if __name__ == "__main__":
    logger.info("Starting test FastAPI server...")
    uvicorn.run(
        "test_fastapi_simple:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
