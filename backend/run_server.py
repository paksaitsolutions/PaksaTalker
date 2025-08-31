import uvicorn
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting server...")
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="debug"
    )
except Exception as e:
    logger.error(f"Failed to start server: {str(e)}")
    raise
