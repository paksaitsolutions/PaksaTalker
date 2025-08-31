import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
