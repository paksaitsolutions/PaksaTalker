import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting PulsaTalker in debug mode...")
        
        # Add the current directory to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import and run the FastAPI app
        from app import app
        import uvicorn
        
        logger.info("Starting uvicorn server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="debug",
            reload=True
        )
    except Exception as e:
        logger.error(f"Failed to start PulsaTalker: {str(e)}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
