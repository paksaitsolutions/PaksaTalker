@echo off
setlocal

:: Set paths
set PYTHON_PATH=D:\PulsaTalker\venv\Scripts\python.exe
set LOG_FILE=D:\PulsaTalker\fastapi_debug.log

:: Run Python with verbose output and log to file
"%PYTHON_PATH%" -v -c "
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s',
    filename=r'%LOG_FILE%',
    filemode='w'
)

logger = logging.getLogger(__name__)

try:
    logger.info('Starting FastAPI application...')
    logger.info('Python executable: %s', sys.executable)
    logger.info('Python version: %s', sys.version)
    logger.info('Current working directory: %s', os.getcwd())
    
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI()
    
    @app.get('/')
    async def root():
        logger.info('Root endpoint called')
        return {'message': 'PulsaTalker API is working!'}
    
    logger.info('Starting uvicorn server...')
    uvicorn.run(
        '__main__:app',
        host='0.0.0.0',
        port=8000,
        log_level='debug',
        reload=True
    )
    
except Exception as e:
    logger.exception('Error in FastAPI application')
    raise
" 2>&1 | findstr /v "^#"

endlocal
