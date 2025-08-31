@echo off
setlocal

:: Set Python and environment paths
set PYTHONPATH=D:\PulsaTalker
set UVICORN_LOG_LEVEL=debug

:: Activate virtual environment
call .\venv\Scripts\activate.bat

:: Install dependencies
echo Installing required packages...
pip install fastapi uvicorn python-multipart aiofiles

:: Start the FastAPI server
echo Starting FastAPI server...
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

pause
