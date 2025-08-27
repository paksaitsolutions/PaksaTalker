@echo off
setlocal

echo Setting up PulsaTalker...
echo ======================

:: Set Python path if needed
set PYTHONPATH=%~dp0

:: Run the FastAPI application
echo Starting FastAPI server...
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

if %ERRORLEVEL% NEQ 0 (
    echo Failed to start FastAPI server
    pause
    exit /b %ERRORLEVEL%
)

endlocal
