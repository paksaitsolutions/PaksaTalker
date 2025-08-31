@echo off
setlocal

echo Starting PulsaTalker FastAPI server...
echo ===================================

:: Set Python path to the virtual environment
set PYTHON_PATH=D:\PulsaTalker\venv\Scripts\python.exe
set APP_PATH=D:\PulsaTalker\app.py

:: Check if Python executable exists
if not exist "%PYTHON_PATH%" (
    echo Error: Python executable not found at %PYTHON_PATH%
    pause
    exit /b 1
)

:: Check if app.py exists
if not exist "%APP_PATH%" (
    echo Error: app.py not found at %APP_PATH%
    pause
    exit /b 1
)

echo Python: %PYTHON_PATH%
echo App: %APP_PATH%
echo.

:: Set environment variables
set PYTHONUNBUFFERED=1
set PYTHONPATH=D:\PulsaTalker

:: Run the FastAPI application
"%PYTHON_PATH%" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to start FastAPI server
    pause
    exit /b %ERRORLEVEL%
)

endlocal
