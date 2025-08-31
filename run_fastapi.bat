@echo off
setlocal

:: Set paths
set PYTHON_PATH=D:\PulsaTalker\venv\Scripts\python.exe
set APP_PATH=test_fastapi_simple.py
set LOG_FILE=fastapi.log

echo Starting FastAPI server...
echo Python: %PYTHON_PATH%
echo App: %APP_PATH%
echo Log: %LOG_FILE%
echo.

:: Run the FastAPI application with logging
"%PYTHON_PATH%" -u "%APP_PATH%" > "%LOG_FILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to start FastAPI server
    type "%LOG_FILE%"
    pause
    exit /b %ERRORLEVEL%
)

endlocal
