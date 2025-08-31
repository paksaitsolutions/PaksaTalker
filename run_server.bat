@echo off
echo Starting PulsaTalker server...
set PYTHONPATH=%~dp0
"%PYTHON%" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
if %ERRORLEVEL% NEQ 0 (
    echo Failed to start PulsaTalker server
    pause
)
