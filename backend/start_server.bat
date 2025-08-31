@echo off
setlocal

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment
        exit /b %ERRORLEVEL%
    )
)

REM Activate virtual environment and install requirements
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment
    exit /b %ERRORLEVEL%
)

echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies
    exit /b %ERRORLEVEL%
)

echo Starting backend server...
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 5000 --log-level debug

endlocal
