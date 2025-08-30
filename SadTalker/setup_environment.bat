@echo off
echo Setting up SadTalker environment...

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; exit(0) if sys.version_info >= (3, 8) else exit(1)"
if %ERRORLEVEL% NEQ 0 (
    echo Python 3.8 or higher is required. Found:
    python --version
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv sadtalker_env
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate environment and install requirements
echo Installing requirements...
call sadtalker_env\\Scripts\\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Upgrade pip
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

:: Install PyTorch with CUDA support if available
echo Installing PyTorch...
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install PyTorch. Trying CPU-only version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install PyTorch CPU version.
        pause
        exit /b 1
    )
)

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

:: Install additional requirements for PaksaTalker
if exist "PaksaTalker\\requirements.txt" (
    echo Installing PaksaTalker requirements...
    pip install -r PaksaTalker\\requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install PaksaTalker requirements.
        pause
        exit /b 1
    )
)

echo.
echo ====================================
echo Environment setup completed successfully!
echo ====================================
echo.
echo To activate the environment, run:
echo   sadtalker_env\\Scripts\\activate.bat
echo.
echo Then you can run the web UI with:
echo   webui.bat
echo.
pause
