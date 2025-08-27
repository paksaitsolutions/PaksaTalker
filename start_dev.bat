@echo off
echo Starting PaksaTalker Development Environment...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting backend server on http://localhost:8000...
start "PaksaTalker Backend" cmd /k "python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

echo Starting frontend server on http://localhost:5173...
cd frontend
start "PaksaTalker Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ✅ Development servers starting...
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/api/docs
echo.
echo Press any key to exit...
pause >nul