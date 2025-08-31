@echo off
setlocal

:: Start the backend server
echo Starting FastAPI backend on http://localhost:5000
start "Backend Server" cmd /k "cd /d %~dp0 && .\venv\Scripts\activate.bat && python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 5000"

:: Start the frontend server
timeout /t 5 /nobreak >nul
echo Starting Vite frontend on http://localhost:5173
start "Frontend Server" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:5173
echo.
pause
