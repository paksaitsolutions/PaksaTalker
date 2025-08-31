@echo off
echo Starting PulsaTalker Development Servers...

echo Starting backend on http://localhost:8000...
start "Backend" cmd /k "python start_backend.py"

timeout /t 2 /nobreak >nul

echo Starting frontend on http://localhost:5173...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ✅ Servers starting...
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend: http://localhost:8000
echo.
echo Press any key to exit...
pause >nul