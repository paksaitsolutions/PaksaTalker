@echo off
echo Starting PulsaTalker Development Environment...

echo.
echo [1/3] Installing mock API dependencies...
cd /d %~dp0
call npm install --prefix . -g npm@latest
call npm install --prefix .

start "" cmd /k "cd /d %~dp0 && npm start"

echo.
echo [2/3] Starting Mock API Server on http://localhost:3001
start "" cmd /k "cd /d %~dp0 && node mock-api.js"

echo.
echo [3/3] Starting Frontend Development Server...
cd /d %~dp0\frontend
call npm install
call npm run dev

pause
