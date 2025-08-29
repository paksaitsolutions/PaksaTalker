@echo off
echo Starting PaksaTalker Server...
echo.
echo Building frontend...
cd frontend
call npm run build
cd ..
echo.
echo Starting backend server...
echo Access the application at: http://localhost:8000
echo.
python app.py