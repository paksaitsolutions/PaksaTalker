@echo off
echo Starting PaksaTalker Production Server on Port 8080...
cd /d "D:\PaksaTalker"
call .\venv\Scripts\activate
echo.
echo Server will be available at: http://localhost:8080
echo Press Ctrl+C to stop the server
echo.
python app.py