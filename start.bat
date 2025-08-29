@echo off
echo Starting PaksaTalker Production Server...
cd /d "D:\PaksaTalker"
call .\venv\Scripts\activate
python app.py
pause