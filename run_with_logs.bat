@echo off
echo Starting PulsaTalker with logging...
set PYTHONUNBUFFERED=1
set PYTHONPATH=%~dp0
"%~dp0venv\Scripts\python.exe" -u run_debug.py > debug_output.log 2>&1
echo Logs have been written to debug_output.log
type debug_output.log
pause
