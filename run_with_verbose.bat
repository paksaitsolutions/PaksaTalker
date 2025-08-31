@echo off
set PYTHONUNBUFFERED=1
"%~dp0venv\Scripts\python.exe" -v -c "import sys; print('Python executable:', sys.executable); print('Python version:', sys.version); print('Current working directory:', 'D:\\PulsaTalker')" > verbose_output.txt 2>&1
type verbose_output.txt
