@echo off
echo Checking Python environment...
python --version
echo.
echo Python path:
python -c "import sys; print('\n'.join(sys.path))"
echo.
echo Installed packages:
pip list
echo.
pause
