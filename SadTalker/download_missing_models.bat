@echo off
echo Downloading missing model files...

set "base_url=https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
set "checkpoints_dir=D:\SadTalker\checkpoints"

mkdir "%checkpoints_dir%" 2>nul

:: Function to download a file if it doesn't exist
:download
if not exist "%checkpoints_dir%\%~1" (
    echo Downloading %~1...
    curl -L -o "%checkpoints_dir%\%~1" "%base_url%/%~1"
    if %errorlevel% neq 0 (
        echo Failed to download %~1
        exit /b 1
    )
) else (
    echo %~1 already exists
)
goto :eof

:: Download missing files
call :download "auido2exp_00300-model.pth"
call :download "auido2pose_00140-model.pth"
call :download "facevid2vid_00189-model.pth.tar"
call :download "shape_predictor_68_face_landmarks.dat"

echo.
echo All required files have been downloaded to %checkpoints_dir%
pause
