#!/usr/bin/env powershell

Write-Host "🚀 Starting PaksaTalker Development Environment..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Node.js is available
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Node.js is not installed or not in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting backend server..." -ForegroundColor Yellow

# Start backend server
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
}

# Wait for backend to start
Start-Sleep -Seconds 3

Write-Host "Starting frontend server..." -ForegroundColor Yellow

# Start frontend server
$frontendJob = Start-Job -ScriptBlock {
    Set-Location "$using:PWD\frontend"
    npm run dev
}

Write-Host ""
Write-Host "✅ Development servers started!" -ForegroundColor Green
Write-Host "📱 Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8000/api/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop servers or close this window..." -ForegroundColor Yellow

# Wait for user to stop
try {
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if jobs are still running
        if ($backendJob.State -eq "Failed" -or $backendJob.State -eq "Completed") {
            Write-Host "Backend server stopped" -ForegroundColor Red
            break
        }
        if ($frontendJob.State -eq "Failed" -or $frontendJob.State -eq "Completed") {
            Write-Host "Frontend server stopped" -ForegroundColor Red
            break
        }
    }
} catch {
    Write-Host "Stopping servers..." -ForegroundColor Yellow
}

# Clean up jobs
Stop-Job $backendJob -ErrorAction SilentlyContinue
Stop-Job $frontendJob -ErrorAction SilentlyContinue
Remove-Job $backendJob -ErrorAction SilentlyContinue
Remove-Job $frontendJob -ErrorAction SilentlyContinue

Write-Host "✅ Servers stopped" -ForegroundColor Green