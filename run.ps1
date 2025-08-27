# Set error action preference
$ErrorActionPreference = "Stop"

# Set Python path
$env:PYTHONPATH = $PSScriptRoot

# Function to check if a port is in use
function Test-PortInUse {
    param([int]$port)
    $tcpClient = New-Object Net.Sockets.TcpClient
    try {
        $tcpClient.Connect("localhost", $port)
        $tcpClient.Close()
        return $true
    } catch {
        return $false
    }
}

# Check if port 8000 is in use
if (Test-PortInUse -port 8000) {
    Write-Host "Port 8000 is already in use. Please close any applications using this port and try again." -ForegroundColor Red
    exit 1
}

try {
    Write-Host "Starting PulsaTalker FastAPI server..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    
    # Start the FastAPI server
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start FastAPI server"
    }
} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
    exit 1
}
