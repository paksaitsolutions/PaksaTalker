# Download model files with progress tracking
$checkpointsDir = "D:\SadTalker\checkpoints"

# Create directory if it doesn't exist
if (-not (Test-Path -Path $checkpointsDir)) {
    New-Item -ItemType Directory -Path $checkpointsDir | Out-Null
}

# Function to download file with progress
function Download-FileWithProgress {
    param (
        [string]$url,
        [string]$outputPath
    )
    
    Write-Host "Downloading $([System.IO.Path]::GetFileName($outputPath))..." -ForegroundColor Cyan
    
    try {
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadProgressChanged += {
            $percentComplete = $_.ProgressPercentage
            Write-Progress -Activity "Downloading $([System.IO.Path]::GetFileName($outputPath))" -Status "$percentComplete% Complete:" -PercentComplete $percentComplete
        }
        
        $webClient.DownloadFileAsync([System.Uri]::new($url), $outputPath)
        
        # Wait for download to complete
        while ($webClient.IsBusy) { 
            Start-Sleep -Milliseconds 100 
        }
        
        # Check if file exists and has size > 0
        if (Test-Path $outputPath -PathType Leaf) {
            $fileSize = (Get-Item $outputPath).Length / 1MB
            Write-Host "✓ Successfully downloaded $([System.IO.Path]::GetFileName($outputPath)) ($($fileSize.ToString('0.00')) MB)" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ File download failed: $outputPath" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Error downloading $($_.Exception.Message)" -ForegroundColor Red
        if (Test-Path $outputPath) { Remove-Item $outputPath -Force }
        return $false
    }
    finally {
        if ($webClient -ne $null) {
            $webClient.Dispose()
        }
    }
}

# Files to download
$files = @(
    @{
        Name = "auido2exp_00300-model.pth"
        Url = "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/auido2exp_00300-model.pth"
    },
    @{
        Name = "auido2pose_00140-model.pth"
        Url = "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/auido2pose_00140-model.pth"
    },
    @{
        Name = "facevid2vid_00189-model.pth.tar"
        Url = "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/facevid2vid_00189-model.pth.tar"
    },
    @{
        Name = "shape_predictor_68_face_landmarks.dat"
        Url = "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/shape_predictor_68_face_landmarks.dat"
    }
)

# Download each file
$success = $true
foreach ($file in $files) {
    $outputPath = Join-Path -Path $checkpointsDir -ChildPath $file.Name
    
    # Skip if file already exists and has size > 0
    if (Test-Path $outputPath -PathType Leaf) {
        $fileSize = (Get-Item $outputPath).Length / 1MB
        if ($fileSize -gt 0.1) {  # At least 100KB
            Write-Host "✓ $($file.Name) already exists ($($fileSize.ToString('0.00')) MB)" -ForegroundColor Green
            continue
        } else {
            Write-Host "ℹ️  $($file.Name) exists but is too small, re-downloading..." -ForegroundColor Yellow
            Remove-Item $outputPath -Force -ErrorAction SilentlyContinue
        }
    }
    
    if (-not (Download-FileWithProgress -url $file.Url -outputPath $outputPath)) {
        $success = $false
    }
}

# Show final status and directory contents
Write-Host "`nDownload process completed!" -ForegroundColor Cyan
Write-Host "`nCurrent contents of $checkpointsDir:" -ForegroundColor Cyan
Get-ChildItem -Path $checkpointsDir | Select-Object Name, @{Name='SizeMB';Expression={[math]::Round($_.Length/1MB, 2)}} | Format-Table -AutoSize

if (-not $success) {
    Write-Host "`n❌ Some files failed to download. Please try again or download them manually." -ForegroundColor Red
    exit 1
} else {
    Write-Host "`n✅ All files downloaded successfully!" -ForegroundColor Green
    exit 0
}
