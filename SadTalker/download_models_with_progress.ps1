# Download model files with progress tracking
$checkpointsDir = "D:\SadTalker\checkpoints"
$files = @(
    @{
        Name = "auido2exp_00300-model.pth"
        Url = "https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/checkpoints/auido2exp_00300-model.pth"
    },
    @{
        Name = "auido2pose_00140-model.pth"
        Url = "https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/checkpoints/auido2pose_00140-model.pth"
    },
    @{
        Name = "facevid2vid_00189-model.pth.tar"
        Url = "https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/checkpoints/facevid2vid_00189-model.pth.tar"
    },
    @{
        Name = "shape_predictor_68_face_landmarks.dat"
        Url = "https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/checkpoints/shape_predictor_68_face_landmarks.dat"
    }
)

# Create checkpoints directory if it doesn't exist
if (-not (Test-Path -Path $checkpointsDir)) {
    New-Item -ItemType Directory -Path $checkpointsDir | Out-Null
}

foreach ($file in $files) {
    $outputPath = Join-Path -Path $checkpointsDir -ChildPath $file.Name
    
    Write-Host "`nDownloading $($file.Name)..." -ForegroundColor Cyan
    
    # Download with progress
    try {
        $ProgressPreference = 'SilentlyContinue'
        $wc = New-Object System.Net.WebClient
        $wc.DownloadProgressChanged += {
            $percentComplete = $_.ProgressPercentage
            Write-Progress -Activity "Downloading $($file.Name)" -Status "$percentComplete% Complete:" -PercentComplete $percentComplete
        }
        $wc.DownloadFileAsync([System.Uri]::new($file.Url), $outputPath)
        
        # Wait for download to complete
        while ($wc.IsBusy) { Start-Sleep -Milliseconds 100 }
        
        Write-Host "✅ Successfully downloaded $($file.Name)" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to download $($file.Name)" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
    }
    finally {
        if ($wc -ne $null) {
            $wc.Dispose()
        }
    }
}

# Show final status
Write-Host "`nDownload process completed!" -ForegroundColor Cyan
Write-Host "`nFiles in $checkpointsDir:" -ForegroundColor Cyan
Get-ChildItem -Path $checkpointsDir | Select-Object Name, @{Name='SizeMB';Expression={[math]::Round($_.Length/1MB, 2)}} | Format-Table -AutoSize
