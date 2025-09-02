# AGI Embedding Pipeline Test Launcher
# Starts all services and runs the comprehensive test

Write-Host "🚀 Starting AGI Embedding Pipeline Test" -ForegroundColor Green
Write-Host "=" * 50

# Check if Python environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not detected. Activating..." -ForegroundColor Yellow
    & ".\.venv\Scripts\Activate.ps1"
}

# Function to start service in background
function Start-ServiceInBackground {
    param(
        [string]$ServiceName,
        [string]$Command,
        [string]$WorkingDir = "."
    )
    
    Write-Host "🔄 Starting $ServiceName..." -ForegroundColor Cyan
    
    $job = Start-Job -ScriptBlock {
        param($cmd, $dir)
        Set-Location $dir
        Invoke-Expression $cmd
    } -ArgumentList $Command, $WorkingDir
    
    Write-Host "✅ $ServiceName started (Job ID: $($job.Id))" -ForegroundColor Green
    return $job
}

# Start services
$embeddingJob = Start-ServiceInBackground -ServiceName "Embedding Service" -Command "python embedding_service.py"
$memoryJob = Start-ServiceInBackground -ServiceName "Memory Server" -Command ".\memory\memory_server.exe" -WorkingDir "."

# Wait for services to initialize
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if services are responding
Write-Host "🔍 Checking service health..." -ForegroundColor Cyan

try {
    Invoke-RestMethod -Uri "http://localhost:8003/health" -TimeoutSec 5 | Out-Null
    Write-Host "✅ Embedding service is healthy" -ForegroundColor Green
} catch {
    Write-Host "❌ Embedding service not responding" -ForegroundColor Red
}

try {
    Invoke-RestMethod -Uri "http://localhost:8001/health" -TimeoutSec 5 | Out-Null
    Write-Host "✅ Memory server is healthy" -ForegroundColor Green
} catch {
    Write-Host "❌ Memory server not responding" -ForegroundColor Red
}

# Run the test
Write-Host ""
Write-Host "🧪 Running comprehensive pipeline test..." -ForegroundColor Cyan
Write-Host "=" * 50

python test_embedding_pipeline.py

# Cleanup
Write-Host ""
Write-Host "🧹 Cleaning up background services..." -ForegroundColor Yellow

Stop-Job -Job $embeddingJob -Force
Stop-Job -Job $memoryJob -Force
Remove-Job -Job $embeddingJob -Force
Remove-Job -Job $memoryJob -Force

Write-Host "✅ Test complete and services stopped" -ForegroundColor Green
