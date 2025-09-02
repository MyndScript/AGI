# AGI Launch Script - Best of Both Worlds
# Combines PowerShell simplicity with Python orchestration

param(
    [switch]$CleanStart,
    [switch]$NoUI,
    [switch]$Verbose
)

Write-Host "🚀 AGI System Launch - Production Mode" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Yellow

# Set environment variables
$env:PYTHONPATH = "$PSScriptRoot"
$env:AGI_AGENT_PORT = "8000"
$env:AGI_MEMORY_PORT = "8001"
$env:AGI_PERSONALITY_PORT = "8002"
$env:AGI_OVERSEER_PORT = "8010"

if ($Verbose) {
    Write-Host "Environment Variables:" -ForegroundColor Cyan
    Write-Host "  PYTHONPATH: $env:PYTHONPATH"
    Write-Host "  AGI_AGENT_PORT: $env:AGI_AGENT_PORT"
    Write-Host "  AGI_MEMORY_PORT: $env:AGI_MEMORY_PORT"
    Write-Host "  AGI_PERSONALITY_PORT: $env:AGI_PERSONALITY_PORT"
    Write-Host "  AGI_OVERSEER_PORT: $env:AGI_OVERSEER_PORT"
}

# Function to kill process on port
function Kill-PortProcess {
    param([int]$Port)
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $connections) {
            $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "🔪 Terminating $($process.Name) (PID: $($process.Id)) on port $Port" -ForegroundColor Yellow
                Stop-Process -Id $process.Id -Force
                Start-Sleep -Seconds 1
            }
        }
    } catch {
        Write-Host "⚠️  Could not clear port $Port : $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Function to wait for service
function Wait-ForService {
    param([int]$Port, [string]$ServiceName, [int]$Timeout = 30)
    $startTime = Get-Date
    while (((Get-Date) - $startTime).TotalSeconds -lt $Timeout) {
        try {
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $tcpClient.Connect("localhost", $Port)
            $tcpClient.Close()
            Write-Host "✅ $ServiceName is ready on port $Port" -ForegroundColor Green
            return $true
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Host "❌ $ServiceName failed to start on port $Port" -ForegroundColor Red
    return $false
}

# Clean start if requested
if ($CleanStart) {
    Write-Host "🧹 Performing clean start - clearing all ports..." -ForegroundColor Yellow
    @(3000, 8000, 8001, 8002, 8004, 8010) | ForEach-Object { Kill-PortProcess -Port $_ }
    Start-Sleep -Seconds 2
}

# Start services in order
$services = @(
    @{
        Name = "Overseer Gateway"
        Command = "python"
        Args = "overseer_gateway.py"
        Port = 8010
        WorkingDir = $PSScriptRoot
    },
    @{
        Name = "Memory Server"
        Command = ".\memory\memory_rest_server.exe"
        Args = ""
        Port = 8001
        WorkingDir = $PSScriptRoot
    },
    @{
        Name = "Personality Server"
        Command = "python"
        Args = "decentralized_personality_server.py"
        Port = 8002
        WorkingDir = $PSScriptRoot
    },
    @{
        Name = "Agent Server"
        Command = "python"
        Args = "decentralized_agent_server.py"
        Port = 8000
        WorkingDir = $PSScriptRoot
    }
)

$runningProcesses = @()

foreach ($service in $services) {
    Write-Host "🔮 Summoning $($service.Name) on port $($service.Port)..." -ForegroundColor Magenta

    # Kill any existing process on the port
    Kill-PortProcess -Port $service.Port

    # Start the service in a new terminal window
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $service.Command
    $startInfo.Arguments = $service.Args
    $startInfo.WorkingDirectory = $service.WorkingDir
    $startInfo.UseShellExecute = $true  # Use shell execute to open in new window
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Normal

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo

    try {
        $started = $process.Start()
        if ($started) {
            $runningProcesses += $process
            Write-Host "✨ $($service.Name) launched in new window (PID: $($process.Id))" -ForegroundColor Green

            # Wait for service to be ready
            if (Wait-ForService -Port $service.Port -ServiceName $service.Name) {
                Write-Host "🎉 $($service.Name) is alive and listening!" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "💥 Failed to launch $($service.Name): $($_.Exception.Message)" -ForegroundColor Red
    }

    Start-Sleep -Seconds 1
}

# Start UI if not disabled
if (-not $NoUI) {
    Write-Host "🎨 Starting UI..." -ForegroundColor Magenta
    Kill-PortProcess -Port 3000

    $uiProcess = Start-Process -NoNewWindow -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory ".\ui" -PassThru
    $runningProcesses += $uiProcess

    if (Wait-ForService -Port 3000 -ServiceName "UI") {
        Write-Host "🎉 UI is alive on port 3000!" -ForegroundColor Green
    }
}

# Start Monitor System
Write-Host "📊 Starting System Monitor..." -ForegroundColor Magenta
$monitorProcess = Start-Process -NoNewWindow -FilePath "python" -ArgumentList "monitor_system.py" -WorkingDirectory $PSScriptRoot -PassThru
$runningProcesses += $monitorProcess
Write-Host "🎉 System Monitor started!" -ForegroundColor Green

Write-Host ""
Write-Host "🎊 AGI System Launch Complete!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Yellow
Write-Host "🌐 Overseer Gateway: http://localhost:8010" -ForegroundColor Cyan
Write-Host "🤖 Agent Server: http://localhost:8000" -ForegroundColor Cyan
Write-Host "🧠 Memory Server: http://localhost:8001" -ForegroundColor Cyan
Write-Host "💝 Personality Server: http://localhost:8002" -ForegroundColor Cyan
Write-Host "🔍 Embedding Server: http://localhost:8004" -ForegroundColor Cyan
if (-not $NoUI) {
    Write-Host "🖥️  UI: http://localhost:3000" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "Press Ctrl+C to stop all services..." -ForegroundColor Yellow

# Wait for Ctrl+C
try {
    while ($true) {
        Start-Sleep -Seconds 1

        # Check if any processes have died
        $deadProcesses = $runningProcesses | Where-Object { $_.HasExited }
        if ($deadProcesses) {
            Write-Host "⚠️  Some services have stopped:" -ForegroundColor Red
            foreach ($proc in $deadProcesses) {
                Write-Host "   - $($proc.ProcessName) (PID: $($proc.Id))" -ForegroundColor Red
            }
        }
    }
} finally {
    Write-Host ""
    Write-Host "🛑 Shutting down AGI system..." -ForegroundColor Yellow

    foreach ($process in $runningProcesses) {
        if (-not $process.HasExited) {
            Write-Host "🔪 Terminating $($process.ProcessName) (PID: $($process.Id))" -ForegroundColor Yellow
            try {
                $process.Kill()
            } catch {
                Write-Host "⚠️  Could not terminate $($process.ProcessName): $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }

    Write-Host "✨ All services terminated. Farewell!" -ForegroundColor Green
}
