# Set ports for all services
$env:UI_PORT = 3000
$env:AGI_AGENT_PORT = 8000
$env:AGI_MEMORY_PORT = 8001
$env:AGI_PERSONALITY_PORT = 8002
$env:AGI_EMBEDDING_PORT = 8003
$env:AGI_OVERSEER_PORT = 8010
# PowerShell launch script for AGI system
# This script starts all FastAPI backends and the UI frontend
# Terminals are hidden (no window popups)




# Start all services in separate terminal windows for debugging

# Start overseer gateway (central FastAPI proxy)
Start-Process -NoNewWindow:$false -FilePath python -ArgumentList "overseer_gateway.py"

# Start decentralized agent server (FastAPI, port 8000)
Start-Process -NoNewWindow:$false -FilePath python -ArgumentList "decentralized_agent_server.py"

# Start decentralized personality server (FastAPI, port 8002)
Start-Process -NoNewWindow:$false -FilePath python -ArgumentList "decentralized_personality_server.py"

# Start embedding service (FastAPI, port 8003)
Start-Process -NoNewWindow:$false -FilePath python -ArgumentList "embedding_service.py"

# Start Go memory server (Gin, port 8001)
Start-Process -NoNewWindow:$false -FilePath "go" -ArgumentList "run", "memory_rest_server.go" -WorkingDirectory "./memory"

# Start monitor system for health checking
Start-Process -NoNewWindow:$false -FilePath python -ArgumentList "monitor_system.py"

# Kill any process using port 3000 before starting React UI frontend
$port = 3000
$procIds = Get-NetTCPConnection -LocalPort $port -State Listen | Select-Object -ExpandProperty OwningProcess
foreach ($procId in $procIds) { Stop-Process -Id $procId -Force }
# Start React UI frontend (esbuild, port 3000)
Start-Process -NoNewWindow:$false -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory "./ui"

Write-Host "AGI system launch initiated. All services are starting in separate terminal windows for debugging."
