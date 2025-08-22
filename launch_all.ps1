# PowerShell launch script for AGI system
# This script starts all FastAPI backends and the UI frontend
# Terminals are hidden (no window popups)

# Start decentralized agent server
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "decentralized_agent_server.py"

# Start decentralized memory server
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "decentralized_memory_server.py"

# Start decentralized personality server
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "decentralized_personality_server.py"

# Start React/Vite UI frontend
Start-Process -WindowStyle Hidden -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory "./ui"

Write-Host "AGI system launch initiated. All backends and UI are starting in the background."
