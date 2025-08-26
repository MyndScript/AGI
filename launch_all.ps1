# PowerShell launch script for AGI system
# This script starts all FastAPI backends and the UI frontend
# Terminals are hidden (no window popups)


# Start decentralized agent server (manual uvicorn option)
Start-Process -FilePath "uvicorn" -ArgumentList "decentralized_agent_server:app", "--reload"

# Start decentralized agent server (script option)
Start-Process -FilePath python -ArgumentList "decentralized_agent_server.py"

 # Start decentralized memory server
 Start-Process -FilePath python -ArgumentList "decentralized_memory_server.py"

 # Start decentralized personality server
 Start-Process -FilePath python -ArgumentList "decentralized_personality_server.py"

 # Start React UI frontend (esbuild, port 3000)
 Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory "./ui"

Write-Host "AGI system launch initiated. All backends and UI are starting in the background."
