"""
[Memory]: FastAPI-based decentralized memory server for AGI.
Exposes endpoints for user memory storage and retrieval. Ready for P2P/libp2p integration.
"""

from fastapi import FastAPI, Request
import uvicorn
import os
from agi_api_client import AGIAPIClient

app = FastAPI()
# Centralized API client configuration
MEMORY_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8002")
PERSONALITY_URL = os.getenv("PERSONALITY_SERVER_URL", "http://localhost:8002")
GLOBAL_URL = os.getenv("GLOBAL_SERVER_URL", "http://localhost:8003")
agi_client = AGIAPIClient(MEMORY_URL, PERSONALITY_URL, GLOBAL_URL)

@app.post("/set-memory")
async def set_memory(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    key = data.get("key")
    value = data.get("value")
    if not user_id or not key:
        return {"success": False, "error": "Missing user_id or key"}
    resp = agi_client.set_memory(user_id, key, value)
    return resp

@app.post("/get-memory")
async def get_memory(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    key = data.get("key")
    resp = agi_client.get_memory(user_id, key)
    return resp

@app.post("/get-user-context")
async def get_user_context(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    resp = agi_client.get_user_context(user_id)
    return resp

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
