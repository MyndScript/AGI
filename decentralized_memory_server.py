"""
[Memory]: FastAPI-based decentralized memory server for AGI.
Exposes endpoints for user memory storage and retrieval. Ready for P2P/libp2p integration.
"""

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
user_memory = {}

@app.post("/set-memory")
async def set_memory(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    key = data.get("key")
    value = data.get("value")
    if user_id and key:
        user_memory.setdefault(user_id, {})[key] = value
        return {"success": True}
    return {"success": False, "error": "Missing user_id or key"}

@app.post("/get-memory")
async def get_memory(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    key = data.get("key")
    value = user_memory.get(user_id, {}).get(key, "")
    found = key in user_memory.get(user_id, {})
    return {"value": value, "found": found}

@app.post("/get-user-context")
async def get_user_context(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    context = ""
    if user_id in user_memory:
        for k, v in user_memory[user_id].items():
            context += f"{k}: {v}\n"
    return {"context": context}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
