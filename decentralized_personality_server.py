"""
[Memory]: FastAPI-based decentralized personality server for AGI.
Exposes endpoints for user personality traits, mood, and context. Ready for P2P/libp2p integration.
"""

from fastapi import FastAPI, Request
import uvicorn
import json
import os

app = FastAPI()
personality_data = {}

@app.post("/add-observation")
async def add_observation(request: Request):
    """
    Add an interaction as an observation node for a user (for MCP memory graph integration).
    """
    data = await request.json()
    user_id = data.get("user_id")
    observation = data.get("observation")
    if not user_id or not observation:
        return {"success": False, "error": "Missing user_id or observation"}
    # Store observation in personality_data (extend for MCP graph integration)
    if user_id not in personality_data:
        personality_data[user_id] = {}
    if "observations" not in personality_data[user_id]:
        personality_data[user_id]["observations"] = []
    personality_data[user_id]["observations"].append(observation)
    return {"success": True}

@app.post("/set-personality")
async def set_personality(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    traits = data.get("traits", {})
    mood_vector = data.get("mood_vector", {})
    archetype = data.get("archetype", "guardian")
    if user_id:
        personality_data[user_id] = {
            "traits": traits,
            "mood_vector": mood_vector,
            "archetype": archetype
        }
        return {"success": True}
    return {"success": False, "error": "Missing user_id"}

@app.post("/get-personality")
async def get_personality(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    result = personality_data.get(user_id, {})
    return result

@app.post("/get-personality-context")
async def get_personality_context(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    p = personality_data.get(user_id, {})
    context = f"Archetype: {p.get('archetype', '')}\n"
    context += "Mood Vector: " + ", ".join(f"{k}: {v}" for k, v in p.get("mood_vector", {}).items()) + "\n"
    context += "Traits: " + ", ".join(f"{k}: {v}" for k, v in p.get("traits", {}).items()) + "\n"
    return {"context": context}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
