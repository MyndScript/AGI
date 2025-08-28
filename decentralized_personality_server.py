"""
[Memory]: FastAPI-based decentralized personality server for AGI.
Exposes endpoints for user personality traits, mood, and context. Ready for P2P/libp2p integration.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
 # removed unused import 'json'
import os
from agi_api_client import AGIAPIClient

app = FastAPI()
# Centralized API client configuration
MEMORY_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8002")
PERSONALITY_URL = os.getenv("PERSONALITY_SERVER_URL", "http://localhost:8002")
GLOBAL_URL = os.getenv("GLOBAL_SERVER_URL", "http://localhost:8003")
agi_client = AGIAPIClient(MEMORY_URL, PERSONALITY_URL, GLOBAL_URL)


class AddObservationRequest(BaseModel):
    user_id: str
    observation: str

@app.post("/add-observation")
def add_observation(body: AddObservationRequest):
    """
    Add an interaction as an observation node for a user (for MCP memory graph integration).
    """
    user_id = body.user_id
    observation = body.observation
    if not user_id or not observation:
        return {"success": False, "error": "Missing user_id or observation"}
    resp = agi_client.add_observation(user_id, observation)
    return resp


class SetPersonalityRequest(BaseModel):
    user_id: str
    traits: dict = {}
    mood_vector: dict = {}
    archetype: str = "guardian"

@app.post("/set-personality")
def set_personality(body: SetPersonalityRequest):
    user_id = body.user_id
    traits = body.traits
    mood_vector = body.mood_vector
    archetype = body.archetype
    if not user_id:
        return {"success": False, "error": "Missing user_id"}
    resp = agi_client.set_personality(user_id, traits, mood_vector, archetype)
    return resp


class GetPersonalityRequest(BaseModel):
    user_id: str

@app.post("/get-personality")
def get_personality(body: GetPersonalityRequest):
    user_id = body.user_id
    resp = agi_client.get_personality(user_id)
    return resp


class GetPersonalityContextRequest(BaseModel):
    user_id: str

@app.post("/get-personality-context")
def get_personality_context(body: GetPersonalityContextRequest):
    user_id = body.user_id
    resp = agi_client.get_personality(user_id)
    context = f"Archetype: {resp.get('archetype', '')}\n"
    context += "Traits: " + ", ".join(f"{k}: {v}" for k, v in resp.get("traits", {}).items()) + "\n"
    context += "Mood Vector: " + ", ".join(f"{k}: {v}" for k, v in resp.get("mood_vector", {}).items()) + "\n"
    return {"context": context}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
