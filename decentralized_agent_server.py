def load_facebook_posts(tmp_path):
    import json
    with open(tmp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('posts', [])
        return []

def dedupe_posts(posts):
    seen = set()
    unique_posts = []
    for post in posts:
        content = post.get('content') if isinstance(post, dict) else None
        if content and content not in seen:
            seen.add(content)
            unique_posts.append(post)
    return unique_posts

def privacy_analysis(posts):
    import re
    private_candidates = []
    public_candidates = []
    for post in posts:
        raw_content = post.get('content') if isinstance(post, dict) else ''
        content = raw_content if isinstance(raw_content, str) else ''
        if re.search(r'\b(email|@|gmail|yahoo|hotmail|phone|address|\d{3}-\d{3}-\d{4}|\d{10}|\d{5}|street|avenue|road|drive|lane|blvd|Dr\.|Mr\.|Ms\.|Mrs\.|\b[A-Z][a-z]+\b)', content):
            private_candidates.append(post)
        else:
            scrubbed = re.sub(r'([A-Z][a-z]+|@\S+|\d{3}-\d{3}-\d{4}|\d{10}|\d{5}|street|avenue|road|drive|lane|blvd|Dr\.|Mr\.|Ms\.|Mrs\.)', '[REDACTED]', content)
            public_candidates.append({'content': scrubbed})
    return private_candidates, public_candidates

def store_private_posts(user_id, private_candidates):
    stored_count = 0
    for idx, post in enumerate(private_candidates):
        content = post.get('content') if isinstance(post, dict) else None
        if content:
            key = f"fb_post_{idx}"
            value = content
            try:
                user_memory_client.set(user_id, key, value)
                stored_count += 1
            except Exception as e:
                logging.error(f"[import_facebook] Error storing post: {e}")
    logging.info(f"[import_facebook] Stored {stored_count} private posts for user_id={user_id}")
    return stored_count


"""
[Memory]: FastAPI-based decentralized agent server for AGI orchestration.
Exposes endpoints for text generation, memory, and context. Ready for P2P/libp2p integration.
"""


import sys
import httpx
from fastapi import  FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn
import subprocess
def clear_port(port):
    # Windows: use netstat and taskkill
    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, shell=False)
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if f":{port}" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit():
                    subprocess.run(["taskkill", "/PID", pid, "/F"], shell=False)
    except Exception as e:
        print(f"Error clearing port {port}: {e}")
import logging
sys.path.append(".")
from personality.personality import Personality

import tempfile
import tempfile
from fastapi.responses import JSONResponse
app = FastAPI()
logging.basicConfig(level=logging.INFO)
@app.post("/import-facebook")
def import_facebook(file: UploadFile = File(...), user_id: str = Form("default")):
    """
    Accepts a Facebook JSON or ZIP file, parses posts, and stores them in user memory.
    """
    logging.info(f"[import_facebook] Received upload for user_id={user_id}, filename={getattr(file, 'filename', '')}")
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file_bytes = file.file.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name
    logging.info(f"[import_facebook] File saved to temp: {tmp_path}, size={len(file_bytes)} bytes")

    posts = load_facebook_posts(tmp_path)
    unique_posts = dedupe_posts(posts)
    private_candidates, public_candidates = privacy_analysis(unique_posts)
    stored_count = store_private_posts(user_id, private_candidates)
    return {
        "success": True,
        "posts_stored": stored_count,
        "private_count": len(private_candidates),
        "public_count": len(public_candidates),
        "public_candidates": public_candidates
    }
# Endpoint to accept scrubbed posts for public/global learning
@app.post("/import-facebook-public")
def import_facebook_public(payload: dict):
    """
    Accepts scrubbed posts for public/global learning (Postgres).
    """
    user_id = payload.get('user_id', 'default')
    posts = payload.get('posts', [])
    stored_count = 0
    for idx, post in enumerate(posts):
        content = post.get('content') if isinstance(post, dict) else None
        if content:
            key = f"fb_public_post_{idx}"
            value = content
            try:
                # Store in global knowledge (Postgres)
                # Use topic as key, insights as value
                user_memory_client.set(user_id, key, value)  # Optionally, use a global API if available
                stored_count += 1
            except Exception as e:
                logging.error(f"[import_facebook_public] Error storing public post: {e}")
    logging.info(f"[import_facebook_public] Stored {stored_count} public posts for global learning")
    return {"success": True, "posts_stored": stored_count}
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # UI dev server
    "http://127.0.0.1:5173",  # UI dev server
    "http://localhost:3000",  # UI dev server
    "http://127.0.0.1:3000",  # UI dev server
    "http://192.168.1.38:3000",  # Your LAN IP
    "http://192.168.1.38:63620", # UI alternate port
    # Add production domains here
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
generator = pipeline("text-generation", model="gpt2")

MEMORY_API_URL = "http://127.0.0.1:8001"

class UserMemoryClient:

    def __init__(self, base_url=MEMORY_API_URL):
        self.base_url = base_url
        self.client = httpx.Client()


    def get_user_context(self, user_id):
        resp = self.client.post(f"{self.base_url}/get-user-context", json={"user_id": user_id})
        data = resp.json()
        return data.get("context", "")


    def set(self, user_id, key, value):
        resp = self.client.post(f"{self.base_url}/set-memory", json={"user_id": user_id, "key": key, "value": value})
        data = resp.json()
        return data.get("success", False)


    def get(self, user_id, key):
        resp = self.client.post(f"{self.base_url}/get-memory", json={"user_id": user_id, "key": key})
        data = resp.json()
        return data.get("value") if data.get("found") else None

user_memory_client = UserMemoryClient()

# Main endpoint for frontend

# Main endpoint for frontend
@app.options("/generate")
def options_generate():
    return JSONResponse(status_code=200, content={})


# Pydantic model for request body

class ObservationRequest(BaseModel):
    user_id: str = "default"

class GenerateRequest(BaseModel):
    prompt: str = ""
    user_id: str = "default"

@app.post("/get-observation")
def get_observation(body: ObservationRequest):
    user_id = body.user_id
    # For demonstration, return last 20 memory keys and values
    context = user_memory_client.get_user_context(user_id)
    lines = context.splitlines()
    history = []
    for line in lines:
        if ": " in line:
            k, v = line.split(": ", 1)
            history.append({"key": k, "value": v})
    return {"observation_history": history[-20:]}


@app.post("/generate")
def generate(body: GenerateRequest):
    logging.info(f"[generate] Received: {body}")
    prompt = body.prompt
    user_id = body.user_id

    # Load personality and context
    personality = Personality(user_id)
    traits = personality.traits
    mood = personality.mood_vector
    recent_interactions = personality.interaction_history[-5:]
    mem = user_memory_client.get_user_context(user_id)

    # Semantic retrieval of relevant moments/glyphs
    with httpx.Client() as client:
        semantic_resp = client.post(f"{MEMORY_API_URL}/semantic_search", json={"user_id": user_id, "query": prompt})
        semantic_data = semantic_resp.json()
        relevant_moments = semantic_data.get("results", [])

    # Build context for response fusion
    context = {
        "traits": traits,
        "mood": mood,
        "recent_interactions": recent_interactions,
        "memory": mem,
        "relevant_moments": relevant_moments
    }
    logging.info(f"[generate] Context: {context}")

    # Pass context to personalized_response (optionally update method to accept context)
    reply = personality.personalized_response(prompt)
    logging.info(f"[generate] Final reply: {reply}")

    # Update personality and memory
    personality.add_interaction({"prompt": prompt, "response": reply})
    user_memory_client.set(user_id, "last_response", reply)

    return {"response": reply}

# Legacy endpoint (optional)

class TextGenerationRequest(BaseModel):
    prompt: str = ""

@app.post("/text-generation")
def text_generation(body: TextGenerationRequest):
    logging.info(f"[text_generation] Received: {body}")
    prompt = body.prompt
    result = generator(prompt, max_length=100, num_return_sequences=1)
    logging.info(f"[text_generation] Generated: {result}")
    return {"generated_text": result[0]["generated_text"]}


# Facebook summary endpoint
from fastapi import Body
@app.post("/facebook-summary")
def facebook_summary(user_id: str = Body("default")):
    posts = []
    # Try to fetch last 20 posts
    for i in range(20):
        key = f"fb_post_{i}"
        value = user_memory_client.get(user_id, key)
        if value:
            posts.append(value)
    if not posts:
        return {"summary": "I don't have any Facebook memories for you yet."}
    summary = "Here are some things I've learned from your Facebook posts: "
    summary += " ".join(posts[:5])
    return {"summary": summary}

if __name__ == "__main__":
    clear_port(8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Internal endpoints for Abigail's MCP toolbox (agent-only)

class AbigailTriggerRequest(BaseModel):
    user_id: str = "default"
    action: Optional[str] = None
    word: Optional[str] = None
    effect: Optional[str] = None


class AbigailGlyphRequest(BaseModel):
    user_id: str = "default"
    query: Optional[str] = None


class AbigailMoodRequest(BaseModel):
    user_id: str = "default"
    decay: bool = False


class AbigailArchetypeRequest(BaseModel):
    user_id: str = "default"
    archetypes: Optional[List[str]] = None


class AbigailMemoryRequest(BaseModel):
    user_id: str = "default"
@app.post("/abigail/trigger")
def abigail_trigger(body: AbigailTriggerRequest):
    user_id = body.user_id
    action = body.action
    word = body.word
    effect = body.effect
    personality = Personality(user_id)
    if action == "add" and word and effect:
        personality.add_trigger(word, effect)
        return {"status": "added", "triggers": personality.list_triggers()}
    elif action == "remove" and word:
        personality.remove_trigger(word)
        return {"status": "removed", "triggers": personality.list_triggers()}
    elif action == "list":
        return {"triggers": personality.list_triggers()}
    return {"error": "invalid action or missing params"}

@app.post("/abigail/glyph")
def abigail_glyph(body: AbigailGlyphRequest):
    user_id = body.user_id
    query = body.query
    personality = Personality(user_id)
    glyph = personality.recall_glyph(query)
    return {"glyph": glyph}

@app.post("/abigail/mood")
def abigail_mood(body: AbigailMoodRequest):
    user_id = body.user_id
    personality = Personality(user_id)
    if body.decay:
        personality.mood_decay()
    return {"mood_vector": personality.mood_vector}

@app.post("/abigail/archetype")
def abigail_archetype(body: AbigailArchetypeRequest):
    user_id = body.user_id
    archetypes = body.archetypes
    personality = Personality(user_id)
    if archetypes:
        personality.fuse_archetypes(*archetypes)
    return {"archetype": personality.archetype}

@app.post("/abigail/memory")
def abigail_memory(body: AbigailMemoryRequest):
    user_id = body.user_id
    context = user_memory_client.get_user_context(user_id)
    return {"user_context": context}