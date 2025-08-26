

"""
[Memory]: FastAPI-based decentralized agent server for AGI orchestration.
Exposes endpoints for text generation, memory, and context. Ready for P2P/libp2p integration.
"""


import sys
import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form
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
async def import_facebook(file: UploadFile = File(...), user_id: str = Form("default")):
    """
    Accepts a Facebook JSON or ZIP file, parses posts, and stores them in user memory.
    """
    logging.info(f"[import_facebook] Received upload for user_id={user_id}, filename={getattr(file, 'filename', '')}")
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file_bytes = await file.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name
    logging.info(f"[import_facebook] File saved to temp: {tmp_path}, size={len(file_bytes)} bytes")

    import json
    posts = []
    filename = getattr(file, 'filename', '') or ''
    # Load JSON file (filtered_posts format)
    with open(tmp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            posts = data
        elif isinstance(data, dict):
            posts = data.get('posts', [])

    # Remove duplicate content
    seen = set()
    unique_posts = []
    for post in posts:
        content = post.get('content') if isinstance(post, dict) else None
        if content and content not in seen:
            seen.add(content)
            unique_posts.append(post)

    # Store posts in user memory
    stored_count = 0
    for idx, post in enumerate(unique_posts):
        content = post.get('content') if isinstance(post, dict) else None
        if content:
            key = f"fb_post_{idx}"
            value = content
            try:
                await user_memory_client.set(user_id, key, value)
                stored_count += 1
            except Exception as e:
                logging.error(f"[import_facebook] Error storing post: {e}")
    logging.info(f"[import_facebook] Stored {stored_count} posts for user_id={user_id}")
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
        self.client = httpx.AsyncClient()

    async def get_user_context(self, user_id):
        resp = await self.client.post(f"{self.base_url}/get-user-context", json={"user_id": user_id})
        data = resp.json()
        return data.get("context", "")

    async def set(self, user_id, key, value):
        resp = await self.client.post(f"{self.base_url}/set-memory", json={"user_id": user_id, "key": key, "value": value})
        data = resp.json()
        return data.get("success", False)

    async def get(self, user_id, key):
        resp = await self.client.post(f"{self.base_url}/get-memory", json={"user_id": user_id, "key": key})
        data = resp.json()
        return data.get("value") if data.get("found") else None

user_memory_client = UserMemoryClient()

# Main endpoint for frontend
@app.options("/generate")
async def options_generate():
    return JSONResponse(status_code=200, content={})
@app.post("/get-observation")
async def get_observation(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    # For demonstration, return last 20 memory keys and values
    context = await user_memory_client.get_user_context(user_id)
    lines = context.splitlines()
    history = []
    for line in lines:
        if ": " in line:
            k, v = line.split(": ", 1)
            history.append({"key": k, "value": v})
    return {"observation_history": history[-20:]}

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    logging.info(f"[generate] Received: {data}")
    prompt = data.get("prompt", "")
    user_id = data.get("user_id", "default")

    # Load personality and context
    personality = Personality(user_id)
    traits = personality.traits
    mood = personality.mood_vector
    archetype = personality.archetype
    recent_interactions = personality.interaction_history[-5:]
    mem = await user_memory_client.get_user_context(user_id)

    # --- Weighted Response Fusion ---
    # 1. Personalized response
    personalized_reply = personality.personalized_response(prompt)

    # 2. Global emotional wisdom (anonymized)
    # For demo: use archetype phrase as global fallback
    global_reply = personality.archetype_phrase(archetype)

    # 3. Dynamic alpha calculation
    context_richness = len(recent_interactions)
    mood_certainty = max(mood.values()) if mood else 0.5
    archetype_match = 1.0 if archetype in personality.ARCHETYPE_TEMPLATES else 0.5
    # Simple heuristic: more context/mood/archetype = higher alpha
    alpha = min(1.0, 0.4 + 0.2 * context_richness + 0.2 * mood_certainty + 0.2 * archetype_match)
    alpha = max(0.0, min(1.0, alpha))

    # 4. Fuse responses
    def fuse_responses(personal, global_, alpha):
        if not personal:
            return global_
        if not global_:
            return personal
        # Weighted fusion: for now, interpolate textually
        if alpha > 0.8:
            return personal
        elif alpha < 0.3:
            return global_
        else:
            return f"{personal} (Also: {global_})"

    reply = fuse_responses(personalized_reply, global_reply, alpha)
    logging.info(f"[generate] Reply: {reply}")

    # Update personality and memory
    personality.add_interaction({"prompt": prompt, "response": reply})
    await user_memory_client.set(user_id, "last_response", reply)

    return {"response": reply}

# Legacy endpoint (optional)
@app.post("/text-generation")
async def text_generation(request: Request):
    data = await request.json()
    logging.info(f"[text_generation] Received: {data}")
    prompt = data.get("prompt", "")
    result = generator(prompt, max_length=100, num_return_sequences=1)
    logging.info(f"[text_generation] Generated: {result}")
    return {"generated_text": result[0]["generated_text"]}


# Facebook summary endpoint
from fastapi import Body
@app.post("/facebook-summary")
async def facebook_summary(user_id: str = Body("default")):
    posts = []
    # Try to fetch last 20 posts
    for i in range(20):
        key = f"fb_post_{i}"
        value = await user_memory_client.get(user_id, key)
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
@app.post("/abigail/trigger")
async def abigail_trigger(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    action = data.get("action")  # add, remove, list
    word = data.get("word")
    effect = data.get("effect")
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
async def abigail_glyph(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    query = data.get("query")
    personality = Personality(user_id)
    glyph = personality.recall_glyph(query)
    return {"glyph": glyph}

@app.post("/abigail/mood")
async def abigail_mood(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    personality = Personality(user_id)
    if "decay" in data:
        personality.mood_decay()
    return {"mood_vector": personality.mood_vector}

@app.post("/abigail/archetype")
async def abigail_archetype(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    archetypes = data.get("archetypes")
    personality = Personality(user_id)
    if archetypes:
        personality.fuse_archetypes(*archetypes)
    return {"archetype": personality.archetype}

@app.post("/abigail/memory")
async def abigail_memory(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    context = await user_memory_client.get_user_context(user_id)
    return {"user_context": context}