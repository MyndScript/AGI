
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
sys.path.append(".")
from personality.personality import Personality

import zipfile
import tempfile
app = FastAPI()
@app.post("/import-facebook")
async def import_facebook(file: UploadFile = File(...), user_id: str = Form("default")):
    """
    Accepts a Facebook JSON or ZIP file, parses posts, and stores them in user memory.
    """
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    posts = []
    # If ZIP, extract and find posts JSON
    filename = getattr(file, 'filename', '') or ''
    if filename and filename.lower().endswith('.zip'):
        with zipfile.ZipFile(tmp_path, 'r') as z:
            for name in z.namelist():
                if 'your_posts.json' in name:
                    with z.open(name) as f:
                        import json
                        data = json.load(f)
                        posts = data.get('posts', [])
                    break
    else:
        # Assume JSON file
        import json
        with open(tmp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            posts = data.get('posts', [])

    # Store posts in user memory
    stored_count = 0
    for post in posts:
        content = post.get('data', [{}])[0].get('post')
        timestamp = post.get('timestamp')
        if content:
            key = f"fb_post_{timestamp}"
            value = content
            # Store in memory server
            try:
                await user_memory_client.set(user_id, key, value)
                stored_count += 1
            except Exception:
                pass

    return {"success": True, "posts_stored": stored_count}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
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

    # Update personality and memory
    personality.add_interaction({"prompt": prompt, "response": reply})
    await user_memory_client.set(user_id, "last_response", reply)

    return {"response": reply}

# Legacy endpoint (optional)
@app.post("/text-generation")
async def text_generation(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return {"generated_text": result[0]["generated_text"]}

if __name__ == "__main__":
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