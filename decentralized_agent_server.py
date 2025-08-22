"""
[Memory]: FastAPI-based decentralized agent server for AGI orchestration.
Exposes endpoints for text generation, memory, and context. Ready for P2P/libp2p integration.
"""


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn
import sys
sys.path.append(".")
from personality.personality import Personality

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
generator = pipeline("text-generation", model="gpt2")

# Simple in-memory user memory (replace with Go server later)
user_memory = {}

# Main endpoint for frontend
@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    user_id = data.get("user_id", "default")

    # Load personality
    personality = Personality(user_id)
    traits = personality.traits
    mood = personality.mood_vector
    archetype = personality.archetype
    recent_interactions = personality.interaction_history[-5:]

    # Load memory
    mem = user_memory.get(user_id, {})

    # Build context
    context = f"Traits: {traits}\nMood: {mood}\nArchetype: {archetype}\nRecent: {recent_interactions}\nMemory: {mem}\n"
    full_prompt = context + "User: " + prompt

    result = generator(full_prompt, max_length=100, num_return_sequences=1)
    response = result[0]["generated_text"]

    # Update personality and memory
    personality.add_interaction({"prompt": prompt, "response": response})
    user_memory[user_id] = mem  # (extend with actual updates as needed)

    return {"response": response}

# Legacy endpoint (optional)
@app.post("/text-generation")
async def text_generation(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return {"generated_text": result[0]["generated_text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
