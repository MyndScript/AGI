"""
[Memory]: FastAPI-based decentralized agent server for AGI orchestration.
Exposes endpoints for text generation, memory, and context. Ready for P2P/libp2p integration.
"""


import sys
import grpc
import memory.pb.memory_pb2 as memory_pb2
import memory.pb.memory_pb2_grpc as memory_grpc
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn
sys.path.append(".")
from personality.personality import Personality

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://0.0.0.0",
        # Add LAN IPs if needed, e.g. "http://192.168.1.100"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
generator = pipeline("text-generation", model="gpt2")

class UserMemoryClient:
    def __init__(self, address="127.0.0.1:50051", cert_path="memory/cert.pem"):
        with open(cert_path, "rb") as f:
            creds = grpc.ssl_channel_credentials(f.read())
        self.channel = grpc.secure_channel(address, creds)
        self.stub = memory_grpc.UserMemoryServiceStub(self.channel)

    def get_user_context(self, user_id):
        req = memory_pb2.UserContextRequest(user_id=user_id)  # type: ignore[attr-defined]
        resp = self.stub.GetUserContext(req)
        return resp.context

    def set(self, user_id, key, value):
        req = memory_pb2.SetRequest(user_id=user_id, key=key, value=value)  # type: ignore[attr-defined]
        resp = self.stub.Set(req)
        return resp.success

    def get(self, user_id, key):
        req = memory_pb2.GetRequest(user_id=user_id, key=key)  # type: ignore[attr-defined]
        resp = self.stub.Get(req)
        return resp.value if resp.found else None

user_memory_client = UserMemoryClient()

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

    # Load memory from Go server
    mem = user_memory_client.get_user_context(user_id)

    # Build context
    context = f"Traits: {traits}\nMood: {mood}\nArchetype: {archetype}\nRecent: {recent_interactions}\nMemory: {mem}\n"
    full_prompt = context + "User: " + prompt

    result = generator(full_prompt, max_length=100, num_return_sequences=1)
    response = result[0]["generated_text"]

    # Update personality and memory
    personality.add_interaction({"prompt": prompt, "response": response})
    user_memory_client.set(user_id, "last_response", response)

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
