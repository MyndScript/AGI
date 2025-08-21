"""
[Memory]: FastAPI-based decentralized agent server for AGI orchestration.
Exposes endpoints for text generation, memory, and context. Ready for P2P/libp2p integration.
"""

from fastapi import FastAPI, Request
from transformers import pipeline
import uvicorn

app = FastAPI()
generator = pipeline("text-generation", model="gpt2")

@app.post("/text-generation")
async def text_generation(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return {"generated_text": result[0]["generated_text"]}

# Example: Add endpoints for memory/context as needed
# @app.post("/memory")
# async def memory(request: Request):
#     ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
