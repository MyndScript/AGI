
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
import httpx


"""
[Memory]: FastAPI-based decentralized memory server for AGI.
Overseer-compatible: All API calls routed through overseer gateway (port 8010).
Exposes endpoints for user memory storage and retrieval. Ready for P2P/libp2p integration.
"""


from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# All API client calls routed through overseer gateway
OVERSEER_PORT = int(os.getenv("AGI_OVERSEER_PORT", 8010))
OVERSEER_URL = f"http://localhost:{OVERSEER_PORT}"
GO_MEMORY_URL = f"http://localhost:8001"

# Restrict CORS to overseer gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8010"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Proxy semantic search to Go server
@app.post("/semantic_search")
async def semantic_search(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GO_MEMORY_URL}/semantic_search", json=data)
        return JSONResponse(resp.json())


# Proxy set-memory to Go server
@app.post("/set-memory")
async def set_memory(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GO_MEMORY_URL}/set-memory", json=data)
        return JSONResponse(resp.json())


# Proxy get-memory to Go server
@app.post("/get-memory")
async def get_memory(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GO_MEMORY_URL}/get-memory", json=data)
        return JSONResponse(resp.json())


# Proxy get-user-context to Go server
@app.post("/get-user-context")
async def get_user_context(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GO_MEMORY_URL}/get-user-context", json=data)
        return JSONResponse(resp.json())
# Proxy store-post to Go server
@app.post("/store-post")
async def store_post(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GO_MEMORY_URL}/store-post", json=data)
        return JSONResponse(resp.json())

# Proxy get-posts to Go server
@app.post("/get-posts")
async def get_posts(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GO_MEMORY_URL}/get-posts", json=data)
        return JSONResponse(resp.json())

# Proxy store-journal-entry to Go server
@app.post("/store-journal-entry")
async def store_journal_entry(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GO_MEMORY_URL}/store-journal-entry", json=data)
        return JSONResponse(resp.json())

# Proxy get-journal-entries to Go server
@app.post("/get-journal-entries")
async def get_journal_entries(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GO_MEMORY_URL}/get-journal-entries", json=data)
        return JSONResponse(resp.json())

if __name__ == "__main__":
    print("[AGI Memory Server] Starting on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
