from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
import json

# Pydantic models based on your proto definitions
class SetRequest(BaseModel):
    user_id: str
    key: str
    value: str

class SetResponse(BaseModel):
    success: bool

class GetRequest(BaseModel):
    user_id: str
    key: str

class GetResponse(BaseModel):
    value: Optional[str] = None
    found: bool

class UserContextRequest(BaseModel):
    user_id: str

class UserContextResponse(BaseModel):
    context: str

class Moment(BaseModel):
    id: str
    user_id: str
    summary: str
    emotion: str
    glyph: str
    tags: List[str]
    timestamp: int
    embedding: str

class SetMomentRequest(BaseModel):
    moment: Moment

class GetMomentsRequest(BaseModel):
    user_id: str
    tags: Optional[List[str]] = None
    since: Optional[int] = None
    until: Optional[int] = None

class GetMomentsResponse(BaseModel):
    moments: List[Moment]

class SemanticSearchRequest(BaseModel):
    user_id: str
    query: str

class SemanticSearchResponse(BaseModel):
    results: List[Moment]

# In-memory store for demonstration
memory_store = {}
moments_store = {}
user_contexts = {}

@app.post("/set", response_model=SetResponse)
def set_memory(req: SetRequest):
    memory_store[(req.user_id, req.key)] = req.value
    return SetResponse(success=True)

@app.post("/get", response_model=GetResponse)
def get_memory(req: GetRequest):
    value = memory_store.get((req.user_id, req.key))
    found = value is not None
    return GetResponse(value=value, found=found)

@app.post("/user_context", response_model=UserContextResponse)
def get_user_context(req: UserContextRequest):
    # Build context from memory_store
    items = [(k, v) for (uid, k), v in memory_store.items() if uid == req.user_id]
    count = len(items)
    context = f"Total memory items: {count}\n"
    for k, v in items:
        context += f"{k}: {v}\n"
    return UserContextResponse(context=context)
@app.post("/dump_memory")
def dump_memory(req: UserContextRequest):
    # Return all key-value pairs for a user
    mem = {k: v for (uid, k), v in memory_store.items() if uid == req.user_id}
    return {"memory": mem}

@app.post("/set_moment", response_model=SetResponse)
def set_moment(req: SetMomentRequest):
    moments_store.setdefault(req.moment.user_id, []).append(req.moment)
    return SetResponse(success=True)

@app.post("/get_moments", response_model=GetMomentsResponse)
def get_moments(req: GetMomentsRequest):
    moments = moments_store.get(req.user_id, [])
    # Filter by tags, since, until if provided
    if req.tags:
        moments = [m for m in moments if set(req.tags).intersection(set(m.tags))]
    if req.since:
        moments = [m for m in moments if m.timestamp >= req.since]
    if req.until:
        moments = [m for m in moments if m.timestamp <= req.until]
    return GetMomentsResponse(moments=moments)

@app.post("/semantic_search", response_model=SemanticSearchResponse)
def semantic_search(req: SemanticSearchRequest):
    # Dummy search: return all moments for user
    moments = moments_store.get(req.user_id, [])
    return SemanticSearchResponse(results=moments)
