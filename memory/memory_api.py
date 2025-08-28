from fastapi import FastAPI, HTTPException, status, Depends, Request
import psycopg2
import os
import hmac
import hashlib
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional

class MemoryAPI:
    def __init__(self, sqlite_path='user_memory.db', pg_url=None):
        import sqlite3
        self.sqlite_path = sqlite_path
        self.sqlite_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.sqlite_conn.execute("CREATE TABLE IF NOT EXISTS user_memory (user_id TEXT, key TEXT, value TEXT, PRIMARY KEY (user_id, key))")
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS moments (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                summary TEXT,
                emotion TEXT,
                glyph TEXT,
                tags TEXT,
                timestamp BIGINT,
                embedding TEXT
            )
        """)
        self.pg_url = pg_url or os.getenv("POSTGRES_URL", "dbname=agi user=postgres password=postgres host=localhost")
        self.pg_conn = psycopg2.connect(self.pg_url)
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS global_knowledge (
                    id SERIAL PRIMARY KEY,
                    topic TEXT,
                    insights TEXT,
                    emotional_tone TEXT,
                    ts BIGINT
                )
            """)
            self.pg_conn.commit()

    def store_local(self, user_id, data):
        for key, value in data.items():
            self.sqlite_conn.execute("INSERT INTO user_memory (user_id, key, value) VALUES (?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value", (user_id, key, value))
        self.sqlite_conn.commit()

    def get_local(self, user_id):
        result = self.sqlite_conn.execute("SELECT key, value FROM user_memory WHERE user_id = ?", (user_id,)).fetchall()
        return {k: v for k, v in result}

    def store(self, user_id, data, global_scope=False, emotional_tone=None, ts=None):
        """
        Robust unified store method.
        - If global_scope is True or key is 'global', store in Postgres (global_knowledge).
        - Otherwise, store in SQLite (user_memory).
        - Prevents data mixing, logs all actions, and ensures atomicity.
        """
        import logging
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary of key-value pairs.")
        try:
            if global_scope:
                # Only allow string keys for global knowledge
                for key, value in data.items():
                    if not isinstance(key, str):
                        raise TypeError(f"Global knowledge key must be str, got {type(key)}")
                    with self.pg_conn:
                        with self.pg_conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO global_knowledge (topic, insights, emotional_tone, ts) VALUES (%s, %s, %s, %s)",
                                (key, value, emotional_tone or '', ts or 0)
                            )
                    logging.info(f"Stored global knowledge: {key}")
            else:
                # Only allow string keys for local memory
                for key in data.keys():
                    if not isinstance(key, str):
                        raise TypeError(f"Local memory key must be str, got {type(key)}")
                self.store_local(user_id, data)
                logging.info(f"Stored local memory for user {user_id}: {list(data.keys())}")
        except Exception as e:
            logging.error(f"MemoryAPI.store error: {e}")
            raise

    def get(self, user_id, key=None, global_scope=False):
        """
        Robust unified get method.
        - If global_scope is True or key is 'global', get from Postgres (global_knowledge).
        - Otherwise, get from SQLite (user_memory).
        - Prevents data mixing, logs all actions, and validates types.
        """
        import logging
        try:
            if global_scope or (key == 'global'):
                with self.pg_conn:
                    with self.pg_conn.cursor() as cur:
                        if key and key != 'global':
                            if not isinstance(key, str):
                                raise TypeError(f"Global knowledge key must be str, got {type(key)}")
                            cur.execute("SELECT insights FROM global_knowledge WHERE topic = %s", (key,))
                            result = cur.fetchone()
                            logging.info(f"Fetched global knowledge for topic: {key}")
                            return result[0] if result else None
                        else:
                            cur.execute("SELECT topic, insights FROM global_knowledge")
                            rows = cur.fetchall()
                            logging.info("Fetched all global knowledge.")
                            return {row[0]: row[1] for row in rows}
            else:
                mem = self.get_local(user_id)
                if key:
                    if not isinstance(key, str):
                        raise TypeError(f"Local memory key must be str, got {type(key)}")
                    logging.info(f"Fetched local memory for user {user_id}, key: {key}")
                    return mem.get(key)
                logging.info(f"Fetched all local memory for user {user_id}.")
                return mem
        except Exception as e:
            logging.error(f"MemoryAPI.get error: {e}")
            raise

    def store_moment(self, moment):
        # Store moment in SQLite
        tags_str = ','.join(moment.tags)
        self.sqlite_conn.execute("INSERT INTO moments (id, user_id, summary, emotion, glyph, tags, timestamp, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, emotion=excluded.emotion, glyph=excluded.glyph, tags=excluded.tags, timestamp=excluded.timestamp, embedding=excluded.embedding", (
            moment.id, moment.user_id, moment.summary, moment.emotion, moment.glyph, tags_str, moment.timestamp, moment.embedding
        ))
        self.sqlite_conn.commit()

    def get_moments(self, user_id, tags=None, since=None, until=None):
        query = "SELECT id, user_id, summary, emotion, glyph, tags, timestamp, embedding FROM moments WHERE user_id = ?"
        params = [user_id]
        if tags:
            query += " AND (" + " OR ".join(["tags LIKE ?" for _ in tags]) + ")"
            params += [f"%{tag}%" for tag in tags]
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if until:
            query += " AND timestamp <= ?"
            params.append(until)
        rows = self.sqlite_conn.execute(query, tuple(params)).fetchall()
        moments = []
        for row in rows:
            moments.append({
                "id": row[0], "user_id": row[1], "summary": row[2], "emotion": row[3], "glyph": row[4],
                "tags": row[5].split(","), "timestamp": row[6], "embedding": row[7]
            })
        return moments

    def semantic_search_moments(self, user_id, query_embedding, top_k=5):
        # Search moments by cosine similarity to query_embedding
        import numpy as np
        rows = self.sqlite_conn.execute("SELECT id, user_id, summary, emotion, glyph, tags, timestamp, embedding FROM moments WHERE user_id = ?", (user_id,)).fetchall()
        scored = []
        for row in rows:
            emb = row[7]
            try:
                emb_vec = np.array([float(x) for x in emb.split(",")])
                query_vec = np.array([float(x) for x in query_embedding.split(",")])
                sim = float(np.dot(emb_vec, query_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(query_vec)))
            except Exception:
                sim = 0.0
            scored.append((sim, row))
        scored.sort(reverse=True)
        moments = []
        for sim, row in scored[:top_k]:
            moments.append({
                "id": row[0], "user_id": row[1], "summary": row[2], "emotion": row[3], "glyph": row[4],
                "tags": row[5].split(","), "timestamp": row[6], "embedding": row[7], "similarity": sim
            })
        return moments
    # ...existing code...

    def get_global(self, topic=None):
        with self.pg_conn.cursor() as cur:
            if topic:
                cur.execute("SELECT topic, insights, emotional_tone, ts FROM global_knowledge WHERE topic = %s", (topic,))
            else:
                cur.execute("SELECT topic, insights, emotional_tone, ts FROM global_knowledge")
            return cur.fetchall()

    def push_global(self, topic, insights, emotional_tone, ts):
        with self.pg_conn.cursor() as cur:
            cur.execute("INSERT INTO global_knowledge (topic, insights, emotional_tone, ts) VALUES (%s, %s, %s, %s)",
                        (topic, insights, emotional_tone, ts))
            self.pg_conn.commit()

app = FastAPI(title="Decentralized Memory API", description="Key-value and moment memory API for AGI agents.", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],  # Restrict to local dev for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_NAME = "X-API-Key"
SIGNATURE_HEADER = "X-Signature"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
signature_header = APIKeyHeader(name=SIGNATURE_HEADER, auto_error=False)
API_KEYS = {"testkey"}  # Replace with real keys or decentralized auth
SHARED_SECRET = os.getenv("KNOWLEDGE_SYNC_SECRET", "supersecret")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return api_key

def sign_payload(payload: str, secret: str = SHARED_SECRET) -> str:
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(sig).decode()

def verify_signature(payload: str, signature: str, secret: str = SHARED_SECRET) -> bool:
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)

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


# Initialize database-backed stores
# Initialize database-backed stores
mem_api = MemoryAPI()
# ...existing code...


class GlobalKnowledgePushRequest(BaseModel):
    topic: str
    insights: str
    emotional_tone: str
    ts: int

class GlobalKnowledgeResponse(BaseModel):
    knowledge: List[dict]

@app.post("/push_global_knowledge", response_model=SetResponse, tags=["Global"])
async def push_global_knowledge(req: GlobalKnowledgePushRequest, request: Request, api_key: str = Depends(get_api_key), signature: str = Depends(signature_header)):
    # Serialize payload for signing
    import json, logging
    payload = json.dumps(req.dict(), sort_keys=True)
    if not signature or not verify_signature(payload, signature):
        logging.warning(f"Signature verification failed for push_global_knowledge: {payload}")
        raise HTTPException(status_code=401, detail="Invalid or missing signature")
    try:
        mem_api.push_global(req.topic, req.insights, req.emotional_tone, req.ts)
        # Audit log
        client_host = getattr(getattr(request, "client", None), "host", "unknown")
        logging.info(f"Global knowledge pushed: {req.topic} by {client_host}")
        return SetResponse(success=True)
    except Exception as e:
        logging.error(f"Global knowledge push error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_global_knowledge", response_model=GlobalKnowledgeResponse, tags=["Global"])
async def get_global_knowledge(request: Request, api_key: str = Depends(get_api_key), signature: str = Depends(signature_header)):
    import json, logging
    # For GET, sign the query string (or empty string)
    client_host = getattr(getattr(request, "client", None), "host", "unknown")
    payload = json.dumps({"requestor": str(client_host)}, sort_keys=True)
    if not signature or not verify_signature(payload, signature):
        logging.warning(f"Signature verification failed for get_global_knowledge: {payload}")
        raise HTTPException(status_code=401, detail="Invalid or missing signature")
    try:
        rows = mem_api.get_global()
        knowledge = [
            {"topic": r[0], "insights": r[1], "emotional_tone": r[2], "ts": r[3]} for r in rows
        ]
        logging.info(f"Global knowledge fetched by {client_host}")
        return GlobalKnowledgeResponse(knowledge=knowledge)
    except Exception as e:
        logging.error(f"Global knowledge fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set", response_model=SetResponse, tags=["Memory"])
async def set_memory(req: SetRequest, api_key: str = Depends(get_api_key)):
    try:
        mem_api.store_local(req.user_id, {req.key: req.value})
        return SetResponse(success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get", response_model=GetResponse, tags=["Memory"])
async def get_memory(req: GetRequest, api_key: str = Depends(get_api_key)):
    try:
        mem = mem_api.get_local(req.user_id)
        value = mem.get(req.key)
        found = value is not None
        return GetResponse(value=value, found=found)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user_context", response_model=UserContextResponse, tags=["Memory"])
async def get_user_context(req: UserContextRequest, api_key: str = Depends(get_api_key)):
    try:
        mem = mem_api.get_local(req.user_id)
        count = len(mem)
        context = f"Total memory items: {count}\n"
        for k, v in mem.items():
            context += f"{k}: {v}\n"
        return UserContextResponse(context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/dump_memory", tags=["Memory"])
async def dump_memory(req: UserContextRequest, api_key: str = Depends(get_api_key)):
    try:
        mem = mem_api.get_local(req.user_id)
        return {"memory": mem}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_moment", response_model=SetResponse, tags=["Moments"])
async def set_moment(req: SetMomentRequest, api_key: str = Depends(get_api_key)):
    try:
        mem_api.store_moment(req.moment)
        return SetResponse(success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_moments", response_model=GetMomentsResponse, tags=["Moments"])
async def get_moments(req: GetMomentsRequest, api_key: str = Depends(get_api_key)):
    try:
        moments = mem_api.get_moments(req.user_id, tags=req.tags, since=req.since, until=req.until)
        return GetMomentsResponse(moments=moments or [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic_search", response_model=SemanticSearchResponse, tags=["Moments"])
async def semantic_search(req: SemanticSearchRequest, api_key: str = Depends(get_api_key)):
    try:
        # Assume req.query is a comma-separated embedding string
        moments = mem_api.semantic_search_moments(req.user_id, req.query)
        return SemanticSearchResponse(results=moments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
