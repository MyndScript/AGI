
"""
[Personality]: FastAPI-based decentralized personality server for AGI.
Overseer-compatible: All API calls routed through overseer gateway (port 8010).
Exposes endpoints for user personality traits, mood, and context. Ready for P2P/libp2p integration.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from datetime import datetime
from personality.personality import Personality
from memory.memory_api import MemoryAPIClient, ConversationAnalyzer
from typing import Dict, List, Optional, Any

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Initialize memory client for advanced analysis
memory_client = MemoryAPIClient()
conversation_analyzer = ConversationAnalyzer()

# All API client calls routed through overseer gateway
OVERSEER_PORT = int(os.getenv("AGI_OVERSEER_PORT", 8010))
OVERSEER_URL = f"http://localhost:{OVERSEER_PORT}"

# Memory-safe personality cache with LRU eviction
import threading
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def clear(self):
        with self.lock:
            self.cache.clear()

# Replace simple dict with LRU cache
personality_cache = LRUCache(capacity=50)  # Limit to 50 concurrent users

# Restrict CORS to overseer gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8010"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for the personality server"""
    cache_size = len(personality_cache.cache) if hasattr(personality_cache, 'cache') else 0
    return {
        "status": "healthy",
        "service": "personality_server",
        "port": 8002,
        "cache_size": cache_size,
        "cache_capacity": personality_cache.capacity if hasattr(personality_cache, 'capacity') else 0,
        "timestamp": datetime.now().isoformat()
    }

class BatchPersonalityUpdateRequest(BaseModel):
    user_id: str
    conversations: List[Dict[str, Any]]
    analysis_type: Optional[str] = "comprehensive"  # "comprehensive", "quick", "emotional"

class PersonalityInsightsRequest(BaseModel):
    user_id: str
    include_global_comparison: bool = True
    include_suggestions: bool = True

class SimilarUsersRequest(BaseModel):
    user_id: str
    trait: Optional[str] = None
    limit: int = 5

@app.post("/batch-personality-update")
def batch_personality_update(request: BatchPersonalityUpdateRequest):
    """Batch process personality updates for efficiency"""
    try:
        user_id = request.user_id
        conversations = request.conversations
        analysis_type = request.analysis_type

        logging.info(f"[batch_personality_update] Processing {len(conversations)} conversations for user {user_id}")

        # Use memory client's batch processing
        memory_client.update_personality_batch(user_id, conversations)

        # Get updated personality analysis
        analysis = conversation_analyzer.analyze_conversation(conversations)

        return {
            "success": True,
            "user_id": user_id,
            "conversations_processed": len(conversations),
            "analysis_type": analysis_type,
            "personality_insights": analysis,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"[batch_personality_update] Error: {e}")
        return {"success": False, "error": str(e)}, 500

@app.post("/get-personality-insights")
def get_personality_insights(request: PersonalityInsightsRequest):
    """Get comprehensive personality insights with global comparison"""
    try:
        user_id = request.user_id

        # Get user's full context from memory system
        user_context = memory_client.get_user_context(user_id)

        insights = {
            "user_id": user_id,
            "personality_profile": user_context.get("advanced_analysis", {}),
            "basic_personality": user_context.get("personality_data", {}),
            "timestamp": datetime.now().isoformat()
        }

        if request.include_global_comparison:
            # Get global insights for comparison
            global_insights = memory_client.get_global_insights()
            insights["global_comparison"] = global_insights

        if request.include_suggestions:
            # Get personalized response suggestions (placeholder)
            # This method would need to be implemented in MemoryAPIClient
            insights["response_suggestions"] = ["Use empathetic tone", "Ask follow-up questions"]

        return insights

    except Exception as e:
        logging.error(f"[get_personality_insights] Error: {e}")
        return {"success": False, "error": str(e)}, 500

@app.post("/find-similar-users")
def find_similar_users(request: SimilarUsersRequest):
    """Find users with similar personality profiles"""
    try:
        user_id = request.user_id
        trait = request.trait
        limit = request.limit

        similar_users = memory_client.get_similar_users(user_id, trait, limit)

        return {
            "user_id": user_id,
            "trait_filter": trait,
            "similar_users": similar_users,
            "count": len(similar_users),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"[find_similar_users] Error: {e}")
        return {"success": False, "error": str(e)}, 500

@app.post("/analyze-conversation")
def analyze_conversation_endpoint(request: BatchPersonalityUpdateRequest):
    """Standalone conversation analysis endpoint"""
    try:
        conversations = request.conversations

        # Perform comprehensive analysis
        analysis = conversation_analyzer.analyze_conversation(conversations)

        return {
            "success": True,
            "analysis": analysis,
            "conversations_analyzed": len(conversations),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"[analyze_conversation] Error: {e}")
        return {"success": False, "error": str(e)}, 500
    personality.add_interaction({"observation": observation})
    logging.info(f"[add_observation] observation added successfully")
    return {"success": True}


class GetPersonalityRequest(BaseModel):
    user_id: str

@app.post("/get-personality")
def get_personality(body: GetPersonalityRequest):
    user_id = body.user_id
    logging.info(f"[get_personality] user_id={user_id}")
    if not user_id:
        logging.error("Missing user_id in get_personality")
        return {"success": False, "error": "Missing user_id"}, 422

    # Handle personality retrieval directly
    personality = personality_cache.get(user_id)
    if personality is None:
        personality = Personality(user_id)
        personality_cache.put(user_id, personality)

    resp = {
        "archetype": personality.archetype,
        "traits": personality.traits,
        "mood_vector": personality.mood_vector
    }
    logging.info(f"[get_personality] response={resp}")
    return resp


@app.post("/mood-decay")
def mood_decay(body: dict):
    user_id = body.get("user_id")
    if not user_id:
        return {"success": False, "error": "Missing user_id"}

    personality = personality_cache.get(user_id)
    if personality is None:
        personality = Personality(user_id)
        personality_cache.put(user_id, personality)

    personality.mood_decay()
    return {"success": True, "mood_vector": personality.mood_vector}

class GetPersonalityContextRequest(BaseModel):
    user_id: str

@app.post("/get-personality-context")
def get_personality_context(body: GetPersonalityContextRequest):
    user_id = body.user_id
    logging.info(f"[get_personality_context] user_id={user_id}")

    # Handle personality context directly
    personality = personality_cache.get(user_id)
    if personality is None:
        personality = Personality(user_id)
        personality_cache.put(user_id, personality)

    resp = {
        "archetype": personality.archetype,
        "traits": personality.traits,
        "mood_vector": personality.mood_vector
    }
    context = f"Archetype: {resp.get('archetype', '')}\n"
    context += "Traits: " + ", ".join(f"{k}: {v}" for k, v in resp.get("traits", {}).items()) + "\n"
    context += "Mood Vector: " + ", ".join(f"{k}: {v}" for k, v in resp.get("mood_vector", {}).items()) + "\n"
    logging.info(f"[get_personality_context] context={context}")
    return {"context": context}


class UpdateTraitRequest(BaseModel):
    user_id: str
    traits: dict

@app.post("/update-trait")
def update_trait(body: UpdateTraitRequest):
    user_id = body.user_id
    traits = body.traits
    logging.info(f"[update_trait] user_id={user_id}, traits={traits}")

    personality = personality_cache.get(user_id)
    if personality is None:
        personality = Personality(user_id)
        personality_cache.put(user_id, personality)

    # Update traits using the correct method name
    for trait, score in traits.items():
        personality.update_trait(trait, score)
    return {"success": True, "traits": personality.traits}


class GetTraitsRequest(BaseModel):
    user_id: str

@app.post("/get-traits")
def get_traits(body: GetTraitsRequest):
    user_id = body.user_id
    logging.info(f"[get_traits] user_id={user_id}")

    personality = personality_cache.get(user_id)
    if personality is None:
        personality = Personality(user_id)
        personality_cache.put(user_id, personality)

    return {"traits": personality.traits}


@app.post("/archetype-fusion")
def archetype_fusion(body: dict):
    user_id = body.get("user_id")
    archetypes = body.get("archetypes", [])
    if not user_id or not archetypes:
        return {"success": False, "error": "Missing user_id or archetypes"}

    personality = personality_cache.get(user_id)
    if personality is None:
        personality = Personality(user_id)
        personality_cache.put(user_id, personality)

    personality.fuse_archetypes(*archetypes)
    return {"success": True, "archetype": personality.archetype, "traits": personality.traits}

class SetPersonalityRequest(BaseModel):
    user_id: str
    traits: dict
    mood_vector: dict
    archetype: str

@app.post("/set-personality")
def set_personality(body: SetPersonalityRequest):
    user_id = body.user_id
    traits = body.traits
    mood_vector = body.mood_vector
    archetype = body.archetype
    logging.info(f"[set_personality] user_id={user_id}")

    personality = Personality(user_id)
    personality.traits = traits
    personality.mood_vector = mood_vector
    personality.archetype = archetype
    personality._save_data()

    personality_cache.put(user_id, personality)
    return {"success": True}


@app.post("/personalized-response")
def personalized_response(body: dict):
    user_id = body.get("user_id") if isinstance(body, dict) else getattr(body, "user_id", None)
    prompt = body.get("prompt") if isinstance(body, dict) else getattr(body, "prompt", None)
    logging.info(f"[personalized_response] user_id={user_id}, prompt={prompt}")
    if not user_id or not prompt:
        logging.error("Missing user_id or prompt in personalized_response")
        return {"success": False, "error": "Missing user_id or prompt"}

    personality = personality_cache.get(user_id)
    if personality is None:
        personality = Personality(user_id)
        personality_cache.put(user_id, personality)

    reply = personality.personalized_response(prompt)
    logging.info(f"[personalized_response] reply={reply}")
    return {"reply": reply}

if __name__ == "__main__":
    logging.info("[AGI Personality Server] Starting on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
