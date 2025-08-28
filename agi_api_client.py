"""
Centralized API Client for AGI Decentralized Stack
Handles all memory, personality, and global knowledge operations via REST endpoints.
"""
import httpx
import os
import hmac
import hashlib
import base64
import json

API_KEY = os.getenv("AGI_API_KEY", "testkey")
SHARED_SECRET = os.getenv("KNOWLEDGE_SYNC_SECRET", "supersecret")
HEADERS = {"X-API-Key": API_KEY}

# Utility for signing payloads
def sign_payload(payload: str, secret: str = SHARED_SECRET) -> str:
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(sig).decode()

class AGIAPIClient:
    def __init__(self, memory_url, personality_url, global_url):
        self.memory_url = memory_url
        self.personality_url = personality_url
        self.global_url = global_url

    def post(self, url, data):
        payload = json.dumps(data, sort_keys=True)
        signature = sign_payload(payload)
        headers = HEADERS.copy()
        headers["X-Signature"] = signature
        resp = httpx.post(url, json=data, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get(self, url, params=None):
        payload = json.dumps(params or {}, sort_keys=True)
        signature = sign_payload(payload)
        headers = HEADERS.copy()
        headers["X-Signature"] = signature
        resp = httpx.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # Memory operations
    def set_memory(self, user_id, key, value):
        return self.post(f"{self.memory_url}/set-memory", {"user_id": user_id, "key": key, "value": value})

    def get_memory(self, user_id, key):
        return self.post(f"{self.memory_url}/get-memory", {"user_id": user_id, "key": key})

    def get_user_context(self, user_id):
        return self.post(f"{self.memory_url}/get-user-context", {"user_id": user_id})

    # Personality operations
    def set_personality(self, user_id, traits, mood_vector, archetype):
        return self.post(f"{self.personality_url}/set-personality", {
            "user_id": user_id,
            "traits": traits,
            "mood_vector": mood_vector,
            "archetype": archetype
        })

    def get_personality(self, user_id):
        return self.post(f"{self.personality_url}/get-personality", {"user_id": user_id})

    def add_observation(self, user_id, observation):
        return self.post(f"{self.personality_url}/add-observation", {"user_id": user_id, "observation": observation})

    # Global knowledge operations
    def push_global_knowledge(self, topic, insights, emotional_tone, ts):
        return self.post(f"{self.global_url}/push_global_knowledge", {
            "topic": topic,
            "insights": insights,
            "emotional_tone": emotional_tone,
            "ts": ts
        })

    def get_global_knowledge(self):
        return self.get(f"{self.global_url}/get_global_knowledge")
