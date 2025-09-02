# =============================
# AGI Agent Server (Python, FastAPI)
# Overseer-compatible: Registered with overseer gateway (port 8010)
# All frontend requests are routed through overseer; backend-to-backend calls use direct service URLs
# CORS is restricted to overseer gateway for security
# =============================

import os
import sys
# Connection pooling for better performance
import httpx
from httpx import Timeout

# Create a client with connection pooling and reasonable timeouts
http_client = httpx.Client(
    timeout=Timeout(10.0, connect=5.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
)
import logging
import subprocess
import tempfile
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Environment configuration
AGENT_PORT = int(os.getenv('AGI_AGENT_PORT', 8000))
MEMORY_API_URL = os.getenv('MEMORY_API_URL', 'http://127.0.0.1:8001')
PERSONALITY_API_URL = os.getenv('PERSONALITY_API_URL', 'http://127.0.0.1:8002')

sys.path.append(".")

from memory.memory_api import MemoryAPIClient

# Initialize memory client
user_memory_client = MemoryAPIClient()

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

# Initialize FastAPI app and logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agi_agent_server")

@app.get("/health")
def health_check():
    """Health check endpoint for the agent server"""
    return {
        "status": "healthy",
        "service": "agent_server",
        "port": AGENT_PORT,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/import-facebook")
def import_facebook(file: UploadFile = File(...), user_id: str = Form("default")):
    """
    Accepts a Facebook JSON or ZIP file, parses posts, and stores them in user memory.
    """
    logger.info(f"[import_facebook] Received upload for user_id={user_id}, filename={getattr(file, 'filename', '')}")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file_bytes = file.file.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name
        logger.info(f"[import_facebook] File saved to temp: {tmp_path}, size={len(file_bytes)} bytes")

        posts = load_facebook_posts(tmp_path)
        logger.info(f"[import_facebook] Loaded {len(posts)} posts from file.")
        unique_posts = dedupe_posts(posts)
        logger.info(f"[import_facebook] Dedupe complete: {len(unique_posts)} unique posts.")
        private_candidates, public_candidates = privacy_analysis(unique_posts)
        logger.info(f"[import_facebook] Privacy analysis: {len(private_candidates)} private, {len(public_candidates)} public.")
        stored_count = store_private_posts(user_id, private_candidates)
        logger.info(f"[import_facebook] Stored {stored_count} private posts for user_id={user_id}.")
        
        # Return complete response with all fields expected by frontend
        return JSONResponse({
            "success": True,
            "posts_stored": stored_count,
            "private_count": len(private_candidates),
            "public_count": len(public_candidates),
            "public_candidates": public_candidates
        })
    except Exception as e:
        logger.error(f"[import_facebook] Exception: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
# Endpoint to accept scrubbed posts for public/global learning
@app.post("/import-facebook-public")
def import_facebook_public(payload: dict):
    """
    Accepts scrubbed posts for public/global learning (Postgres).
    """
    user_id = payload.get('user_id', 'default')
    posts = payload.get('posts', [])
    stored_count = 0
    for idx, post in enumerate(posts):
        content = post.get('content') if isinstance(post, dict) else None
        if content:
            key = f"fb_public_post_{idx}"
            value = content
            try:
                # Store in global knowledge (Postgres)
                # Use topic as key, insights as value
                user_memory_client.store_memory(user_id, key, value)  # Optionally, use a global API if available
                stored_count += 1
            except Exception as e:
                logging.error(f"[import_facebook_public] Error storing public post: {e}")
    logging.info(f"[import_facebook_public] Stored {stored_count} public posts for global learning")
    return {"success": True, "posts_stored": stored_count}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8010"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Legacy text generation - removed transformers dependency
# Now uses personality-driven response system instead
# generator = pipeline("text-generation", model="gpt2")  # DEPRECATED

MEMORY_API_URL = "http://127.0.0.1:8001"  # Direct internal call, not via overseer

# Use the MemoryAPIClient from memory_api.py
user_memory_client = MemoryAPIClient()

# Main endpoint for frontend

# Main endpoint for frontend
@app.options("/generate")
def options_generate():
    return JSONResponse(status_code=200, content={})


# Pydantic model for request body

class ObservationRequest(BaseModel):
    user_id: str = "default"

class GenerateRequest(BaseModel):
    prompt: str = ""
    user_id: str = "default"

@app.post("/get-observation")
def get_observation(body: ObservationRequest):
    user_id = body.user_id
    # For demonstration, return last 20 memory keys and values
    context = user_memory_client.get_user_context(user_id)
    if isinstance(context, dict):
        basic_context = context.get('basic_context', '')
        lines = basic_context.splitlines()
    else:
        lines = str(context).splitlines()
    history = []
    for line in lines:
        if ": " in line:
            k, v = line.split(": ", 1)
            history.append({"key": k, "value": v})
    return {"observation_history": history[-20:]}


def _safely_extract_dict(data: dict, key: str, default: Optional[dict] = None) -> dict:
    """Safely extract dictionary value with type checking."""
    if default is None:
        default = {}
    value = data.get(key, default)
    return value if isinstance(value, dict) else default

def _extract_personality_data(user_context: dict) -> tuple:
    """Extract and validate personality data from user context."""
    advanced_analysis = _safely_extract_dict(user_context, 'advanced_analysis')
    personality_scores = _safely_extract_dict(advanced_analysis, 'personality_scores')
    emotional_profile = _safely_extract_dict(advanced_analysis, 'emotional_profile')
    communication_style = _safely_extract_dict(advanced_analysis, 'communication_style')
    
    # Extract key components
    traits = personality_scores
    mood = emotional_profile.get('emotion_scores', {}) if emotional_profile else {}
    personality_data = _safely_extract_dict(user_context, 'personality_data')
    archetype = personality_data.get('archetype', 'balanced')
    
    return traits, mood, archetype, communication_style, emotional_profile

def get_user_personality_context(user_id: str) -> dict:
    """Get comprehensive personality context for a user."""
    try:
        user_context = user_memory_client.get_user_context(user_id)
        if not isinstance(user_context, dict):
            user_context = {'basic_context': str(user_context)}
        
        traits, mood, archetype, communication_style, emotional_profile = _extract_personality_data(user_context)
        
        return {
            'user_context': user_context,
            'traits': traits,
            'mood': mood,
            'archetype': archetype,
            'communication_style': communication_style,
            'emotional_profile': emotional_profile
        }
        
    except Exception as e:
        logging.error(f"Failed to get advanced user context: {e}")
        return get_fallback_personality_context(user_id)

def get_fallback_personality_context(user_id: str) -> dict:
    """Fallback personality context using direct API calls."""
    try:
        with httpx.Client(timeout=5.0) as client:
            personality_resp = client.post(f"{PERSONALITY_API_URL}/get-personality", json={"user_id": user_id})
            personality_resp.raise_for_status()
            personality_data = personality_resp.json()
            return {
                'user_context': {'basic_context': ''},
                'traits': personality_data.get("traits", {}),
                'mood': personality_data.get("mood_vector", {}),
                'archetype': personality_data.get("archetype", ""),
                'communication_style': {},
                'emotional_profile': {}
            }
    except Exception as fallback_e:
        logging.error(f"Fallback personality fetch failed: {fallback_e}")
        return {
            'user_context': {'basic_context': ''},
            'traits': {}, 'mood': {}, 'archetype': '',
            'communication_style': {}, 'emotional_profile': {}
        }

def fetch_recent_interactions_data(user_id: str) -> list:
    """Fetch recent user interactions data (renamed to avoid duplication)."""
    try:
        # Get recent interactions from memory directly
        recent_mem = user_memory_client.get_user_context(user_id)
        if isinstance(recent_mem, dict):
            basic_context = recent_mem.get('basic_context', '')
            lines = basic_context.splitlines()
        else:
            lines = str(recent_mem).splitlines()
        interactions = []
        for line in lines[-20:]:  # Last 20 memory entries
            if ": " in line:
                k, v = line.split(": ", 1)
                if "interaction" in k.lower() or "response" in k.lower():
                    interactions.append({"key": k, "value": v})
        return interactions[-10:]  # Return last 10
    except Exception as e:
        logging.error(f"Failed to get recent interactions: {e}")
        return []

def get_semantic_context(user_id: str, prompt: str, archetype: str) -> list:
    """Get semantically relevant moments."""
    try:
        with httpx.Client(timeout=5.0) as client:
            # Include personality traits in semantic search for better relevance
            enhanced_query = f"{prompt} personality:{archetype}"
            semantic_resp = client.post(f"{MEMORY_API_URL}/semantic_search",
                                      json={"user_id": user_id, "query": enhanced_query})
            semantic_resp.raise_for_status()
            semantic_data = semantic_resp.json()
            return semantic_data.get("results", [])
    except Exception as e:
        logging.error(f"Failed to get semantic search: {e}")
        return []

def get_personalized_response(user_id: str, prompt: str, context: dict) -> str:
    """Get AI-generated personalized response."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response_resp = client.post(f"{PERSONALITY_API_URL}/personalized-response",
                                      json={"user_id": user_id, "prompt": prompt, "context": context})
            response_resp.raise_for_status()
            response_data = response_resp.json()
            return response_data.get("reply", "I'm here for you. How can I help?")
    except Exception as e:
        logging.error(f"Failed to get personalized response: {e}")
        return "I'm here for you. How can I help?"

def update_conversation_memory(user_id: str, prompt: str, reply: str, context: dict):
    """Update conversation memory and personality analysis."""
    try:
        conversation_data = [{
            'user_message': prompt,
            'assistant_response': reply,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'context': context
        }]
        user_memory_client.update_personality_batch(user_id, conversation_data)
        user_memory_client.add_conversation_context(user_id, prompt, reply)
    except Exception as e:
        logging.error(f"Failed to update conversation memory: {e}")

@app.post("/generate")
def generate(body: GenerateRequest):
    """Main generation endpoint - now modular and maintainable."""
    logging.info(f"[generate] Received: {body}")
    prompt = body.prompt
    user_id = body.user_id

    # Get personality context
    personality_ctx = get_user_personality_context(user_id)
    
    # Get additional context
    recent_interactions = fetch_recent_interactions_data(user_id)
    relevant_moments = get_semantic_context(user_id, prompt, personality_ctx['archetype'])
    
    # Get memory context
    mem = personality_ctx['user_context'].get('basic_context', '')

    # Build enhanced context for response fusion
    context = {
        "traits": personality_ctx['traits'],
        "mood": personality_ctx['mood'],
        "archetype": personality_ctx['archetype'],
        "recent_interactions": recent_interactions,
        "memory": mem,
        "relevant_moments": relevant_moments,
        "communication_style": personality_ctx['communication_style'],
        "emotional_profile": personality_ctx['emotional_profile']
    }
    logging.info(f"[generate] Enhanced context built successfully")

    # Get personalized response
    reply = get_personalized_response(user_id, prompt, context)
    
    # Update memory and personality
    update_conversation_memory(user_id, prompt, reply, context)

    return {"response": reply}

@app.post("/get-recent-interactions")
def get_recent_interactions(body: ObservationRequest):
    user_id = body.user_id
    # Get recent interactions from memory
    try:
        # Get last 10 interactions from memory
        recent_mem = user_memory_client.get_user_context(user_id)
        if isinstance(recent_mem, dict):
            basic_context = recent_mem.get('basic_context', '')
            lines = basic_context.splitlines()
        else:
            lines = str(recent_mem).splitlines()
        interactions = []
        for line in lines[-20:]:  # Last 20 memory entries
            if ": " in line:
                k, v = line.split(": ", 1)
                if "interaction" in k.lower() or "response" in k.lower():
                    interactions.append({"key": k, "value": v})
        return {"recent_interactions": interactions[-10:]}  # Return last 10
    except Exception as e:
        logging.error(f"[get_recent_interactions] Failed: {e}")
        return {"recent_interactions": []}

# Legacy endpoint (optional)

class TextGenerationRequest(BaseModel):
    prompt: str = ""

@app.post("/text-generation")
def text_generation(body: TextGenerationRequest):
    """Legacy endpoint - now redirects to personality-driven generation."""
    logging.info(f"[text_generation] Received: {body} - redirecting to personality system")
    prompt = body.prompt
    
    # Use the personality-driven system instead of transformers
    generate_request = GenerateRequest(prompt=prompt, user_id="default")
    result = generate(generate_request)
    
    # Format for legacy compatibility
    generated_text = result.get("response", "")
    
    logging.info(f"[text_generation] Generated via personality system: {generated_text}")
    return {"generated_text": generated_text}

def load_facebook_posts(tmp_path):
    import json
    with open(tmp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('posts', [])
        return []

def dedupe_posts(posts):
    seen = set()
    unique_posts = []
    for post in posts:
        content = post.get('content') if isinstance(post, dict) else None
        if content and content not in seen:
            seen.add(content)
            unique_posts.append(post)
    return unique_posts

def privacy_analysis(posts):
    import re
    private_candidates = []
    public_candidates = []
    for post in posts:
        raw_content = post.get('content') if isinstance(post, dict) else ''
        content = raw_content if isinstance(raw_content, str) else ''
        if re.search(r'\b(email|@|gmail|yahoo|hotmail|phone|address|\d{3}-\d{3}-\d{4}|\d{10}|\d{5}|street|avenue|road|drive|lane|blvd|Dr\.|Mr\.|Ms\.|Mrs\.|\b[A-Z][a-z]+\b)', content):
            private_candidates.append(post)
        else:
            scrubbed = re.sub(r'([A-Z][a-z]+|@\S+|\d{3}-\d{3}-\d{4}|\d{10}|\d{5}|street|avenue|road|drive|lane|blvd|Dr\.|Mr\.|Ms\.|Mrs\.)', '[REDACTED]', content)
            public_candidates.append({'content': scrubbed})
    return private_candidates, public_candidates

def store_private_posts(user_id, private_candidates):
    stored_count = 0
    import time
    current_timestamp = int(time.time())
    
    for idx, post in enumerate(private_candidates):
        try:
            content = post.get('content') if isinstance(post, dict) else None
            if content:
                post_id = f"fb_post_{user_id}_{idx}_{current_timestamp}"
                tags = post.get('tags', [])
                # Use current timestamp if post doesn't have one (Facebook posts often don't have timestamps in export)
                timestamp = post.get('timestamp', current_timestamp - (idx * 60))  # Stagger by minutes
                user_memory_client.store_post(user_id, post_id, content, timestamp, tags)
                stored_count += 1
        except Exception as e:
            logging.error(f"[import_facebook] Error storing post: {e}")
    logging.info(f"[import_facebook] Stored {stored_count} private posts for user_id={user_id}")
    return stored_count


# Facebook summary endpoint
from fastapi import Body
@app.post("/facebook-summary")
def facebook_summary(user_id: str = Body("default")):
    posts = []
    # Try to fetch last 20 posts
    posts = user_memory_client.get_posts(user_id)
    posts = [p["content"] for p in posts if "content" in p]
    if not posts:
        return {"summary": "I don't have any Facebook memories for you yet."}
# =============================
# Journal Helper Functions
# =============================

async def get_personality_data(user_id: str) -> dict:
    """Get personality data for a user."""
    try:
        # This would typically fetch from the personality server
        # For now, return default personality data
        return {
            "traits": {
                "openness": 0.7,
                "conscientiousness": 0.8,
                "extraversion": 0.6,
                "agreeableness": 0.9,
                "neuroticism": 0.3
            },
            "values": ["reflection", "growth", "understanding"],
            "goals": ["personal development", "meaningful connections"]
        }
    except Exception as e:
        logger.error(f"Error getting personality data for user {user_id}: {e}")
        return {}

async def generate_journal_analysis(content: str, personality_data: dict) -> str:
    """Generate AI analysis of a journal entry."""
    try:
        # This would use the personality system to analyze the entry
        # For now, provide a basic analysis
        analysis = f"Based on your personality profile, I notice you're reflecting on {content[:50]}..."
        
        # Add personality insights
        if personality_data.get("traits", {}).get("openness", 0) > 0.7:
            analysis += " Your openness to experience suggests you're exploring new perspectives."
        if personality_data.get("traits", {}).get("conscientiousness", 0) > 0.7:
            analysis += " Your conscientious nature shows in your thoughtful approach."
            
        return analysis
        
    except Exception as e:
        logger.error(f"Error generating journal analysis: {e}")
        return "Your journal entry has been processed. I can see you're reflecting deeply on your experiences."

# =============================
# Journal Endpoints
# =============================

class JournalEntryRequest(BaseModel):
    id: str
    content: str
    timestamp: str
    date: str
    time: str
    user_id: str

class JournalAnalysisRequest(BaseModel):
    current_entry: str
    previous_entries: List[dict]
    user_id: str

class JournalEntry(BaseModel):
    id: str
    content: str
    timestamp: str
    date: str
    time: str
    user_id: str

class JournalRequest(BaseModel):
    content: str
    user_id: str

class GetJournalEntriesRequest(BaseModel):
    user_id: str
    limit: int = 50

class MemoryRequest(BaseModel):
    user_id: str = "default"

@app.post("/add-journal-entry")
def add_journal_entry(entry: JournalEntryRequest):
    """Add a new journal entry to the user's memory"""
    try:
        logger.info(f"[add_journal_entry] Adding journal entry for user_id={entry.user_id}")
        
        # Store the journal entry in memory
        journal_data = {
            "id": entry.id,
            "content": entry.content,
            "timestamp": entry.timestamp,
            "date": entry.date,
            "time": entry.time,
            "type": "journal_entry"
        }
        
        # Use the memory client to store the journal entry
        success = user_memory_client.store_journal_entry(entry.user_id, journal_data)
        
        if success:
            logger.info(f"[add_journal_entry] Successfully stored journal entry for user_id={entry.user_id}")
            return {"success": True, "message": "Journal entry saved successfully"}
        else:
            logger.error(f"[add_journal_entry] Failed to store journal entry for user_id={entry.user_id}")
            return {"success": False, "error": "Failed to save journal entry"}
            
    except Exception as e:
        logger.error(f"[add_journal_entry] Error saving journal entry: {e}")
        return {"success": False, "error": str(e)}

@app.post("/get-journal-entries")
def get_journal_entries(request: GetJournalEntriesRequest):
    """Retrieve journal entries for a user"""
    try:
        logger.info(f"[get_journal_entries] Retrieving entries for user_id={request.user_id}, limit={request.limit}")
        
        # Get journal entries from memory
        entries = user_memory_client.get_journal_entries(request.user_id, request.limit)
        
        logger.info(f"[get_journal_entries] Retrieved {len(entries)} entries for user_id={request.user_id}")
        return {"entries": entries}
        
    except Exception as e:
        logger.error(f"[get_journal_entries] Error retrieving journal entries: {e}")
        return {"entries": [], "error": str(e)}

def _get_personality_context(user_id: str) -> str:
    """Get personality context for journal analysis."""
    try:
        personality_resp = http_client.post(
            f"{PERSONALITY_API_URL}/get-personality-context",
            json={"user_id": user_id}
        )
        if personality_resp.status_code == 200:
            personality_data = personality_resp.json()
            return personality_data.get("context", "")
    except Exception as e:
        logger.warning(f"Could not fetch personality context: {e}")
    return ""

def _build_analysis_prompt(current_entry: str, previous_entries: List[dict], personality_context: str) -> str:
    """Build comprehensive analysis prompt for journal entry."""
    analysis_prompt = f"""
    Analyze this journal entry and provide insights. Consider the user's personality context and compare with previous entries if available.

    CURRENT JOURNAL ENTRY:
    {current_entry}

    USER PERSONALITY CONTEXT:
    {personality_context}

    PREVIOUS ENTRIES (last {len(previous_entries)}):
    """
    
    for i, entry in enumerate(previous_entries[:3]):  # Limit to last 3 for context
        analysis_prompt += f"\n{i+1}. {entry.get('date', 'Unknown date')}: {entry.get('content', '')[:200]}..."
    
    analysis_prompt += """

    Please provide:
    1. Key themes and emotions expressed
    2. Patterns or insights about the user's thoughts
    3. Any connections to previous entries
    4. Gentle, supportive observations
    5. Questions that might help deeper reflection

    Keep your response empathetic, insightful, and encouraging.
    """
    return analysis_prompt

def _generate_analysis(prompt: str, user_id: str) -> str:
    """Generate analysis using the text generation endpoint."""
    try:
        gen_resp = http_client.post(
            f"http://localhost:{AGENT_PORT}/generate",
            json={"prompt": prompt, "user_id": user_id}
        )
        
        if gen_resp.status_code == 200:
            gen_data = gen_resp.json()
            return gen_data.get("response", "Analysis generated successfully.")
        else:
            return "I analyzed your journal entry and found it very insightful. Your thoughts show depth and self-awareness."
            
    except Exception as e:
        logger.warning(f"Text generation failed: {e}")
        return "Your journal entry has been processed. I can see you're reflecting deeply on your experiences."

@app.post("/analyze-journal-entry")
def analyze_journal_entry(request: JournalAnalysisRequest):
    """Analyze a journal entry and provide insights"""
    try:
        logger.info(f"[analyze_journal_entry] Analyzing entry for user_id={request.user_id}")
        
        # Get personality context for the user
        personality_context = _get_personality_context(request.user_id)
        
        # Prepare analysis prompt
        analysis_prompt = _build_analysis_prompt(request.current_entry, request.previous_entries, personality_context)
        
        # Generate analysis
        analysis = _generate_analysis(analysis_prompt, request.user_id)
        
        logger.info(f"[analyze_journal_entry] Analysis completed for user_id={request.user_id}")
        return {"analysis": analysis}
        
    except Exception as e:
        logger.error(f"[analyze_journal_entry] Error analyzing journal entry: {e}")
        return {"analysis": "I apologize, but I couldn't analyze your journal entry right now. Please try again later."}

def _fetch_personality_context(user_id: str) -> Optional[str]:
    """Fetch personality context for memory compilation."""
    try:
        personality_resp = http_client.post(
            f"{PERSONALITY_API_URL}/get-personality-context",
            json={"user_id": user_id}
        )
        if personality_resp.status_code == 200:
            personality_data = personality_resp.json()
            return personality_data.get("context", "")
    except Exception as e:
        logger.warning(f"Could not fetch personality context: {e}")
    return None

def _fetch_journal_summary(user_id: str) -> Optional[str]:
    """Fetch recent journal entries summary."""
    try:
        journal_entries = user_memory_client.get_journal_entries(user_id, limit=5)
        if journal_entries and isinstance(journal_entries, list) and journal_entries:
            journal_summary = "\n".join([f"• {entry.get('date', 'Unknown date')}: {entry.get('content', '')[:100]}..." for entry in journal_entries[-3:]])
            return journal_summary
    except Exception as e:
        logger.warning(f"Could not fetch journal entries: {e}")
    return None

def _fetch_conversation_patterns(user_id: str) -> Optional[str]:
    """Fetch conversation patterns from memory API."""
    try:
        user_context_resp = user_memory_client.get_user_context(user_id)
        if user_context_resp and isinstance(user_context_resp, dict):
            return user_context_resp.get("recent_conversations", "")
    except Exception as e:
        logger.warning(f"Could not fetch user context: {e}")
    return None

def _compile_memory_context(personality_context: str, journal_summary: str, conversation_patterns: str) -> str:
    """Compile all memory components into unified context."""
    memory_context = []
    
    if personality_context:
        memory_context.append(f"PERSONALITY PROFILE:\n{personality_context}")
    if journal_summary:
        memory_context.append(f"RECENT JOURNAL REFLECTIONS:\n{journal_summary}")
    if conversation_patterns:
        memory_context.append(f"CONVERSATION PATTERNS:\n{conversation_patterns}")
    
    if memory_context:
        return "\n\n".join(memory_context)
    else:
        return "Hello! I'm getting to know you. We can start building your memory through conversations, journal entries, and shared experiences."

@app.post("/abigail/memory")
def get_user_memory(request: MemoryRequest):
    """Get user's memory context - aggregated view of personality, interactions, and journal entries"""
    try:
        logger.info(f"[abigail/memory] Fetching memory context for user_id={request.user_id}")
        
        # Fetch all memory components
        personality_context = _fetch_personality_context(request.user_id) or ""
        journal_summary = _fetch_journal_summary(request.user_id) or ""
        conversation_patterns = _fetch_conversation_patterns(request.user_id) or ""
        
        # Compile unified context
        user_context = _compile_memory_context(personality_context, journal_summary, conversation_patterns)
        
        logger.info(f"[abigail/memory] Successfully compiled memory context for user_id={request.user_id}")
        return {"user_context": user_context}
        
    except Exception as e:
        logger.error(f"[abigail/memory] Error fetching user memory: {e}")
        return {"user_context": "I'm still learning about you. Let's continue our conversation to build your personal memory."}

if __name__ == "__main__":
    import atexit

    def cleanup():
        """Clean up resources on shutdown"""
        try:
            http_client.close()
            print("[AGI Agent Server] HTTP client closed")
        except Exception as e:
            print(f"[AGI Agent Server] Error closing HTTP client: {e}")

    atexit.register(cleanup)

    print("[AGI Agent Server] Starting on port 8000...")
    clear_port(8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)