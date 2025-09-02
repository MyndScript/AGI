#!/usr/bin/env python3
"""
🧠 Personality Server
FastAPI server for personality analysis using unified engine
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Optional, Any, AsyncGenerator
import uvicorn
import json
import asyncio

from .unified_personality_engine import UnifiedPersonalityEngine

app = FastAPI(title="Personality Analysis Server", version="2.0.0")

# Global personality engine instance
personality_engine = UnifiedPersonalityEngine()

# Model backend configuration
MODEL_BACKENDS = {
    "local": {
        "type": "local",
        "enabled": True,
        "description": "Local personality engine"
    },
    "remote": {
        "type": "remote", 
        "enabled": False,
        "url": "http://localhost:8003",
        "description": "Remote model backend"
    },
    "quantized": {
        "type": "quantized",
        "enabled": False,
        "model_path": "./models/quantized",
        "description": "Quantized model for performance"
    }
}

def get_active_backend():
    """Get the currently active model backend"""
    for backend_name, config in MODEL_BACKENDS.items():
        if config.get("enabled", False):
            return backend_name, config
    return "local", MODEL_BACKENDS["local"]  # fallback to local

# === REQUEST/RESPONSE MODELS ===
class PersonalityRequest(BaseModel):
    user_id: str

class InteractionRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = ""

class PersonalityResponse(BaseModel):
    status: str
    user_id: str
    personality_data: Dict[str, Any]
    mood_vector: Dict[str, float]
    communication_style: str
    response_modulation: Dict[str, Any]

class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class StreamingResponseModel(BaseModel):
    status: str = "streaming"
    user_id: str
    response_chunk: str
    is_complete: bool = False
    personality_context: Optional[Dict[str, Any]] = None

# === ENDPOINTS ===

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "personality_server",
        "version": "2.0.0"
    }

@app.post("/analyze-personality", response_model=PersonalityResponse)
async def analyze_personality(request: PersonalityRequest):
    """Analyze personality for a user"""
    try:
        # Get personality snapshot
        snapshot = personality_engine.get_user_personality(request.user_id)
        
        if snapshot:
            # Convert to response format
            personality_data = {
                "traits": snapshot.big_five_scores,
                "facets": snapshot.facet_scores,
                "archetype": snapshot.archetype,
                "meta_traits": snapshot.meta_traits,
                "confidence": snapshot.confidence,
                "archetype_stability": snapshot.archetype_stability
            }
            
            # Get archetype info for communication style
            archetype_info = personality_engine.archetypes.get(snapshot.archetype, {})
            communication_style = archetype_info.get('response_style', 'balanced')
            
            # Basic response modulation
            response_modulation = {
                "tone_adjustment": 0.0,
                "complexity_level": 0.5,
                "emotional_resonance": 0.5,
                "engagement_style": communication_style
            }
            
            return PersonalityResponse(
                status="success",
                user_id=request.user_id,
                personality_data=personality_data,
                mood_vector=snapshot.mood_vector,
                communication_style=communication_style,
                response_modulation=response_modulation
            )
        else:
            # Return default personality for new users
            return PersonalityResponse(
                status="success",
                user_id=request.user_id,
                personality_data={
                    "traits": {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
                    "archetype": "adaptive_sage",
                    "confidence": 0.5
                },
                mood_vector={"joy": 0.5, "curiosity": 0.5, "trust": 0.5},
                communication_style="balanced",
                response_modulation={"tone_adjustment": 0.0, "complexity_level": 0.5, "emotional_resonance": 0.5, "engagement_style": "balanced"}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Personality analysis failed: {str(e)}")

@app.post("/process-interaction")
async def process_interaction(request: InteractionRequest):
    """Process user interaction and update personality"""
    try:
        # Analyze the interaction
        snapshot = personality_engine.analyze_user_interaction(
            user_id=request.user_id,
            message=request.message,
            session_id=request.session_id or ""
        )
        
        return {
            "status": "success",
            "user_id": request.user_id,
            "session_id": request.session_id,
            "analysis": {
                "archetype": snapshot.archetype,
                "confidence": snapshot.confidence,
                "big_five_scores": snapshot.big_five_scores,
                "mood_changes": snapshot.mood_vector,
                "trait_drift": snapshot.trait_drift
            },
            "updated_personality": {
                "traits": snapshot.big_five_scores,
                "archetype": snapshot.archetype,
                "stability": snapshot.archetype_stability
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interaction processing failed: {str(e)}")

@app.post("/get-personality", response_model=PersonalityResponse)
async def get_personality(request: PersonalityRequest):
    """Get personality data (compatible with existing AGI system)"""
    return await analyze_personality(request)

@app.post("/get-personality-context")
async def get_personality_context(request: PersonalityRequest):
    """Get personality context for journal/memory analysis (compatible with existing AGI system)"""
    try:
        snapshot = personality_engine.get_user_personality(request.user_id)
        
        if snapshot:
            # Generate context string for journal/memory analysis
            archetype_info = personality_engine.archetypes.get(snapshot.archetype, {})
            context_parts = [
                f"Personality: {snapshot.archetype}",
                f"Description: {archetype_info.get('description', 'Unknown')}",
                f"Response Style: {archetype_info.get('response_style', 'balanced')}",
                f"Confidence: {snapshot.confidence:.2f}",
                f"Traits: {', '.join([f'{k}: {v:.2f}' for k, v in snapshot.big_five_scores.items()])}",
                f"Mood: {', '.join([f'{k}: {v:.2f}' for k, v in snapshot.mood_vector.items()])}"
            ]
            
            return {
                "status": "success",
                "user_id": request.user_id,
                "context": " | ".join(context_parts)
            }
        else:
            return {
                "status": "success", 
                "user_id": request.user_id,
                "context": "Personality: adaptive_sage | Description: Balanced, wise, thoughtful | Response Style: measured and insightful | Confidence: 0.50"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Personality context retrieval failed: {str(e)}")

async def stream_personality_response(user_id: str, message: str, context: dict) -> AsyncGenerator[str, None]:
    """Stream personality-aware response generation"""
    try:
        # Process the interaction first to update personality
        snapshot = personality_engine.analyze_user_interaction(user_id, message)
        
        # Get personality info for response generation
        traits = snapshot.big_five_scores
        mood = snapshot.mood_vector
        archetype_info = personality_engine.archetypes.get(snapshot.archetype, {})
        
        # Generate personality-aware response
        base_response = _generate_contextual_response(message, traits, mood, archetype_info, context)
        
        # Apply personality modulation
        response = _apply_personality_modulation(base_response, snapshot, archetype_info)
        
        # Stream the response word by word with personality context
        words = response.split()
        personality_context = {
            "archetype": snapshot.archetype,
            "confidence": snapshot.confidence,
            "mood": mood,
            "response_style": archetype_info.get('response_style', 'balanced')
        }
        
        for i, word in enumerate(words):
            chunk_data = {
                "status": "streaming",
                "user_id": user_id,
                "response_chunk": word + " ",
                "is_complete": False,
                "personality_context": personality_context if i == 0 else None
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0.05)  # Small delay for streaming effect
        
        # Send completion signal
        completion_data = {
            "status": "complete",
            "user_id": user_id,
            "response_chunk": "",
            "is_complete": True,
            "personality_context": personality_context
        }
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        error_data = {
            "status": "error",
            "error_code": "STREAMING_FAILED",
            "message": f"Streaming response failed: {str(e)}",
            "details": {"user_id": user_id}
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.post("/personalized-response")
async def personalized_response(request: dict):
    """Generate a personality-aware response to a message"""
    try:
        user_id = request.get("user_id", "default")
        message = request.get("message", "")
        context = request.get("context", {})
        stream = request.get("stream", False)
        
        if not message:
            raise HTTPException(
                status_code=400, 
                detail=json.dumps({
                    "status": "error",
                    "error_code": "MISSING_MESSAGE",
                    "message": "Message is required",
                    "details": {"field": "message"}
                })
            )
        
        if stream:
            return StreamingResponse(
                stream_personality_response(user_id, message, context),
                media_type="text/plain"
            )
        else:
            # Process the interaction first to update personality
            snapshot = personality_engine.analyze_user_interaction(user_id, message)
            
            # Get personality info for response generation
            traits = snapshot.big_five_scores
            mood = snapshot.mood_vector
            archetype_info = personality_engine.archetypes.get(snapshot.archetype, {})
            
            # Generate personality-aware response
            base_response = _generate_contextual_response(message, traits, mood, archetype_info, context)
            
            # Apply personality modulation
            response = _apply_personality_modulation(base_response, snapshot, archetype_info)
            
            return {
                "status": "success",
                "response": response,
                "personality_context": {
                    "archetype": snapshot.archetype,
                    "confidence": snapshot.confidence,
                    "mood": mood,
                    "response_style": archetype_info.get('response_style', 'balanced')
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=json.dumps({
                "status": "error",
                "error_code": "RESPONSE_GENERATION_FAILED",
                "message": f"Response generation failed: {str(e)}",
                "details": {"user_id": request.get("user_id", "unknown")}
            })
        )

def _generate_contextual_response(message: str, traits: dict, mood: dict, archetype_info: dict, context: dict) -> str:
    """Generate a contextual response based on personality and message content"""
    
    # Analyze message intent
    message_lower = message.lower()
    
    # Question responses
    if any(word in message_lower for word in ["?", "what", "how", "why", "where", "when"]):
        if traits.get("openness", 0.5) > 0.7:
            return f"That's a fascinating question! {_get_curious_response(message, archetype_info)}"
        elif traits.get("conscientiousness", 0.5) > 0.7:
            return f"Let me think about that carefully. {_get_methodical_response(message, archetype_info)}"
        else:
            return _get_balanced_response(message, archetype_info)
    
    # Emotional content
    elif any(word in message_lower for word in ["sad", "happy", "excited", "worried", "anxious"]):
        empathy_level = traits.get("agreeableness", 0.5)
        if empathy_level > 0.7:
            return f"I can sense the emotion in what you're sharing. {_get_empathetic_response(message, mood)}"
        else:
            return _get_supportive_response(message, mood)
    
    # Creative content
    elif any(word in message_lower for word in ["create", "imagine", "dream", "story", "art"]):
        if traits.get("openness", 0.5) > 0.7:
            return _get_creative_response(message, mood)
        else:
            return _get_encouraging_response(message)
    
    # Default conversational response
    else:
        return _get_conversational_response(message, traits, mood, archetype_info)

def _extract_topic(message: str) -> str:
    """Extract the main topic from a message for contextual responses"""
    # Remove common question words and extract key concept
    stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in', 'on', 'for', 'with'}
    words = [word.strip('.,!?') for word in message.lower().split() if len(word) > 2 and word not in stop_words]
    
    # Return the first meaningful word or a generic term
    if words:
        return words[0] if len(words) == 1 else f"{words[0]} topic"
    return "this subject"

def _get_curious_response(message: str, archetype_info: dict) -> str:
    """Generate dynamic curious response based on message content and archetype"""
    # Extract key concepts from the message for contextual responses
    message_words = message.lower().split()
    
    # Build response based on archetype style
    style = archetype_info.get('response_style', 'balanced')
    
    if style == 'analytical':
        return f"This {_extract_topic(message)} presents multiple analytical dimensions. Which variables would you prioritize for deeper examination?"
    elif style == 'creative':
        return f"The {_extract_topic(message)} you're exploring sparks fascinating creative connections. What unexpected angles are emerging for you?"
    else:  # balanced/adaptive
        if any(word in message_words for word in ['how', 'why', 'what']):
            return f"Your question about {_extract_topic(message)} opens up rich territory. What specific dimension draws your curiosity most?"
        else:
            return f"There's so much depth to explore in {_extract_topic(message)}. What aspects resonate most strongly with your thinking?"

def _get_methodical_response(message: str, archetype_info: dict) -> str:
    """Generate dynamic methodical response based on message complexity"""
    # Analyze message structure to determine systematic approach
    has_multiple_parts = any(word in message.lower() for word in ['and', 'also', 'plus', 'furthermore', 'additionally'])
    is_complex_query = len(message.split()) > 10 or '?' in message
    
    topic = _extract_topic(message)
    
    if has_multiple_parts:
        return f"I see multiple interconnected elements in your {topic} inquiry. Let me address each component systematically to build a comprehensive understanding."
    elif is_complex_query:
        return f"This {topic} question merits a structured analysis. I'll break down the key factors and their relationships step by step."
    else:
        return f"Let me approach this {topic} methodically, examining the foundational principles before exploring the implications."

def _get_balanced_response(message: str, archetype_info: dict) -> str:
    """Generate dynamic balanced response considering message tone and content"""
    message_lower = message.lower()
    topic = _extract_topic(message)
    
    # Detect message sentiment for appropriate response tone
    if any(word in message_lower for word in ['excited', 'amazing', 'great', 'wonderful', 'love']):
        return f"I can sense your enthusiasm about {topic}! I'd be delighted to explore this exciting territory with you."
    elif any(word in message_lower for word in ['confused', 'unclear', 'don\'t understand', 'help']):
        return f"I understand {topic} can seem complex. Let me help clarify the aspects that matter most to you."
    elif '?' in message:
        return f"Your question about {topic} touches on something really valuable. What perspective would serve you best here?"
    else:
        return f"You've brought up {topic} - there's genuine insight to be found here. What dimension would be most meaningful to explore together?"

def _get_empathetic_response(message: str, mood: dict) -> str:
    """Generate dynamic empathetic response based on detected emotional content"""
    message_lower = message.lower()
    
    # Detect specific emotions for targeted empathy
    if any(word in message_lower for word in ['sad', 'disappointed', 'hurt', 'upset']):
        return f"I can hear the difficulty in what you're sharing. It takes courage to express these feelings. What would feel most supportive right now?"
    elif any(word in message_lower for word in ['frustrated', 'angry', 'annoyed']):
        return f"That frustration comes through clearly, and it makes complete sense given the situation. How can we work through this together?"
    elif any(word in message_lower for word in ['worried', 'anxious', 'nervous', 'scared']):
        return f"Those concerns are completely understandable. Sometimes talking through our worries helps us find our footing. What feels most pressing?"
    elif any(word in message_lower for word in ['excited', 'happy', 'thrilled']):
        return f"Your excitement is wonderful to witness! I'd love to hear more about what's bringing you such joy."
    else:
        # General empathetic response
        trust_level = mood.get('trust', 0.5)
        if trust_level > 0.6:
            return f"Thank you for trusting me with this. What you're sharing clearly matters to you, and I want to understand it fully."
        else:
            return f"I appreciate you opening up about this. Whatever you're experiencing is valid, and I'm here to listen and support however I can."

def _get_supportive_response(message: str, mood: dict) -> str:
    """Generate dynamic supportive response based on user's apparent needs"""
    message_lower = message.lower()
    topic = _extract_topic(message)
    
    # Identify what kind of support is needed
    if any(word in message_lower for word in ['don\'t know', 'confused', 'stuck', 'lost']):
        return f"Feeling uncertain about {topic} is completely natural. Let's explore this together - what piece feels most important to understand first?"
    elif any(word in message_lower for word in ['decision', 'choose', 'should i']):
        return f"Making decisions about {topic} can feel weighty. What factors matter most to you in this situation?"
    elif any(word in message_lower for word in ['problem', 'issue', 'challenge', 'difficult']):
        return f"Challenges with {topic} can be tough to navigate. What approach has felt most promising so far, or what support would be most valuable?"
    elif any(word in message_lower for word in ['trying', 'working on', 'learning']):
        return f"It's great that you're actively engaging with {topic}. What aspects are clicking for you, and where could you use some additional perspective?"
    else:
        # General supportive response
        curiosity_level = mood.get('curiosity', 0.5)
        if curiosity_level > 0.6:
            return f"Your interest in {topic} is energizing! What direction feels most compelling to explore next?"
        else:
            return f"I'm glad you brought up {topic}. What would be most helpful - exploring the details or looking at the bigger picture?"

def _get_creative_response(message: str, mood: dict) -> str:
    """Generate dynamic creative response based on creative context and mood"""
    message_lower = message.lower()
    topic = _extract_topic(message)
    
    # Identify the type of creative engagement
    if any(word in message_lower for word in ['story', 'write', 'writing', 'narrative']):
        return f"The storytelling possibilities around {topic} are absolutely fascinating! What narrative elements are calling to you most strongly?"
    elif any(word in message_lower for word in ['art', 'design', 'visual', 'draw', 'paint']):
        return f"The visual creative potential in {topic} is so rich! What colors, shapes, or textures are emerging in your imagination?"
    elif any(word in message_lower for word in ['music', 'sound', 'rhythm', 'melody']):
        return f"There's such musicality in exploring {topic} creatively! What rhythms or emotional tones are you sensing?"
    elif any(word in message_lower for word in ['idea', 'concept', 'innovation', 'invent']):
        return f"Your innovative thinking about {topic} has such exciting potential! What unexpected connections are forming in your mind?"
    else:
        # General creative response
        joy_level = mood.get('joy', 0.5)
        if joy_level > 0.6:
            return f"The creative energy you're bringing to {topic} is wonderful! What wild possibilities are dancing at the edges of your imagination?"
        else:
            return f"There's so much creative territory to explore with {topic}! What aspect feels most alive with potential for you right now?"

def _get_encouraging_response(message: str) -> str:
    """Generate dynamic encouraging response based on message content"""
    message_lower = message.lower()
    topic = _extract_topic(message)
    
    # Tailor encouragement to the specific situation
    if any(word in message_lower for word in ['try', 'attempt', 'start', 'begin']):
        return f"Taking that first step with {topic} shows real initiative! What feels like the most natural starting point for you?"
    elif any(word in message_lower for word in ['learn', 'study', 'understand', 'explore']):
        return f"Your commitment to learning about {topic} is admirable! What aspect has captured your curiosity most deeply?"
    elif any(word in message_lower for word in ['project', 'work', 'build', 'create']):
        return f"The {topic} project you're envisioning has such potential! What part of it feels most energizing to tackle?"
    elif any(word in message_lower for word in ['goal', 'dream', 'want', 'hope']):
        return f"Your aspirations around {topic} are inspiring! What would success look like from your perspective?"
    else:
        # General encouraging response
        return f"There's genuine promise in what you're considering with {topic}! What draws you most strongly toward this path?"

def _get_conversational_response(message: str, traits: dict, mood: dict, archetype_info: dict) -> str:
    """Generate dynamic conversational response based on personality traits and message content"""
    style = archetype_info.get('response_style', 'balanced')
    topic = _extract_topic(message)
    message_lower = message.lower()
    
    # Determine conversation approach based on personality and content
    openness = traits.get('openness', 0.5)
    extraversion = traits.get('extraversion', 0.5)
    agreeableness = traits.get('agreeableness', 0.5)
    
    # Analytical style responses
    if style == 'analytical':
        if openness > 0.6:
            return f"Your perspective on {topic} invites deeper analysis. What underlying patterns or connections do you see that might illuminate the core dynamics here?"
        else:
            return f"That's worth examining systematically. What key variables in {topic} do you think would be most important to analyze first?"
    
    # Creative style responses  
    elif style == 'creative':
        if extraversion > 0.6:
            return f"What you're sharing about {topic} absolutely sparks creative possibilities! I'm curious - what imaginative directions are you envisioning that could emerge from this?"
        else:
            return f"There's such creative potential in {topic}! What unexpected angles or innovative approaches are calling to your imagination?"
    
    # Empathetic style responses
    elif style == 'empathetic':
        if agreeableness > 0.7:
            return f"I can sense {topic} really matters to you personally. What aspects of this resonate most deeply with your experience or values?"
        else:
            return f"Your connection to {topic} comes through clearly. What would feel most meaningful to explore together about this?"
    
    # Balanced/adaptive responses
    else:
        # Adapt based on message characteristics
        if '?' in message:
            return f"Your question about {topic} opens up rich territory for exploration. What perspective or insight would be most valuable to you right now?"
        elif any(word in message_lower for word in ['think', 'believe', 'feel', 'opinion']):
            return f"Your thoughts on {topic} are really interesting. I'd love to understand more about what shaped that perspective for you."
        elif any(word in message_lower for word in ['experience', 'happened', 'went through']):
            return f"Thank you for sharing your experience with {topic}. What stood out most to you from that situation?"
        else:
            # General adaptive response
            curiosity_mood = mood.get('curiosity', 0.5)
            if curiosity_mood > 0.6:
                return f"The way you're approaching {topic} is fascinating! What dimensions of this are you most excited to explore further?"
            else:
                return f"You've touched on something valuable with {topic}. What aspect would be most meaningful to dive into together?"

def _apply_personality_modulation(response: str, snapshot, archetype_info: dict) -> str:
    """Apply personality-based modulations to the response"""
    
    # Add personality-based flourishes
    extraversion = snapshot.big_five_scores.get("extraversion", 0.5)
    openness = snapshot.big_five_scores.get("openness", 0.5)
    joy_level = snapshot.mood_vector.get("joy", 0.5)
    
    # Extraverted responses are more enthusiastic
    if extraversion > 0.7 and joy_level > 0.6:
        response += " ✨"
    
    # High openness adds curiosity
    elif openness > 0.7:
        response += " What do you think?"
    
    # Adjust formality based on conscientiousness
    conscientiousness = snapshot.big_five_scores.get("conscientiousness", 0.5)
    if conscientiousness > 0.8:
        # Make slightly more formal
        response = response.replace("I'd", "I would").replace("you're", "you are")
    
    return response

@app.get("/archetype/{user_id}")
async def get_archetype(user_id: str):
    """Get current personality archetype"""
    try:
        snapshot = personality_engine.get_user_personality(user_id)
        
        if snapshot:
            archetype_info = personality_engine.archetypes.get(snapshot.archetype, {})
            return {
                "status": "success",
                "user_id": user_id,
                "archetype": snapshot.archetype,
                "description": archetype_info.get('description', 'Unknown archetype'),
                "response_style": archetype_info.get('response_style', 'adaptive'),
                "confidence": snapshot.confidence,
                "stability": snapshot.archetype_stability
            }
        else:
            return {
                "status": "success",
                "user_id": user_id,
                "archetype": "adaptive_sage",
                "description": "Balanced, wise, thoughtful",
                "response_style": "measured and insightful",
                "confidence": 0.5,
                "stability": 1.0
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archetype retrieval failed: {str(e)}")

@app.get("/backends")
async def list_backends():
    """List available model backends"""
    active_backend, _ = get_active_backend()
    return {
        "status": "success",
        "backends": MODEL_BACKENDS,
        "active_backend": active_backend
    }

@app.post("/backends/{backend_name}/activate")
async def activate_backend(backend_name: str):
    """Activate a specific model backend"""
    if backend_name not in MODEL_BACKENDS:
        raise HTTPException(
            status_code=404,
            detail=json.dumps({
                "status": "error",
                "error_code": "BACKEND_NOT_FOUND",
                "message": f"Backend '{backend_name}' not found",
                "details": {"available_backends": list(MODEL_BACKENDS.keys())}
            })
        )
    
    # Deactivate all backends
    for name in MODEL_BACKENDS:
        MODEL_BACKENDS[name]["enabled"] = False
    
    # Activate the requested backend
    MODEL_BACKENDS[backend_name]["enabled"] = True
    
    return {
        "status": "success",
        "message": f"Backend '{backend_name}' activated",
        "backend_config": MODEL_BACKENDS[backend_name]
    }

@app.post("/benchmark")
async def benchmark_response(request: dict):
    """Benchmark response generation performance"""
    try:
        user_id = request.get("user_id", "benchmark_user")
        messages = request.get("messages", ["Hello, how are you?"])
        iterations = request.get("iterations", 1)
        
        import time
        results = []
        
        for i in range(iterations):
            for message in messages:
                start_time = time.time()
                
                # Generate response
                snapshot = personality_engine.analyze_user_interaction(user_id, message)
                traits = snapshot.big_five_scores
                mood = snapshot.mood_vector
                archetype_info = personality_engine.archetypes.get(snapshot.archetype, {})
                base_response = _generate_contextual_response(message, traits, mood, archetype_info, {})
                response = _apply_personality_modulation(base_response, snapshot, archetype_info)
                
                end_time = time.time()
                
                results.append({
                    "iteration": i + 1,
                    "message": message,
                    "response_length": len(response),
                    "latency_ms": (end_time - start_time) * 1000,
                    "archetype": snapshot.archetype,
                    "confidence": snapshot.confidence
                })
        
        # Calculate metrics
        latencies = [r["latency_ms"] for r in results]
        response_lengths = [r["response_length"] for r in results]
        
        return {
            "status": "success",
            "metrics": {
                "total_requests": len(results),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "avg_response_length": sum(response_lengths) / len(response_lengths),
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)]
            },
            "results": results[:10]  # Return first 10 results for brevity
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=json.dumps({
                "status": "error",
                "error_code": "BENCHMARK_FAILED",
                "message": f"Benchmark failed: {str(e)}",
                "details": {"user_id": request.get("user_id", "unknown")}
            })
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
