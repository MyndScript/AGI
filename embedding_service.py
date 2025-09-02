#!/usr/bin/env python3
"""
AGI Embedding Service
Production-ready embedding server for the AGI ecosystem.
Supports multiple embedding models for different use cases.
"""

import os
import logging
from typing import List, Dict
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agi_embedding_service")

# Environment configuration
EMBEDDING_PORT = int(os.getenv('AGI_EMBEDDING_PORT', 8003))

app = FastAPI(title="AGI Embedding Service", version="1.0.0")

# Model registry - production-ready models
MODELS = {
    "semantic": "all-mpnet-base-v2",           # General semantic understanding (768 dims)
    "personality": "BAAI/bge-large-en-v1.5",  # Best for personality analysis (1024 dims) 
    "fast": "all-MiniLM-L6-v2",               # Quick queries (384 dims)
    "code": "microsoft/codebert-base",         # Code-aware embeddings (768 dims)
    "emotion": "all-mpnet-base-v2",            # For emotional content analysis
    "default": "all-mpnet-base-v2"             # Fallback model
}

# Global model cache
loaded_models: Dict[str, SentenceTransformer] = {}

class EmbedRequest(BaseModel):
    text: str
    model: str = "default"
    normalize: bool = True

class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    dimensions: int
    text_length: int

class BatchEmbedRequest(BaseModel):
    texts: List[str]
    model: str = "default"
    normalize: bool = True

class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int
    count: int

def load_model(model_key: str) -> SentenceTransformer:
    """Load and cache embedding model."""
    if model_key not in loaded_models:
        model_name = MODELS.get(model_key, MODELS["default"])
        logger.info(f"Loading embedding model: {model_name}")
        try:
            loaded_models[model_key] = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback to default if available
            if model_key != "default" and "default" not in loaded_models:
                logger.info("Loading fallback model: all-MiniLM-L6-v2")
                loaded_models["default"] = SentenceTransformer("all-MiniLM-L6-v2")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return loaded_models[model_key]

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "embedding_service",
        "port": EMBEDDING_PORT,
        "loaded_models": list(loaded_models.keys()),
        "available_models": list(MODELS.keys())
    }

@app.get("/models")
def list_models():
    """List available embedding models."""
    return {
        "available_models": MODELS,
        "loaded_models": list(loaded_models.keys())
    }

@app.post("/embed", response_model=EmbedResponse)
def embed_text(request: EmbedRequest):
    """Generate embedding for a single text."""
    try:
        model = load_model(request.model)
        
        # Generate embedding
        embedding = model.encode(
            request.text, 
            normalize_embeddings=request.normalize,
            convert_to_numpy=True
        )
        
        # Convert to Python list
        embedding_list = embedding.tolist()
        
        logger.info(f"Generated embedding for text (length: {len(request.text)}) using {request.model}")
        
        return EmbedResponse(
            embedding=embedding_list,
            model=request.model,
            dimensions=len(embedding_list),
            text_length=len(request.text)
        )
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/batch", response_model=BatchEmbedResponse)
def embed_texts(request: BatchEmbedRequest):
    """Generate embeddings for multiple texts (more efficient)."""
    try:
        model = load_model(request.model)
        
        # Generate embeddings
        embeddings = model.encode(
            request.texts, 
            normalize_embeddings=request.normalize,
            convert_to_numpy=True
        )
        
        # Convert to Python lists
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        logger.info(f"Generated {len(embeddings_list)} embeddings using {request.model}")
        
        return BatchEmbedResponse(
            embeddings=embeddings_list,
            model=request.model,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            count=len(embeddings_list)
        )
        
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity")
def compute_similarity(request: dict):
    """Compute cosine similarity between two texts."""
    try:
        text1 = request.get("text1")
        text2 = request.get("text2")
        model_key = request.get("model", "default")
        
        if not text1 or not text2:
            raise HTTPException(status_code=400, detail="Both text1 and text2 required")
        
        model = load_model(model_key)
        
        # Generate embeddings
        embeddings = model.encode([text1, text2], normalize_embeddings=True)
        
        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1])
        
        return {
            "similarity": float(similarity),
            "model": model_key,
            "text1_length": len(text1),
            "text2_length": len(text2)
        }
        
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load default model on startup."""
    logger.info("Starting AGI Embedding Service...")
    try:
        # Preload the default model for faster first requests
        load_model("default")
        logger.info("AGI Embedding Service ready!")
    except Exception as e:
        logger.error(f"Failed to preload default model: {e}")

if __name__ == "__main__":
    logger.info(f"Starting AGI Embedding Service on port {EMBEDDING_PORT}")
    uvicorn.run(
        "embedding_service:app",
        host="0.0.0.0",
        port=EMBEDDING_PORT,
        reload=False
    )
