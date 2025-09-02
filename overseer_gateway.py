"""
Overseer Gateway: Central API router for AGI system
Listens on port 8010 and proxies requests to Go (memory), Python (personality on 8002, agent/global on 8000) backends.
"""


from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Restrict to frontend only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GO_MEMORY_URL = "http://localhost:8001"
PYTHON_PERSONALITY_URL = "http://localhost:8002"  # Ensure personality server runs on 8002
PYTHON_AGENT_URL = "http://localhost:8000"

@app.get("/health")
async def health_check():
    """Health check endpoint for the overseer gateway"""
    return {
        "status": "healthy",
        "service": "overseer_gateway",
        "port": 8010,
        "timestamp": datetime.now().isoformat()
    }

# Endpoint to backend mapping (all memory endpoints now use Go backend)
ENDPOINT_MAP = {
    # Agent endpoints
    "import-facebook": PYTHON_AGENT_URL,
    "import-facebook-public": PYTHON_AGENT_URL,
    "get-observation": PYTHON_AGENT_URL,
    "generate": PYTHON_AGENT_URL,
    "text-generation": PYTHON_AGENT_URL,
    "facebook-summary": PYTHON_AGENT_URL,
    "abigail/trigger": PYTHON_AGENT_URL,
    "abigail/glyph": PYTHON_AGENT_URL,
    "abigail/mood": PYTHON_AGENT_URL,
    "abigail/archetype": PYTHON_AGENT_URL,
    "abigail/memory": PYTHON_AGENT_URL,
    # Journal endpoints
    "add-journal-entry": PYTHON_AGENT_URL,
    "get-journal-entries": PYTHON_AGENT_URL,
    "analyze-journal-entry": PYTHON_AGENT_URL,
    # Memory endpoints (Go only)
    "set-global-memory": GO_MEMORY_URL,
    "get-user-context": GO_MEMORY_URL,
    "semantic_search": GO_MEMORY_URL,
    "set-memory": GO_MEMORY_URL,
    "get-global-memory": GO_MEMORY_URL,
    "set-global-moment": GO_MEMORY_URL,
    "get-global-moments": GO_MEMORY_URL,
    "get-memory": GO_MEMORY_URL,
    "get_moments": GO_MEMORY_URL,
    # Personality endpoints (all routed to port 8002)
    "add-observation": PYTHON_PERSONALITY_URL,
    "set-personality": PYTHON_PERSONALITY_URL,
    "get-personality": PYTHON_PERSONALITY_URL,
    "get-personality-context": PYTHON_PERSONALITY_URL,
}

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(full_path: str, request: Request):
    # Remove leading slash if present
    endpoint = full_path.lstrip("/")
    # Route to backend based on first segment
    first_segment = endpoint.split("/")[0]
    if first_segment == "memory":
        target = GO_MEMORY_URL
    elif first_segment == "agent":
        target = PYTHON_AGENT_URL
    elif first_segment == "personality":
        target = PYTHON_PERSONALITY_URL
    elif first_segment == "abigail":
        target = PYTHON_AGENT_URL
    else:
        # Fallback to endpoint map for legacy direct endpoints
        target = ENDPOINT_MAP.get(endpoint)
    print(f"Routing {request.method} /{full_path} to {target if target else 'UNKNOWN'}")
    if not target:
        return Response(content="Unknown endpoint", status_code=404)
    # Strip first segment for backend routing
    segments = full_path.lstrip("/").split("/")
    if first_segment in ["memory", "agent", "personality", "abigail"]:
        backend_path = "/" + "/".join(segments[1:]) if len(segments) > 1 else "/"
    else:
        backend_path = "/" + full_path.lstrip("/")
    print(f"Forwarding to {target}{backend_path}")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.request(
                request.method,
                f"{target}{backend_path}",
                content=await request.body(),
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            )
            # Ensure CORS headers are present in all responses
            response_headers = dict(resp.headers)
            response_headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            response_headers["Access-Control-Allow-Credentials"] = "true"
            response_headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response_headers["Access-Control-Allow-Headers"] = "*"
            return Response(content=resp.content, status_code=resp.status_code, headers=response_headers)
    except httpx.ReadTimeout:
        print(f"[Gateway] Timeout routing {request.method} {full_path} to {target}{backend_path}")
        return Response(content=f"Gateway timeout: backend {target} did not respond in time.", status_code=504)
    except httpx.RequestError as e:
        print(f"[Gateway] Request error routing {request.method} {full_path} to {target}{backend_path}: {e}")
        return Response(content=f"Gateway error: {str(e)}", status_code=502)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
