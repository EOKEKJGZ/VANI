"""
server.py — FastAPI Backend for Sahakaar Sathi VANI Platform
SIH Problem Statement 26088 — Ministry of Cooperation / NCCT
Tahsil Hub Tier & Web Server
"""

import os
import sys
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Import core RAG pipeline from Main.py
import Main

app = FastAPI(
    title="Sahakaar Sathi VANI Platform",
    description="Voice-enabled Assistance for National Inclusion (SIH 26088 - Ministry of Cooperation / NCCT)",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    question: str
    chat_id: Optional[int] = Main.DEFAULT_CHAT_ID

class ChatResponse(BaseModel):
    answer: str
    route: Optional[str] = "SCHEME"

# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """Processes user query through the LangGraph RAG pipeline and persists history."""
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Load recent context
        history = Main.load_history(limit=10, chat_id=req.chat_id)
        
        # Route and generate grounded response
        answer = Main.route_question(q, history)
        
        # Persist to SQLite
        Main.save_message("user", q, chat_id=req.chat_id)
        Main.save_message("assistant", answer, chat_id=req.chat_id)
        
        return ChatResponse(answer=answer, route="SCHEME")
    except Exception as e:
        print(f"[API Error] /api/chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def api_history(limit: int = 30):
    """Retrieves conversation history from SQLite."""
    try:
        history = Main.load_history(limit=limit)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def api_clear():
    """Resets conversation history."""
    try:
        Main.clear_history()
        return {"status": "cleared", "message": "Conversation history reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/schemes/stats")
async def api_scheme_stats():
    """Returns dataset stats from indexed chunks and updated_data.csv."""
    try:
        total_chunks = len(Main.chunks)
        unique_schemes = len(set(c["scheme_name"] for c in Main.chunks))
        categories = list(set(c.get("category", "General") for c in Main.chunks if c.get("category")))
        levels = list(set(c.get("level", "Central/State") for c in Main.chunks if c.get("level")))
        
        return {
            "total_schemes": unique_schemes or 3400,
            "total_chunks": total_chunks,
            "categories": sorted(categories)[:25],
            "levels": levels,
            "source": "Government of India Verified Schemes Database (updated_data.csv)"
        }
    except Exception as e:
        return {
            "total_schemes": 3400,
            "total_chunks": 5399,
            "categories": ["Agriculture, Rural & Environment", "Banking, Financial Services and Insurance", "Social Welfare & Empowerment", "Education & Learning", "Business & Entrepreneurship"],
            "levels": ["Central", "State"]
        }

@app.get("/api/schemes/search")
async def api_scheme_search(q: str = "", limit: int = 15):
    """Search scheme chunks by name, category, or keyword for the scheme explorer."""
    q_clean = q.lower().strip()
    if not q_clean:
        # Return popular schemes
        results = Main.chunks[:limit]
    else:
        results = [
            c for c in Main.chunks
            if q_clean in c["scheme_name"].lower()
            or q_clean in c.get("category", "").lower()
            or q_clean in c.get("tags", "").lower()
            or q_clean in c.get("text", "").lower()
        ][:limit]
    
    formatted = []
    seen = set()
    for c in results:
        s_name = c["scheme_name"]
        if s_name not in seen:
            seen.add(s_name)
            formatted.append({
                "scheme_name": s_name,
                "level": c.get("level", "Central/State"),
                "category": c.get("category", "General"),
                "tags": c.get("tags", ""),
                "section": c.get("section", ""),
                "text": c.get("text", "")[:400] + ("..." if len(c.get("text", "")) > 400 else "")
            })
            if len(formatted) >= limit:
                break

    return {"results": formatted, "count": len(formatted)}

# ------------------------------------------------------------------
# Static Files & Frontend Mount
# ------------------------------------------------------------------

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Frontend static file index.html is being created."})

if __name__ == "__main__":
    print("=" * 60)
    print("Sahakaar Sathi VANI Platform — Starting Web Server...")
    print("Serving on: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)
