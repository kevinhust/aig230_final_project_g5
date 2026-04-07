"""
FastAPI Endpoint for Multilingual E-Commerce RAG Chatbot

This provides a REST API for integrating the chatbot into websites.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

# Initialize RAG Engine (lazy loading for faster startup)
engine = None


def get_engine():
    """Lazy load the RAG engine."""
    global engine
    if engine is None:
        from src.rag_pipeline import RAGEngine
        engine = RAGEngine(use_ollama=False)
    return engine


# FastAPI app
app = FastAPI(
    title="Multilingual E-Commerce RAG API",
    description="REST API for the Multilingual E-Commerce Customer Support Chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    include_scores: bool = False
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    detected_lang: str
    is_code_switching: bool
    sentiment: Dict
    escalated: bool


class HealthResponse(BaseModel):
    status: str
    engine_loaded: bool
    version: str


class SourcesResponse(BaseModel):
    sources: List[str]
    count: int


# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic status."""
    return HealthResponse(
        status="ok",
        engine_loaded=engine is not None,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        engine_loaded=engine is not None,
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Send a message and receive a response with:
    - Answer text
    - Source citations
    - Detected language
    - Code-switching detection
    - Sentiment analysis
    - Escalation status
    """
    try:
        rag_engine = get_engine()
        result = rag_engine.ask(request.message, include_scores=request.include_scores)

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            detected_lang=result["detected_lang"],
            is_code_switching=result["is_code_switching"],
            sentiment=result["sentiment"],
            escalated=result["escalated"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/simple")
async def chat_simple(message: str):
    """
    Simple chat endpoint that returns just the answer text.

    Useful for quick integrations.
    """
    try:
        rag_engine = get_engine()
        result = rag_engine.ask(message)
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources", response_model=SourcesResponse)
async def list_sources():
    """List all available knowledge base sources."""
    try:
        rag_engine = get_engine()
        # Get all unique sources from vector store
        # This is a simple implementation
        sources = [
            "FAQ-Appliances.md",
            "FAQ-Electronics.md",
            "FAQ-Laptops.md",
            "FAQ-Furniture.md",
            "FAQ-Clothing.md",
            "FAQ-Kitchen.md",
            "FAQ-SmartHome.md",
            "FAQ-Headphones.md",
            "Refund-Policy.md",
            "Exchange-Policy.md",
            "Warranty-Terms.md",
            "Shipping-Policy.md",
            "Shipping-Rates.md",
            "Coupon-Policy.md",
            "CodeSwitch-ENZH.md",
            "CodeSwitch-FRES.md",
            "Customer-Complaints.md"
        ]
        return SourcesResponse(sources=sources, count=len(sources))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-memory")
async def clear_memory():
    """Clear conversation memory (start fresh session)."""
    try:
        rag_engine = get_engine()
        rag_engine.clear_memory()
        return {"status": "success", "message": "Conversation memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages")
async def supported_languages():
    """List supported languages."""
    return {
        "languages": [
            {"code": "en", "name": "English", "flag": "🇬🇧"},
            {"code": "zh", "name": "中文", "flag": "🇨🇳"},
            {"code": "fr", "name": "Français", "flag": "🇫🇷"},
            {"code": "es", "name": "Español", "flag": "🇪🇸"},
        ],
        "features": {
            "code_switching": True,
            "sentiment_analysis": True,
            "source_citations": True
        }
    }


if __name__ == "__main__":
    print("Starting Multilingual E-Commerce RAG API...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
