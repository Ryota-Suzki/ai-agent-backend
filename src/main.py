import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.src.config import settings

from backend.src.agent.ai_agent import create_graph, State, get_ollama_embeddings
from backend.src.qdrant.qdrant import search_code_knowledge 

app = FastAPI(title="AI Agent API")
logger = logging.getLogger("uvicorn.error")

# ===== CORS設定 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ====================


graph = create_graph()

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    logger.info("🚀 [STREAM START] クライアントとの接続を確立しました")
    try:
        points = search_code_knowledge(
            query=req.message,
            embeddings_model=get_ollama_embeddings(),
            repository_id=settings.repository_id
        )

        logger.info(f"Qdrant Search hits: {len(points)}")
        
        hits = [
            hit.payload["content"]
            for hit in points 
            if hit.payload and "content" in hit.payload and hit.score > 0.6
        ]
        knowledge_context = "\n---\n".join(hits) if hits else "該当するコードナレッジは見つかりませんでした。"
        
        async def stream_generator():
            input_data: State = {
                "input": req.message,
                "search_results": knowledge_context,
                "output": "",
                "messages": []
            }
            
            async for event in graph.astream_events(input_data, version="v2"):
                kind = event["event"]
                
                if kind == "on_llm_stream":
                    chunk = event["data"]["chunk"]
                    
                    if isinstance(chunk, str):
                        text = chunk
                    elif hasattr(chunk, "text"):
                        text = chunk.text
                    elif hasattr(chunk, "content"):
                        text = chunk.content
                    else:
                        text = str(chunk)
                        
                    if text:
                        yield text

        return StreamingResponse(
            stream_generator(),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error in chat streaming: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))