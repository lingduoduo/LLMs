#!/usr/bin/env python3
"""LLM Intelligent Gateway"""
import asyncio
import json
import time
import uuid
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "Missing OPENAI_API_KEY. Please set it in your environment or in a .env file."
    )

app = FastAPI(title="LLM Intelligent Gateway", version="1.0.0")

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

client = OpenAI( api_key=OPENAI_API_KEY)

# Engine configuration
engines = {
    "fast": {"model": "gpt-3.5-turbo", "tier": "Fast"},
    "balanced": {"model": "gpt-4", "tier": "Balanced"},
    "premium": {"model": "o3-mini", "tier": "Premium"},
}


def select_engine(question: str):
    """Intelligently select an engine"""
    content = question.lower()

    # Complexity keyword matching
    complex_words = ["design", "architecture", "analysis", "system", "algorithm", "optimization"]
    medium_words = ["explain", "principle", "method", "process"]
    simple_words = ["what is", "definition", "translate"]

    if any(word in content for word in complex_words) or len(question) > 50:
        engine_type = "premium"
    elif any(word in content for word in medium_words) or len(question) > 20:
        engine_type = "balanced"
    else:
        engine_type = "fast"

    engine = engines[engine_type]
    print(f"Selected engine: {engine['model']} ({engine['tier']})")
    return engine


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completion requests"""
    user_question = request.messages[-1].content
    engine = select_engine(user_question)

    try:
        if request.stream:
            return StreamingResponse(
                stream_response(request, engine),
                media_type="text/plain",
            )
        else:
            return await complete_response(request, engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(request: ChatRequest, engine):
    """Streaming response"""
    if client:
        try:
            response = client.chat.completions.create(
                model=engine["model"],
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                stream=True,
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    data = {"choices": [{"delta": {"content": chunk.choices[0].delta.content}}]}
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception:
            pass

    # Simulated response
    for part in ["This", "is", "a", engine["tier"], "model", "response"]:
        data = {"choices": [{"delta": {"content": part}}]}
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.2)
    yield "data: [DONE]\n\n"


async def complete_response(request: ChatRequest, engine):
    """Full (non-streaming) response"""
    print(f"Processing request using {engine['tier']} model...")

    if client:
        try:
            response = client.chat.completions.create(
                model=engine["model"],
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                timeout=25,  # Set API call timeout
            )

            return {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content,
                        },
                        "finish_reason": response.choices[0].finish_reason,
                    }
                ],
            }
        except Exception as e:
            print(f"API call failed, using simulated response: {e}")

    # Simulated response - generate answers with different lengths based on complexity
    user_question = request.messages[-1].content
    if engine["tier"] == "Fast":
        content = f"This is a quick answer to a simple question: {user_question[:20]}..."
    elif engine["tier"] == "Balanced":
        content = (
            f"This is a more detailed answer for a moderately complex question. "
            f"Question: {user_question}. This requires more comprehensive analysis and explanation."
        )
    else:  # Premium
        content = (
            f"This is an in-depth analysis for a complex question: {user_question}. "
            f"It needs to be addressed from multiple angles, including technical architecture, "
            f"implementation plans, performance optimization, and other key considerations."
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "engines": len(engines)}


@app.get("/stats")
async def stats():
    return {"engine_types": list(engines.keys()), "api_available": client is not None}


if __name__ == "__main__":
    print("LLM Intelligent Gateway starting up")
    print("Address: http://localhost:8000")
    print(f"Engines: {len(engines)}")
    print(f"API: {'Available' if client else 'Simulated'}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
