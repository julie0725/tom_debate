"""
api_server.py
-------------
FastAPI 백엔드 서버 (SSE 진행률 포함)
실행: uvicorn api_server:app --reload --port 8000
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import queue
import threading
from typing import AsyncGenerator
import uuid
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from user.ai_user import AIUser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "..", path), encoding="utf-8") as f:
        return yaml.safe_load(f)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config()
session_queues: dict[str, queue.Queue] = {}

class RunRequest(BaseModel):
    scenario: str
    question: str
    question2: str = ""
    question3: str = ""


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def run_pipeline_sse(scenario: str, question: str, question2: str = "", question3: str = "", session_id: str = "") -> AsyncGenerator[str, None]:
    q: queue.Queue = queue.Queue()

    eq = session_queues.get(session_id)

    def event_callback(event: str, data: dict):
        if eq is not None:
            eq.put((event, data))

    loop = asyncio.get_event_loop()

    def progress_callback(message: str, pct: int):
        q.put(("progress", {"message": message, "pct": pct}))

    def _run():
        try:
            ai_user = AIUser(config=config)
            ai_user.progress_callback = progress_callback
            ai_user.event_callback = event_callback
            metadata = {}
            if question2.strip():
                metadata["q2"] = question2.strip()
            if question3.strip():
                metadata["q3"] = question3.strip()
            state = ai_user.submit_from_text(scenario=scenario, question=question, metadata=metadata or None)
            q.put(("done", {
                "status": state.status,
                "debate_round": state.debate_round,
                "debate_triggered": state.debate_triggered,
                "q1": state.final_answer.get_value("q1"),
                "q2": state.final_answer.get_value("q2"),
                "q3": state.final_answer.get_value("q3"),
            }))
        except Exception as e:
            q.put(("error", {"message": str(e)}))

    yield sse("progress", {"message": "파이프라인 시작 중...", "pct": 5})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        try:
            event, data = await loop.run_in_executor(None, lambda: q.get(timeout=120))
            yield sse(event, data)
            if event in ("done", "error"):
                break
        except queue.Empty:
            yield sse("error", {"message": "Timeout: pipeline took too long"})
            break

    thread.join(timeout=5)


@app.post("/run")
async def run(req: RunRequest):
    session_id = str(uuid.uuid4())
    session_queues[session_id] = queue.Queue()

    async def _stream():
        yield sse("session", {"session_id": session_id})
        async for chunk in run_pipeline_sse(req.scenario, req.question, req.question2, req.question3, session_id):
            yield chunk
        session_queues.pop(session_id, None)

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/debate-stream/{session_id}")
async def debate_stream(session_id: str):
    async def _stream():
        if session_id not in session_queues:
            yield sse("error", {"message": "session not found"})
            return
        eq = session_queues[session_id]
        loop = asyncio.get_event_loop()
        while True:
            try:
                event, data = await loop.run_in_executor(None, lambda: eq.get(timeout=120))
                yield sse(event, data)
                if event in ("done", "error"):
                    break
            except queue.Empty:
                yield sse("error", {"message": "timeout"})
                break

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/health")
def health():
    return {"status": "ok"}