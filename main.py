from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not defined.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "capacitor://localhost",
        "ionic://localhost",
        "http://localhost",
        "https://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)


class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatIn(BaseModel):
    text: str
    messages: Optional[List[ChatMsg]] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "koala-api"}


@app.post("/ai/chat")
def ai_chat(payload: ChatIn):
    try:
        history = (
            [{"role": m.role, "content": m.content} for m in payload.messages]
            if payload.messages
            else [{"role": "user", "content": payload.text}]
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Koala, a clear and helpful AI assistant for second-hand resellers. "
                        "You respond in French, in a concise and actionable way. "
                        "If the request is unclear, ask one short clarifying question."
                    ),
                },
                *history,
            ],
        )

        return {"answer": response.choices[0].message.content}

    except Exception as e:
        msg = str(e)

        if "insufficient_quota" in msg or "429" in msg:
            raise HTTPException(
                status_code=429,
                detail="OpenAI quota exceeded or billing inactive.",
            )

        if "invalid_api_key" in msg or "401" in msg:
            raise HTTPException(status_code=401, detail="Invalid OpenAI API key.")

        raise HTTPException(status_code=500, detail="Internal server error.")
