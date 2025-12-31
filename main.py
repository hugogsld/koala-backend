from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# =========================
# Load .env (robuste)
# =========================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # ✅ charge Koala-backend/.env même si tu lances uvicorn ailleurs

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
  raise RuntimeError("OPENAI_API_KEY n'est pas défini dans l'environnement (ajoute-le dans Koala-backend/.env).")

# =========================
# App
# =========================
app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# =========================
# OpenAI Client
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Models
# =========================
class ChatMsg(BaseModel):
  role: Literal["user", "assistant"]
  content: str


class ChatIn(BaseModel):
  text: str
  messages: Optional[List[ChatMsg]] = None

# =========================
# Routes
# =========================
@app.get("/health")
def health():
  return {"status": "ok", "service": "koala-api"}


@app.post("/ai/chat")
def ai_chat(payload: ChatIn):
  try:
    history = []
    if payload.messages:
      history = [{"role": m.role, "content": m.content} for m in payload.messages]

    if not history:
      history = [{"role": "user", "content": payload.text}]

    response = client.chat.completions.create(
      model="gpt-4.1-mini",
      messages=[
        {
          "role": "system",
          "content": (
            "Tu es Koala, un assistant vocal bienveillant et clair. "
            "Tu réponds en français, de façon concise, utile et actionnable. "
            "Si la demande est floue, tu poses 1 question courte."
          ),
        },
        *history,
      ],
    )

    return {"answer": response.choices[0].message.content}

  except Exception as e:
    msg = str(e)

    if "insufficient_quota" in msg or "Error code: 429" in msg:
      raise HTTPException(
        status_code=429,
        detail="Quota OpenAI dépassé ou facturation inactive. Vérifie ton plan/billing OpenAI.",
      )

    if "invalid_api_key" in msg or "Error code: 401" in msg:
      raise HTTPException(status_code=401, detail="Clé OpenAI invalide.")

    raise HTTPException(status_code=500, detail=f"Erreur serveur: {msg}")
