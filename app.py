# main.py (Google GenAI-based)
import os
import json
import secrets
import hashlib
import hmac
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import requests
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Google GenAI SDK
from google import genai

# local DB helpers (your existing module)
from db_schema import (
    init_db,
    get_setting,
    set_setting,
    append_message,
    get_conversation,
    create_user,
    get_user,
    create_session,
    get_session,
    list_conversations,
)

# initialize DB
init_db()

# ---------------------------
# GenAI client config
# ---------------------------
# Use GEMINI_API_KEY or GOOGLE_API_KEY environment variable
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    # allow client to pick up env var if user prefers, but warn
    # you can set GEMINI_API_KEY or GOOGLE_API_KEY in your environment
    print("Warning: No GEMINI_API_KEY/GOOGLE_API_KEY found in environment; genai.Client() may pick up defaults.")

# create GenAI client (Gemini Developer API)
# If you want Vertex AI usage, set vertexai=True and set project/location envs as described in docs.
client = genai.Client(api_key=API_KEY)

# default model names (adjust via env if desired)
GENAI_CHAT_MODEL = os.getenv("GENAI_CHAT_MODEL", "gemini-2.5-flash")
GENAI_EMBED_MODEL = os.getenv("GENAI_EMBED_MODEL", "gemini-embedding-001")
SIMILARITY_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.68"))

# ---------------------------
# Configuration & env
# ---------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ARTICLE_PATH = DATA_DIR / "article.txt"
if not ARTICLE_PATH.exists():
    raise FileNotFoundError("Place article.txt at data/article.txt")

# ---------------------------
# Admin defaults (include user_prompt)
# ---------------------------
DEFAULT_ADMIN_SETTINGS = {
    "welcome": "Bun venit! Cu ce vă pot ajuta astăzi?",
    "fallback": "Ne pare rău — nu am găsit un răspuns în baza noastră. Vă rugăm să contactați suportul.",
    "tone": "Răspunde în limba română într-un ton prietenos și concis.",
    "user_prompt": "Use the following context to answer the question concisely. Provide a short source note if relevant.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
}

# DEFAULT_ADMIN_SETTINGS = {
#     "welcome": "Welcome! How can I help you today?",
#     "fallback": "We're sorry — we couldn't find an answer in our database. Please contact support.",
#     "tone": "You are a helpful assistant. Respond in a friendly and concise tone.",
#     "user_prompt": "Use the following context to answer the question concisely. Provide a short source note if relevant.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
# }

if get_setting("admin") is None:
    set_setting("admin", DEFAULT_ADMIN_SETTINGS)

# ---------------------------
# admin user (simple)
# ---------------------------
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex() + ":" + dk.hex()

def verify_password(stored: str, password: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        dk_new = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(dk_new.hex(), dk_hex)
    except Exception:
        return False

if get_user("admin") is None:
    create_user("admin", hash_password("admin123"))

# ---------------------------
# GenAI wrappers: embeddings & generation
# ---------------------------
def genai_embed_batch(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Use Google GenAI client to embed a batch of texts.
    Returns list of vectors (list of floats) for each text.
    """
    m = model or GENAI_EMBED_MODEL
    try:
        # Many GenAI SDKs expose an embeddings helper, but here we call models.embed_content
        resp = client.models.embed_content(model=m, contents=texts)
        # resp may be a list, dict, or an object. Try a few ways to extract embeddings.
        # 1) If it's a list of vectors -> return it
        if isinstance(resp, list):
            return resp
        # 2) If it has attribute 'embeddings' or 'embedding' or 'data'
        if hasattr(resp, "embeddings"):
            emb_field = getattr(resp, "embeddings")
            # sometimes that is the raw vectors
            if isinstance(emb_field, list):
                return emb_field
        # 3) If it's a mapping
        if isinstance(resp, dict):
            for key in ("embeddings", "data", "vectors", "embedding", "vectors_list"):
                if key in resp:
                    return resp[key]
        # 4) try to see if resp has .to_dict()
        try:
            d = resp.to_dict() if hasattr(resp, "to_dict") else None
            if d:
                for key in ("embeddings", "data", "vectors", "embedding"):
                    if key in d:
                        return d[key]
        except Exception:
            pass
        # As a fallback: if resp has a 'text' or string representation, raise to help debugging
        raise RuntimeError(f"Unexpected embeddings response shape: {type(resp)} / {repr(resp)[:400]}")
    except Exception as e:
        raise RuntimeError(f"GenAI embedding request failed: {e}")

def genai_embed_one(text: str, model: Optional[str] = None) -> List[float]:
    res = genai_embed_batch([text], model=model)
    if not isinstance(res, list) or len(res) == 0:
        raise RuntimeError("Empty embedding from GenAI")
    # res[0] may be a dict with 'embedding' key, or the raw vector
    first = res[0]
    if isinstance(first, dict) and "embedding" in first:
        return list(first["embedding"])
    return list(first)

def genai_generate(system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
    """
    Use Google GenAI client to generate text. Returns the text string.
    Sends system and user prompts as two contents parts (system first).
    """
    m = model or GENAI_CHAT_MODEL
    try:
        # pass system prompt and user prompt as list - SDK will convert to appropriate Content structures
        # Many SDK examples use client.models.generate_content(model=..., contents=[system, user])
        resp = client.models.generate_content(model=m, contents=[system_prompt, user_prompt])
        # resp may have .text attribute or .parts; prefer resp.text
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # If resp has .parts, join text from parts
        if hasattr(resp, "parts") and getattr(resp, "parts") is not None:
            parts = getattr(resp, "parts")
            texts = []
            for p in parts:
                # p may have inline_data or text methods
                try:
                    if hasattr(p, "text") and p.text:
                        texts.append(p.text)
                    elif hasattr(p, "as_text"):
                        texts.append(p.as_text())
                    else:
                        texts.append(str(p))
                except Exception:
                    texts.append(str(p))
            return "\n".join(texts).strip()
        # If resp is mapping-like
        if isinstance(resp, dict):
            if "text" in resp:
                return resp["text"]
            # try to inspect parts/data
            if "parts" in resp and isinstance(resp["parts"], list):
                return " ".join([str(x.get("text") or x) for x in resp["parts"]])
        # fallback to stringified resp
        return str(resp)
    except Exception as e:
        raise RuntimeError(f"GenAI generate request failed: {e}")

# ---------------------------
# Build KB: chunk + embed (using GenAI)
# ---------------------------
print("Loading article and preparing chunks (GenAI embeddings)...")
article_text = ARTICLE_PATH.read_text(encoding="utf-8")

def chunk_text(text: str) -> list:
    """
    LLM-assisted chunker: takes raw article text and returns a list of chunk strings,
    split by natural sections or questions (question-wise), preserving original text.
    """
    import re
    import json

    # Prompt LLM to split text into chunks/questions
    prompt = f"""
    You are an AI assistant. Split the following article text into logical chunks based on questions or headings.
    Preserve the text exactly. Return ONLY a JSON array of strings (each string is a chunk).

    Article:
    {text}
    """

    try:
        # Use your GenAI wrapper to generate the chunked JSON
        response = genai_generate(
            system_prompt="Split article into question-wise chunks as JSON.",
            user_prompt=prompt
        )

        # Extract JSON array from response
        match = re.search(r"\[.*\]", response, flags=re.DOTALL)
        if match:
            chunks = json.loads(match.group())
        else:
            # fallback: return whole text as a single chunk
            chunks = [text]

    except Exception:
        # fallback: return whole text as a single chunk
        chunks = [text]

    # Clean up
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks

chunks = chunk_text(article_text)
print(f"Created {len(chunks)} chunks.")

# embed all chunks via GenAI
print("Requesting embeddings for chunks from Google GenAI...")

try:
    raw_embs = genai_embed_batch(chunks, model=GENAI_EMBED_MODEL)

    normalized_embs = []
    for e in raw_embs:
        # Case 1: new SDK → ContentEmbedding object
        if hasattr(e, "values"):
            normalized_embs.append(np.array(e.values, dtype=float))
        # Case 2: dict returned
        elif isinstance(e, dict) and "embedding" in e:
            normalized_embs.append(np.array(e["embedding"], dtype=float))
        # Case 3: plain list
        elif isinstance(e, list):
            normalized_embs.append(np.array(e, dtype=float))
        else:
            raise RuntimeError(f"Unknown embedding item type: {type(e)} {e!r}")

    chunk_vectors = normalized_embs
    print(f"Received {len(chunk_vectors)} chunk embeddings (dim {len(chunk_vectors[0]) if chunk_vectors else 0}).")

except Exception as e:
    raise RuntimeError(f"Failed to get embeddings from GenAI: {e}")


def embed_query(q: str):
    try:
        result = genai_embed_batch([q], model=GENAI_EMBED_MODEL)

        # EXPECTS result = [ContentEmbedding] or [{"embedding": [...] }]
        e = result[0]

        if hasattr(e, "values"):
            return np.array(e.values, dtype=float)

        if isinstance(e, dict) and "embedding" in e:
            return np.array(e["embedding"], dtype=float)

        if isinstance(e, list):
            return np.array(e, dtype=float)

        raise RuntimeError(f"Unknown embedding format for query: {type(e)} {e}")

    except Exception as e:
        raise RuntimeError(f"Failed to embed query via GenAI: {e}")

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def top_k_chunks(q: str, k=3):
    qv = embed_query(q)
    sims = [cosine_sim(qv, v) for v in chunk_vectors]
    idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return [(i, sims[i], chunks[i]) for i in idx]

# ---------------------------
# fallback synthesizer (if GenAI generate fails)
# ---------------------------
def synthesize_reply_from_chunks(tone: str, question: str, top_chunks: List[str]):
    header = f"{tone}\n\n"
    context = "\n\n".join([f"CONTEXT:\n{c}" for c in top_chunks])
    reply = header + "Using the context below, answer the question concisely.\n\n" + context + "\n\nQuestion: " + question + "\n\nAnswer:\n"
    return reply.strip()

# ---------------------------
# FastAPI app & endpoints
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard")
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/settings")
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/chat")
async def chat_ui_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/history")
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})

# Public settings endpoint for the widget (no auth)
@app.get("/settings_public")
def settings_public():
    s = get_setting("admin") or DEFAULT_ADMIN_SETTINGS
    return {
        "welcome": s.get("welcome", DEFAULT_ADMIN_SETTINGS["welcome"]),
        "fallback": s.get("fallback", DEFAULT_ADMIN_SETTINGS["fallback"]),
        "tone": s.get("tone", DEFAULT_ADMIN_SETTINGS["tone"])
    }

# Conversations list endpoint
@app.get("/conversations")
def conversations():
    return {"conversations": list_conversations()}

class ChatRequest(BaseModel):
    message: str
    conversation_id: str

def require_token(authorization: Optional[str]):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization")
    token = authorization.split(" ", 1)[1].strip()
    sess = get_session(token)
    if not sess:
        raise HTTPException(status_code=401, detail="Invalid session token")
    return sess["username"]

@app.post("/admin/login")
async def admin_login(data: Dict[str, str]):
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Missing username or password")
    user = get_user(username)
    if not user or not verify_password(user["password_hash"], password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_urlsafe(32)
    create_session(token, username)
    return {"token": token, "username": username}

@app.get("/admin/settings")
async def admin_get_settings(authorization: Optional[str] = Header(None)):
    require_token(authorization)
    s = get_setting("admin")
    if s is None:
        set_setting("admin", DEFAULT_ADMIN_SETTINGS)
        s = DEFAULT_ADMIN_SETTINGS
    return s

@app.post("/admin/settings")
async def admin_update_settings(payload: Dict[str, str], authorization: Optional[str] = Header(None)):
    require_token(authorization)
    s = get_setting("admin") or DEFAULT_ADMIN_SETTINGS
    # allow updating welcome/fallback/tone/user_prompt
    for k in ["welcome", "fallback", "tone", "user_prompt"]:
        if k in payload:
            s[k] = payload[k]
    set_setting("admin", s)
    return {"ok": True, "settings": s}

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    message = payload.message
    conv_id = payload.conversation_id

    # Save user message
    append_message(conv_id, "user", message)

    low = message.lower()
    if "refund" in low or "money back" in low or "return my money" in low or "moneyback" in low:
        append_message(conv_id, "system", "needs_human", {"reason": "refund_keyword"})
        return {"needs_human": True}

    # RAG retrieval
    top = top_k_chunks(message, k=3)
    scores = [s for (_, s, _) in top]
    best_score = max(scores) if scores else 0.0

    settings = get_setting("admin") or DEFAULT_ADMIN_SETTINGS
    fallback = settings.get("fallback", DEFAULT_ADMIN_SETTINGS["fallback"])
    tone = settings.get("tone", DEFAULT_ADMIN_SETTINGS["tone"])
    user_prompt_template = settings.get("user_prompt") or DEFAULT_ADMIN_SETTINGS["user_prompt"]

    if best_score < SIMILARITY_THRESHOLD:
        append_message(conv_id, "assistant", fallback, {"reason": "low_similarity", "scores": scores})
        return {"needs_human": False, "reply": fallback}

    top_texts = [c for (_, _, c) in top]
    context_text = "\n\n".join([f"CHUNK_{i+1}:\n{top_texts[i]}" for i in range(len(top_texts))])

    # Compose user prompt using admin-provided template
    try:
        user_prompt = user_prompt_template.format(context=context_text, question=message)
    except Exception:
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {message}\n\nAnswer:"

    system_prompt = tone

    # Generate via GenAI; fallback to synthesizer if fails
    try:
        reply = genai_generate(system_prompt, user_prompt, model=GENAI_CHAT_MODEL)
    except Exception as e:
        append_message(conv_id, "system", "genai_error", {"error": str(e)})
        reply = synthesize_reply_from_chunks(tone, message, top_texts)

    append_message(conv_id, "assistant", reply, {"scores": scores, "sources": [i for (i, _, _) in top]})
    return {"needs_human": False, "reply": reply}

@app.get("/conversation/{conv_id}")
async def get_conv(conv_id: str):
    conv = get_conversation(conv_id)
    return {"conversation_id": conv_id, "messages": conv}

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

# run locally for dev
if __name__ == "__main__":
    import uvicorn
    print("Starting server (Google GenAI backend) on :8000; GENAI_CHAT_MODEL:", GENAI_CHAT_MODEL, " EMBED_MODEL:", GENAI_EMBED_MODEL)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    