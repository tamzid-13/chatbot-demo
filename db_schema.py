# db_schema.py
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

DB_PATH = Path(__file__).parent / "app.db"

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT,
        role TEXT,
        text TEXT,
        meta TEXT,
        ts REAL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        username TEXT,
        ts REAL
    )""")
    conn.commit()
    conn.close()

def get_setting(key: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return json.loads(row["value"])

def set_setting(key: str, value: Dict[str, Any]):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
    conn.commit()
    conn.close()

def append_message(conv_id: str, role: str, text: str, meta: Optional[Dict[str,Any]] = None, ts: float = None):
    import time
    conn = _get_conn()
    cur = conn.cursor()
    if ts is None:
        ts = time.time()
    cur.execute("INSERT INTO conversations(id, role, text, meta, ts) VALUES (?, ?, ?, ?, ?)",
                (conv_id, role, text, json.dumps(meta or {}), ts))
    conn.commit()
    conn.close()
    
def list_conversations() -> List[Dict[str, Any]]:
    """
    Returns a list of all conversation IDs with:
    - last_ts (float)
    - message_count (int)
    Ordered by last_ts DESC.
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            id,
            MAX(ts) AS last_ts,
            COUNT(*) AS message_count
        FROM conversations
        GROUP BY id
        ORDER BY last_ts DESC
    """)
    rows = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append({
            "conversation_id": r["id"],
            "last_ts": r["last_ts"],
            "message_count": r["message_count"]
        })
    return result

def get_conversation(conv_id: str) -> List[Dict[str,Any]]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, text, meta, ts FROM conversations WHERE id = ? ORDER BY ts ASC", (conv_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": r["role"], "text": r["text"], "meta": json.loads(r["meta"]), "ts": r["ts"]} for r in rows]

def create_user(username: str, password_hash: str):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO users(username, password_hash) VALUES (?, ?)", (username, password_hash))
    conn.commit()
    conn.close()

def get_user(username: str) -> Optional[Dict[str,str]]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row: return None
    return {"username": row["username"], "password_hash": row["password_hash"]}

def create_session(token: str, username: str, ts: float = None):
    import time
    conn = _get_conn()
    cur = conn.cursor()
    if ts is None:
        ts = time.time()
    cur.execute("INSERT OR REPLACE INTO sessions(token, username, ts) VALUES (?, ?, ?)", (token, username, ts))
    conn.commit()
    conn.close()

def get_session(token: str) -> Optional[Dict[str,Any]]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT token, username, ts FROM sessions WHERE token = ?", (token,))
    row = cur.fetchone()
    conn.close()
    if not row: return None
    return {"token": row["token"], "username": row["username"], "ts": row["ts"]}
