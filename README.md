
# IPTV-Assistant (chatbot-demo)

A lightweight FastAPI + Google GenAI (Gemini) demo that builds a small knowledge base from a local article, creates embeddings, and offers a simple chat UI backed by a local SQLite store.

This project is intended as a small demo / starting point for building a private chatbot that: 1) chunks and embeds a source article, 2) finds relevant chunks for a user query, and 3) uses Google GenAI to synthesize replies (with a fallback synthesizer when generation fails).

## Features

- FastAPI server with Jinja2 templates and static assets (UI pages under `templates/` and `static/`).
- Uses `google-genai` client to request embeddings and generation (Gemini models by default).
- Simple SQLite DB (`app.db`) with helpers in `db_schema.py` for settings, users, sessions and conversation storage.
- Builds an embedding index at startup from `data/article.txt` and performs top-k chunk retrieval for queries.

## Quickstart

Requirements

- Python 3.10+ (use a virtualenv)
- The pinned Python packages are in `requirements.txt`.

Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy or create a `.env` file in the project root. Example variables used by the app (see the provided `.env` for sample values):

- `GEMINI_API_KEY` — your Gemini/GenAI API key (required to call GenAI).
- `GOOGLE_GENAI_USE_VERTEXAI` — set to `true` to use Vertex AI (depends on your environment).
- `GENAI_CHAT_MODEL` — chat model to use (default: `gemini-2.5-flash`).
- `GENAI_EMBED_MODEL` — embedding model (default: `gemini-embedding-001`).
- `SIM_THRESHOLD` — similarity threshold used for retrieval (default: `0.68`).

Important: Do not commit API keys to version control.

4. Ensure `data/article.txt` exists (the app builds its in-memory KB from it on startup). If absent, add your article or content there.

Run the app (development)

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open the UI in your browser at `http://127.0.0.1:8000/login` (or `/dashboard`, `/chat`).

## Files & Structure

- `app.py` — main FastAPI application. Handles embedding generation, indexing, chat endpoints, authentication helpers and templates.
- `db_schema.py` — small SQLite wrapper: init_db, settings, user/session management, conversation storage and retrieval.
- `data/article.txt` — source article used to create chunks and embeddings for retrieval.
- `static/` — frontend JS/CSS: `admin.js`, `chat-widget.js`, `style.css`.
- `templates/` — Jinja2 templates used by the app: `login.html`, `dashboard.html`, `chat.html`, `history.html`, `settings.html`.
- `requirements.txt` — pinned dependencies used by the app.
- `install_pkg.sh` — helper script to install a package and update `requirements.txt`.

## Configuration

Most runtime configuration is via environment variables. The app loads `.env` automatically if present (via `python-dotenv`). Key settings include the GenAI API key, model selection and similarity threshold. You can override model names with `GENAI_CHAT_MODEL` and `GENAI_EMBED_MODEL`.

To reset the local data (sessions / conversations), remove or rename `app.db` in the project root and restart the app; `init_db()` will recreate the schema.

## Notes on behavior

- On startup, the server reads `data/article.txt`, chunks it into manageable text pieces, and requests embeddings from the chosen GenAI embedding model. That means the first startup may take additional time while embeddings are generated and cached in memory.
- If embedding or generation calls fail (network, keys), the app logs the exception and continues; a fallback synthesizer may produce simpler replies from retrieved chunks.

## Troubleshooting

- "No response" or generation failures: verify `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set and valid, and check network access.
- "Missing article" error: ensure `data/article.txt` is present and readable by the app process.
- DB issues: make sure the process has write permission to the project directory so `app.db` can be created.

## Extending the project

- Add more documents by extending the `data/` folder and updating embedding logic if you want multiple-document indices.
- Replace the simple cosine index with a persistent vector DB (e.g., FAISS, Milvus) for larger datasets.
- Add unit tests for `db_schema.py` and the embedding/query helpers.

## Install helper

Use `install_pkg.sh` to install new Python packages and record exact versions in `requirements.txt`:

```bash
./install_pkg.sh <package_name[extras]> [version]
```

## Security

- Keep API keys and secrets out of source control. Use environment variables or a secrets manager in production.

## License

No license file is included in this repository. Add a `LICENSE` file if you wish to publish this project under an open-source license (for example, MIT).

## Where to look next

- Edit `app.py` to change prompts, fallback behavior, or the chunking logic.
- Look at `db_schema.py` to extend user/session storage.

Happy hacking — if you want, I can also add a small `README` snippet showing how to deploy to a cloud provider or a Dockerfile for containerized runs.
