import json
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import markdown
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from storm_runner import run_pipeline, OUTPUT_DIR

load_dotenv()

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "storm", "frontend", "demo_light", ".user_settings.json")
STREAMLIT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "storm", "frontend", "demo_light", "DEMO_WORKING_DIR")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

runs: dict = {}
runs_lock = threading.Lock()


class RunRequest(BaseModel):
    topic: str
    settings: dict = {}


@app.get("/api/settings")
def get_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@app.post("/api/settings")
def save_settings(settings: dict):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    return {"ok": True}


@app.get("/")
def root():
    return RedirectResponse("/static/index.html")


@app.post("/api/run")
def start_run(req: RunRequest):
    run_id = str(uuid.uuid4())[:8]
    with runs_lock:
        runs[run_id] = {
            "topic": req.topic,
            "status": "running",
            "created_at": time.time(),
            "queue": queue.Queue(),
            "article_dir": None,
        }
    t = threading.Thread(target=run_pipeline, args=(run_id, req.topic, runs, req.settings), daemon=True)
    t.start()
    return {"run_id": run_id, "topic": req.topic}


@app.get("/api/run/{run_id}/events")
def run_events(run_id: str):
    with runs_lock:
        run = runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    def stream():
        q = run["queue"]
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/run/{run_id}/status")
def run_status(run_id: str):
    with runs_lock:
        run = runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return {"status": run["status"], "topic": run["topic"], "created_at": run["created_at"]}


def _scan_articles_dir(root_dir, source="web"):
    """Scan a directory for STORM article folders."""
    results = []
    if not os.path.exists(root_dir):
        return results
    for name in os.listdir(root_dir):
        dirpath = os.path.join(root_dir, name)
        if not os.path.isdir(dirpath):
            continue
        has_article = any(f.startswith("storm_gen_article") and f.endswith(".txt") for f in os.listdir(dirpath))
        if has_article:
            stat = os.stat(dirpath)
            results.append({
                "id": name,
                "topic": name.replace("_", " "),
                "created_at": stat.st_mtime,
                "type": "article",
                "source": source,
            })
    return results


@app.get("/api/articles")
def list_articles():
    articles = _scan_articles_dir(OUTPUT_DIR, "web")
    # Also include articles from Streamlit's output dir
    seen_ids = {a["id"] for a in articles}
    for a in _scan_articles_dir(STREAMLIT_OUTPUT_DIR, "streamlit"):
        if a["id"] not in seen_ids:
            articles.append(a)
    articles.sort(key=lambda a: a["created_at"], reverse=True)
    return articles


def _find_article_dir(article_id: str):
    """Find article directory, checking web output first then Streamlit."""
    for root in [OUTPUT_DIR, STREAMLIT_OUTPUT_DIR]:
        dirpath = os.path.join(root, article_id)
        if os.path.isdir(dirpath):
            return dirpath
    return None


@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")

    result = {"topic": article_id.replace("_", " "), "article_text": "", "article_html": "",
              "outline": "", "citation_dict": {}, "conversation_log": []}

    # Read polished article first, fall back to draft
    for fname in ["storm_gen_article_polished.txt", "storm_gen_article.txt"]:
        fpath = os.path.join(dirpath, fname)
        if os.path.exists(fpath):
            text = Path(fpath).read_text(encoding="utf-8", errors="replace")
            result["article_text"] = text
            result["article_html"] = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
            break

    # Outline
    outline_path = os.path.join(dirpath, "storm_gen_outline.txt")
    if os.path.exists(outline_path):
        result["outline"] = Path(outline_path).read_text(encoding="utf-8", errors="replace")

    # References - build citation_dict like Streamlit does
    url_info_path = os.path.join(dirpath, "url_to_info.json")
    if os.path.exists(url_info_path):
        raw = json.loads(Path(url_info_path).read_text(encoding="utf-8", errors="replace"))
        citation_dict = {}
        if "url_to_unified_index" in raw and "url_to_info" in raw:
            for url, index in raw["url_to_unified_index"].items():
                info = raw["url_to_info"].get(url, {})
                citation_dict[str(index)] = {
                    "url": url,
                    "title": info.get("title", ""),
                    "snippets": info.get("snippets", []),
                }
        result["citation_dict"] = citation_dict

    # Conversation log
    conv_path = os.path.join(dirpath, "conversation_log.json")
    if os.path.exists(conv_path):
        result["conversation_log"] = json.loads(Path(conv_path).read_text(encoding="utf-8", errors="replace"))

    return result


@app.delete("/api/articles/{article_id}")
def delete_article(article_id: str):
    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")
    shutil.rmtree(dirpath)
    return {"ok": True}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
