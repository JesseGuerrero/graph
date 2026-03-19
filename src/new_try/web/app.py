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

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

runs: dict = {}
runs_lock = threading.Lock()


class RunRequest(BaseModel):
    topic: str


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
    t = threading.Thread(target=run_pipeline, args=(run_id, req.topic, runs), daemon=True)
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


@app.get("/api/articles")
def list_articles():
    articles = []
    if not os.path.exists(OUTPUT_DIR):
        return articles
    for name in sorted(os.listdir(OUTPUT_DIR), reverse=True):
        dirpath = os.path.join(OUTPUT_DIR, name)
        if not os.path.isdir(dirpath):
            continue
        # Check for generated article
        has_article = any(f.startswith("storm_gen_article") and f.endswith(".txt") for f in os.listdir(dirpath))
        if has_article:
            stat = os.stat(dirpath)
            articles.append({
                "id": name,
                "topic": name.replace("_", " "),
                "created_at": stat.st_mtime,
                "type": "article",
            })
    return articles


@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    dirpath = os.path.join(OUTPUT_DIR, article_id)
    if not os.path.isdir(dirpath):
        raise HTTPException(404, "Article not found")

    result = {"topic": article_id.replace("_", " "), "article_html": "", "outline": "", "references": {}, "conversation_log": []}

    # Read polished article first, fall back to draft
    for fname in ["storm_gen_article_polished.txt", "storm_gen_article.txt"]:
        fpath = os.path.join(dirpath, fname)
        if os.path.exists(fpath):
            text = Path(fpath).read_text(encoding="utf-8")
            result["article_html"] = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
            break

    # Outline
    outline_path = os.path.join(dirpath, "storm_gen_outline.txt")
    if os.path.exists(outline_path):
        result["outline"] = Path(outline_path).read_text(encoding="utf-8")

    # References
    url_info_path = os.path.join(dirpath, "url_to_info.json")
    if os.path.exists(url_info_path):
        result["references"] = json.loads(Path(url_info_path).read_text(encoding="utf-8"))

    # Conversation log
    conv_path = os.path.join(dirpath, "conversation_log.json")
    if os.path.exists(conv_path):
        result["conversation_log"] = json.loads(Path(conv_path).read_text(encoding="utf-8"))

    return result


@app.delete("/api/articles/{article_id}")
def delete_article(article_id: str):
    dirpath = os.path.join(OUTPUT_DIR, article_id)
    if not os.path.isdir(dirpath):
        raise HTTPException(404, "Article not found")
    shutil.rmtree(dirpath)
    return {"ok": True}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
