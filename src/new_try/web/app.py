import json
import os
import queue
import re
import shutil
import sys
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

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), ".user_settings.json")

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


def _dir_name_to_topic(name):
    """Strip trailing _<timestamp> suffix from directory name to get topic."""
    cleaned = re.sub(r'_\d{10}$', '', name)
    return cleaned.replace("_", " ")


def _scan_articles_dir(root_dir):
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
                "topic": _dir_name_to_topic(name),
                "created_at": stat.st_mtime,
                "type": "article",
            })
    return results


@app.post("/api/articles/import")
def import_article(req: dict):
    """Save pasted markdown as a new article."""
    title = (req.get("title") or "Untitled").strip()
    markdown_text = (req.get("markdown") or "").strip()
    if not markdown_text:
        raise HTTPException(400, "Markdown text is required")

    slug = re.sub(r'[^a-zA-Z0-9]+', '_', title).strip('_')
    ts = str(int(time.time()))
    dir_name = f"{slug}_{ts}"
    dirpath = os.path.join(OUTPUT_DIR, dir_name)
    os.makedirs(dirpath, exist_ok=True)

    Path(os.path.join(dirpath, "storm_gen_article_polished.txt")).write_text(
        markdown_text, encoding="utf-8"
    )

    return {"id": dir_name, "topic": title}


@app.get("/api/articles")
def list_articles():
    articles = _scan_articles_dir(OUTPUT_DIR)
    articles.sort(key=lambda a: a["created_at"], reverse=True)
    return articles


def _find_article_dir(article_id: str):
    dirpath = os.path.join(OUTPUT_DIR, article_id)
    return dirpath if os.path.isdir(dirpath) else None


@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")

    result = {"topic": _dir_name_to_topic(article_id), "article_text": "", "article_html": "",
              "outline": "", "citation_dict": {}, "conversation_log": []}

    # Read polished article first, fall back to draft
    for fname in ["storm_gen_article_polished.txt", "storm_gen_article.txt"]:
        fpath = os.path.join(dirpath, fname)
        if os.path.exists(fpath):
            text = Path(fpath).read_text(encoding="utf-8", errors="replace")
            # Strip replacement characters from any encoding issues
            text = text.replace('\ufffd', '\u2014')  # replace with em-dash (most common source)
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
                    "title": info.get("title", "").replace('\ufffd', '\u2014'),
                    "snippets": [s.replace('\ufffd', '\u2014') for s in info.get("snippets", [])],
                }
        result["citation_dict"] = citation_dict

    # Conversation log
    conv_path = os.path.join(dirpath, "conversation_log.json")
    if os.path.exists(conv_path):
        result["conversation_log"] = json.loads(Path(conv_path).read_text(encoding="utf-8", errors="replace"))

    return result


@app.get("/api/articles/{article_id}/narrative")
def get_narrative(article_id: str):
    from narrative import build_narrative_data, generate_storyline_div
    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")
    # Read article text
    for fname in ["storm_gen_article_polished.txt", "storm_gen_article.txt"]:
        fpath = os.path.join(dirpath, fname)
        if os.path.exists(fpath):
            text = Path(fpath).read_text(encoding="utf-8", errors="replace")
            text = text.replace('\ufffd', '\u2014')
            topic = _dir_name_to_topic(article_id)
            data = build_narrative_data(text, topic)
            html = generate_storyline_div(data)
            return {"html": html}
    raise HTTPException(404, "No article text found")


@app.get("/api/articles/{article_id}/images/{filename}")
def get_article_image(article_id: str, filename: str):
    from fastapi.responses import FileResponse
    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")
    # Sanitize filename
    if '/' in filename or '\\' in filename or '..' in filename:
        raise HTTPException(400, "Invalid filename")
    img_path = os.path.join(dirpath, ".image_cache", filename)
    if not os.path.exists(img_path):
        raise HTTPException(404, "Image not found")
    ext = filename.rsplit('.', 1)[-1].lower()
    media_types = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
                   'gif': 'image/gif', 'webp': 'image/webp'}
    return FileResponse(img_path, media_type=media_types.get(ext, 'image/jpeg'))


@app.get("/api/articles/{article_id}/kg")
def get_kg(article_id: str):
    """Return cached KG taxonomy JSON, or 404 if not built yet."""
    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")
    kg_path = os.path.join(dirpath, "kg_taxonomy.json")
    if not os.path.exists(kg_path):
        raise HTTPException(404, "Knowledge graph not built yet")
    return json.loads(Path(kg_path).read_text(encoding="utf-8"))


@app.post("/api/articles/{article_id}/kg/build")
def build_kg(article_id: str):
    """Build KG taxonomy via LLM. Streams SSE progress events."""
    import asyncio
    from storm_runner import create_runner
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.taxonomy import KGTaxonomyBuilder, create_llm_fn_openai

    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")

    # Read article text
    article_text = ""
    for fname in ["storm_gen_article_polished.txt", "storm_gen_article.txt"]:
        fpath = os.path.join(dirpath, fname)
        if os.path.exists(fpath):
            article_text = Path(fpath).read_text(encoding="utf-8", errors="replace")
            article_text = article_text.replace('\ufffd', '\u2014')
            break
    if not article_text:
        raise HTTPException(404, "No article text found")

    # Get LLM settings
    settings = {}
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    api_key = settings.get("cfg_llm_key") or os.getenv("LLM_API_KEY", "")
    api_base = settings.get("cfg_llm_url") or os.getenv("LLM_API_BASE", "")
    model = settings.get("cfg_llm_model") or os.getenv("LLM_MODEL", "claude-opus-4-6")

    if not api_key or not api_base:
        raise HTTPException(400, "LLM API key and URL must be configured in settings")

    q = queue.Queue()
    topic = _dir_name_to_topic(article_id)

    def run_build():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            llm = create_llm_fn_openai(base_url=api_base, api_key=api_key, model=model)

            def on_node(node):
                q.put({"type": "node", "data": {"label": node.label, "depth": node.depth, "node_type": node.node_type.value}})

            builder = KGTaxonomyBuilder(
                llm_fn=llm, markdown=article_text, title=topic,
                max_depth=6, on_node=on_node,
            )
            kg = loop.run_until_complete(builder.build())
            tree = kg.to_tree_dict()

            # Cache to disk
            kg_path = os.path.join(dirpath, "kg_taxonomy.json")
            with open(kg_path, "w", encoding="utf-8") as f:
                json.dump(tree, f, indent=2)

            q.put({"type": "done", "data": tree})
        except Exception as e:
            q.put({"type": "error", "data": {"message": str(e)}})
        finally:
            loop.close()

    t = threading.Thread(target=run_build, daemon=True)
    t.start()

    def stream():
        while True:
            try:
                event = q.get(timeout=60)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.put("/api/articles/{article_id}")
def update_article(article_id: str, req: dict):
    """Update an existing article's markdown text."""
    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")
    markdown_text = (req.get("markdown") or "").strip()
    if not markdown_text:
        raise HTTPException(400, "Markdown text is required")
    Path(os.path.join(dirpath, "storm_gen_article_polished.txt")).write_text(
        markdown_text, encoding="utf-8"
    )
    return {"ok": True}


@app.delete("/api/articles/{article_id}")
def delete_article(article_id: str):
    dirpath = _find_article_dir(article_id)
    if not dirpath:
        raise HTTPException(404, "Article not found")
    shutil.rmtree(dirpath)
    return {"ok": True}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
