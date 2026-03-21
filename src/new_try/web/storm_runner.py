import os
import sys
import queue
import threading
import time
import traceback
from dotenv import load_dotenv

# Use local storm directory instead of any installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'storm'))

from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import SearXNG, DuckDuckGoSearchRM, CachedSerperRM
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, q: queue.Queue, runner=None):
        self.q = q
        self.runner = runner
        self.total_queries = 0
        self.total_sources = 0

    def _put(self, event_type, data=None):
        self.q.put({"type": event_type, "data": data or {}, "time": time.time()})

    def on_identify_perspective_start(self, **kwargs):
        self._put("perspective_start")

    def on_identify_perspective_end(self, perspectives: list[str], **kwargs):
        self._put("perspective_end", {"perspectives": perspectives})

    def on_information_gathering_start(self, **kwargs):
        self._put("gathering_start")

    def on_dialogue_turn_end(self, dlg_turn, **kwargs):
        urls = list(set([r.url for r in dlg_turn.search_results])) if dlg_turn.search_results else []
        queries = len(dlg_turn.search_queries) if dlg_turn.search_queries else 0
        self.total_queries += queries
        self.total_sources += len(urls)
        self._put("dialogue_turn", {
            "queries": queries, "urls": len(urls),
            "total_queries": self.total_queries, "total_sources": self.total_sources,
            "browsed_urls": urls,
        })

    def on_information_gathering_end(self, **kwargs):
        # Report failed URL extractions like Streamlit does
        failed_urls = {}
        if self.runner and hasattr(self.runner, 'rm') and hasattr(self.runner.rm, 'failed_urls'):
            failed_urls = self.runner.rm.failed_urls
        self._put("gathering_end", {
            "total_queries": self.total_queries, "total_sources": self.total_sources,
            "failed_urls": {url: reason for url, reason in failed_urls.items()} if failed_urls else {},
        })

    def on_information_organization_start(self, **kwargs):
        self._put("organization_start")

    def on_direct_outline_generation_end(self, outline: str, **kwargs):
        self._put("outline_draft", {"outline": outline})

    def on_outline_refinement_end(self, outline: str, **kwargs):
        self._put("outline_refined", {"outline": outline})
        self._put("writing_start")


def create_runner(output_dir: str, settings: dict = None) -> STORMWikiRunner:
    settings = settings or {}
    model_name = settings.get("openai_model") or os.getenv("LLM_MODEL", "claude-opus-4-6")
    api_key = settings.get("openai_key") or os.getenv("LLM_API_KEY", "")
    api_base = settings.get("openai_url") or os.getenv("LLM_API_BASE", "")
    searxng_url = settings.get("searxng_url") or os.getenv("SEARXNG_URL", "")

    lm_kwargs = {"api_key": api_key, "api_base": api_base}

    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=1000, temperature=1.0, top_p=0.9, **lm_kwargs))
    lm_configs.set_question_asker_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=1000, temperature=1.0, top_p=0.9, **lm_kwargs))
    outline_tokens = int(settings.get("outline_gen_tokens", 1000))
    article_tokens = int(settings.get("article_gen_tokens", 4000))
    polish_tokens = int(settings.get("article_polish_tokens", 8000))
    lm_configs.set_outline_gen_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=outline_tokens, temperature=1.0, top_p=0.9, **lm_kwargs))
    lm_configs.set_article_gen_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=article_tokens, temperature=1.0, top_p=0.9, **lm_kwargs))
    lm_configs.set_article_polish_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=polish_tokens, temperature=1.0, top_p=0.9, **lm_kwargs))

    max_perspective = int(settings.get("max_perspective", 5))
    max_conv_turn = int(settings.get("max_conv_turn", 5))
    search_top_k = int(settings.get("search_top_k", 5))
    retrieve_top_k = int(settings.get("retrieve_top_k", 10))

    search_provider = settings.get("search_provider", "searxng")
    chunk_size = int(settings.get("chunk_size", 1000))
    serper_key = settings.get("serper_key", "")
    serper_cache = settings.get("serper_cache", True)

    if search_provider == "serper" and serper_key:
        rm = CachedSerperRM(serper_search_api_key=serper_key, k=search_top_k,
                           cache_enabled=serper_cache, snippet_chunk_size=chunk_size)
    elif search_provider == "searxng" and searxng_url:
        rm = SearXNG(searxng_api_url=searxng_url, k=search_top_k, snippet_chunk_size=chunk_size)
    else:
        rm = DuckDuckGoSearchRM(k=search_top_k, snippet_chunk_size=chunk_size)

    args = STORMWikiRunnerArguments(output_dir=output_dir, max_perspective=max_perspective,
                                   max_conv_turn=max_conv_turn, search_top_k=search_top_k,
                                   retrieve_top_k=retrieve_top_k, max_thread_num=3)
    return STORMWikiRunner(args=args, lm_configs=lm_configs, rm=rm)


def run_pipeline(run_id: str, topic: str, runs_dict: dict, settings: dict = None):
    run = runs_dict[run_id]
    q = run["queue"]
    try:
        runner = create_runner(OUTPUT_DIR, settings)
        callback = StreamingCallbackHandler(q, runner=runner)
        runner.run(topic=topic, callback_handler=callback)
        q.put({"type": "polishing_start", "data": {}, "time": time.time()})
        q.put({"type": "done", "data": {"article_dir": runner.article_dir_name}, "time": time.time()})
        run["status"] = "done"
        run["article_dir"] = runner.article_dir_name
    except Exception as e:
        traceback.print_exc()
        q.put({"type": "error", "data": {"message": str(e)}, "time": time.time()})
        run["status"] = "error"
