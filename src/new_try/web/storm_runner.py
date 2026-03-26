import os
import sys
import queue
import threading
import time
import traceback
from dotenv import load_dotenv

# Use local storm directory instead of any installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import SearXNG, DuckDuckGoSearchRM, CachedSerperRM, XaiTwitterRM
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, q: queue.Queue, runner=None, is_serper=False):
        self.q = q
        self.runner = runner
        self.is_serper = is_serper
        self.total_queries = 0
        self.total_sources = 0
        self.total_cached = 0
        self.total_new = 0

    def _get_rm(self):
        """Get the retrieval module from the runner (runner.retriever.rm)."""
        if self.runner and hasattr(self.runner, 'retriever') and hasattr(self.runner.retriever, 'rm'):
            return self.runner.retriever.rm
        return None

    def _put(self, event_type, data=None):
        self.q.put({"type": event_type, "data": data or {}, "time": time.time()})

    def on_identify_perspective_start(self, **kwargs):
        self._put("perspective_start")

    def on_identify_perspective_end(self, perspectives: list[str], **kwargs):
        self._put("perspective_end", {"perspectives": perspectives})

    def on_information_gathering_start(self, **kwargs):
        self._put("gathering_start", {"is_serper": self.is_serper})

    def on_dialogue_turn_end(self, dlg_turn, **kwargs):
        urls = list(set([r.url for r in dlg_turn.search_results])) if dlg_turn.search_results else []
        search_queries = dlg_turn.search_queries or []
        self.total_queries += len(search_queries)
        self.total_sources += len(urls)

        # Build per-query data with cache status for Serper
        query_details = []
        rm = self._get_rm()
        if self.is_serper and rm:
            cache_status = getattr(rm, 'last_query_cache_status', {})
            for q in search_queries:
                was_cached = cache_status.get(q, False)
                if was_cached:
                    self.total_cached += 1
                else:
                    self.total_new += 1
                # Find URLs that came from this query's results
                query_urls = []
                for r in (dlg_turn.search_results or []):
                    if r.url in urls:
                        query_urls.append(r.url)
                query_details.append({
                    "query": q,
                    "cached": was_cached,
                    "urls": list(set(query_urls)),
                })
        else:
            for q in search_queries:
                query_urls = [r.url for r in (dlg_turn.search_results or [])]
                query_details.append({
                    "query": q,
                    "urls": list(set(query_urls)),
                })

        self._put("dialogue_turn", {
            "queries": len(search_queries), "urls": len(urls),
            "total_queries": self.total_queries, "total_sources": self.total_sources,
            "total_cached": self.total_cached, "total_new": self.total_new,
            "query_details": query_details,
            "is_serper": self.is_serper,
        })

    def on_information_gathering_end(self, **kwargs):
        # Report failed URL extractions like Streamlit does
        rm = self._get_rm()
        failed_urls = {}
        if rm and hasattr(rm, 'failed_urls'):
            failed_urls = rm.failed_urls
        end_data = {
            "total_queries": self.total_queries, "total_sources": self.total_sources,
            "total_cached": self.total_cached, "total_new": self.total_new,
            "is_serper": self.is_serper,
            "failed_urls": {url: reason for url, reason in failed_urls.items()} if failed_urls else {},
        }
        if self.total_sources < 25:
            self._put("gathering_end", end_data)
            self._put("error", {"message": f"There are not enough sources ({self.total_sources} found, 25 required)"})
            raise RuntimeError(f"Not enough sources: {self.total_sources} < 25")
        self._put("gathering_end", end_data)

    def on_information_organization_start(self, **kwargs):
        self._put("organization_start")

    def on_direct_outline_generation_end(self, outline: str, **kwargs):
        self._put("outline_draft", {"outline": outline})

    def on_outline_refinement_end(self, outline: str, **kwargs):
        self._put("outline_refined", {"outline": outline})
        self._put("writing_start")


def create_runner(output_dir: str, settings: dict = None) -> tuple:
    """Returns (runner, is_serper, x_rm_or_none) tuple."""
    settings = settings or {}
    model_name = settings.get("openai_model") or os.getenv("LLM_MODEL", "claude-opus-4-6")
    api_key = settings.get("openai_key") or os.getenv("LLM_API_KEY", "")
    api_base = settings.get("openai_url") or os.getenv("LLM_API_BASE", "")
    searxng_url = settings.get("searxng_url") or os.getenv("SEARXNG_URL", "")

    lm_kwargs = {"api_key": api_key, "api_base": api_base}

    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=1000, temperature=0.3, top_p=0.9, **lm_kwargs))
    lm_configs.set_question_asker_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=1000, temperature=0.7, top_p=0.9, **lm_kwargs))
    outline_tokens = int(settings.get("outline_gen_tokens", 1000))
    article_tokens = int(settings.get("article_gen_tokens", 4000))
    polish_tokens = int(settings.get("article_polish_tokens", 8000))
    lm_configs.set_outline_gen_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=outline_tokens, temperature=0.5, top_p=0.9, **lm_kwargs))
    lm_configs.set_article_gen_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=article_tokens, temperature=0.7, top_p=0.9, **lm_kwargs))
    lm_configs.set_article_polish_lm(
        LitellmModel(model=f"openai/{model_name}", max_tokens=polish_tokens, temperature=0.3, top_p=0.9, **lm_kwargs))

    max_perspective = int(settings.get("max_perspective", 5))
    max_conv_turn = int(settings.get("max_conv_turn", 5))
    search_top_k = int(settings.get("search_top_k", 5))
    retrieve_top_k = int(settings.get("retrieve_top_k", 10))

    search_provider = settings.get("search_provider", "searxng")
    chunk_size = int(settings.get("chunk_size", 1000))
    serper_key = settings.get("serper_key", "")
    serper_cache = settings.get("serper_cache", True)

    is_serper = False
    if search_provider == "serper" and serper_key:
        rm = CachedSerperRM(serper_search_api_key=serper_key, k=search_top_k,
                           cache_enabled=serper_cache, snippet_chunk_size=chunk_size)
        is_serper = bool(serper_cache)  # only show cache/cost info when caching enabled
    elif search_provider == "searxng" and searxng_url:
        rm = SearXNG(searxng_api_url=searxng_url, k=search_top_k, snippet_chunk_size=chunk_size)
    else:
        rm = DuckDuckGoSearchRM(k=search_top_k, snippet_chunk_size=chunk_size)

    # xAI Twitter/X search
    xai_enabled = bool(settings.get("xai_enabled", False))
    xai_key = settings.get("xai_key", "")
    x_rm = None
    if xai_enabled and xai_key:
        x_rm = XaiTwitterRM(xai_api_key=xai_key, k=search_top_k)

    include_images = bool(settings.get("include_images", False))
    args = STORMWikiRunnerArguments(output_dir=output_dir, max_perspective=max_perspective,
                                   max_conv_turn=max_conv_turn, search_top_k=search_top_k,
                                   retrieve_top_k=retrieve_top_k, max_thread_num=3,
                                   include_images=include_images)
    return STORMWikiRunner(args=args, lm_configs=lm_configs, rm=rm, x_rm=x_rm), is_serper


def run_pipeline(run_id: str, topic: str, runs_dict: dict, settings: dict = None):
    run = runs_dict[run_id]
    q = run["queue"]
    try:
        runner, is_serper = create_runner(OUTPUT_DIR, settings)
        callback = StreamingCallbackHandler(q, runner=runner, is_serper=is_serper)
        runner.run(topic=topic, callback_handler=callback)
        q.put({"type": "polishing_start", "data": {}, "time": time.time()})
        q.put({"type": "done", "data": {"article_dir": runner.article_dir_name}, "time": time.time()})
        run["status"] = "done"
        run["article_dir"] = runner.article_dir_name
    except Exception as e:
        traceback.print_exc()
        q.put({"type": "error", "data": {"message": str(e)}, "time": time.time()})
        run["status"] = "error"
