import os
import queue
import threading
import time
from dotenv import load_dotenv
from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import DuckDuckGoSearchRM
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, q: queue.Queue):
        self.q = q
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
        queries = len(dlg_turn.search_queries) if dlg_turn.search_queries else 0
        sources = len(dlg_turn.search_results) if dlg_turn.search_results else 0
        self.total_queries += queries
        self.total_sources += sources
        self._put("dialogue_turn", {
            "queries": queries, "urls": sources,
            "total_queries": self.total_queries, "total_sources": self.total_sources,
        })

    def on_information_gathering_end(self, **kwargs):
        self._put("gathering_end", {
            "total_queries": self.total_queries, "total_sources": self.total_sources,
        })

    def on_information_organization_start(self, **kwargs):
        self._put("organization_start")

    def on_direct_outline_generation_end(self, outline: str, **kwargs):
        self._put("outline_draft", {"outline": outline})

    def on_outline_refinement_end(self, outline: str, **kwargs):
        self._put("outline_refined", {"outline": outline})
        self._put("writing_start")


def create_runner(output_dir: str) -> STORMWikiRunner:
    model_name = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    api_key = os.getenv("LLM_API_KEY", "")
    api_base = os.getenv("LLM_API_BASE", "")

    lm = LitellmModel(model=f"openai/{model_name}", api_key=api_key, api_base=api_base)

    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(lm)
    lm_configs.set_question_asker_lm(lm)
    lm_configs.set_outline_gen_lm(lm)
    lm_configs.set_article_gen_lm(lm)
    lm_configs.set_article_polish_lm(lm)

    rm = DuckDuckGoSearchRM(k=3)
    args = STORMWikiRunnerArguments(output_dir=output_dir, max_perspective=5, max_thread_num=3)

    return STORMWikiRunner(args=args, lm_configs=lm_configs, rm=rm)


def run_pipeline(run_id: str, topic: str, runs_dict: dict):
    run = runs_dict[run_id]
    q = run["queue"]
    try:
        runner = create_runner(OUTPUT_DIR)
        callback = StreamingCallbackHandler(q)
        runner.run(topic=topic, callback_handler=callback)
        q.put({"type": "polishing_start", "data": {}, "time": time.time()})
        # polishing is part of runner.run, so after it returns we're done
        q.put({"type": "done", "data": {"article_dir": runner.article_dir_name}, "time": time.time()})
        run["status"] = "done"
        run["article_dir"] = runner.article_dir_name
    except Exception as e:
        q.put({"type": "error", "data": {"message": str(e)}, "time": time.time()})
        run["status"] = "error"
