"""
Knowledge Graph Taxonomy Builder
=================================

Transforms long-form research documents into hierarchical knowledge graphs
with up to 10 levels of depth. Uses an LLM (Claude Opus 4.6) to:

1. Auto-discover knowledge categories from the document
2. Recursively decompose categories into subcategories and entities
3. Extract verifiable claims at leaf nodes
4. Link every node back to its source section and cited URLs
5. Verify leaf claims against web sources
6. Aggregate verification scores bottom-up through the tree

Designed for generalized deep research reports — works on any topic, any length.

Example tree for a geopolitics report:
    Root
    ├── Countries Involved (category, depth 1)
    │   ├── United States (entity, depth 2)
    │   │   ├── Federal Policy (category, depth 3)
    │   │   │   ├── Executive Orders (category, depth 4)
    │   │   │   │   └── "EO 14110 signed Oct 30 2023" (claim, depth 5) ✓
    │   │   │   └── Congressional Action (category, depth 4)
    │   │   │       └── "CHIPS Act allocated $52.7B" (claim, depth 5) ✓
    │   │   └── State Regulation (category, depth 3)
    │   │       ├── "CA SB 1047 vetoed Sep 2024" (claim, depth 4) ✓
    │   │       └── "TX HB 2060 AI council 2023" (claim, depth 4) ✗
    │   ├── European Union (entity, depth 2)
    │   │   └── "EU AI Act adopted Mar 2024" (claim, depth 3) ✓
    │   └── China (entity, depth 2)
    │       └── "Interim AI measures Aug 2023" (claim, depth 3) ✗
    ├── Key Technologies (category, depth 1)
    │   ├── Foundation Models → ...deeper...
    │   └── Semiconductor Supply Chain → ...deeper...
    └── Economic Impact (category, depth 1)
        └── ...

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                   Research Document                  │
    │          (markdown with ## sections, URLs)           │
    └──────────────────────┬──────────────────────────────┘
                           │
                    Phase 1: Discovery
                    LLM reads full doc, emits 3-8
                    top-level knowledge categories
                           │
                    Phase 2: Recursive Decomposition
                    For each category, LLM decides:
                      → decompose into subcategories
                      → OR emit verifiable leaf claims
                    Repeats until depth 10 or all leaves
                           │
                    Phase 2B: Provenance Enrichment
                    LLM links each non-leaf node to
                    source text snippets + cited URLs
                           │
                    Phase 3: Leaf Verification (optional)
                    Each leaf claim verified against
                    web sources (URL resolver + LLM judge)
                           │
                    Phase 4: Bottom-Up Aggregation
                    Leaf verdicts (pass/fail) roll up
                    as percentage scores to parents
                           │
                    ┌──────┴──────┐
                    │  Verified   │
                    │  Knowledge  │
                    │    Graph    │
                    └─────────────┘

Node types:
    ROOT     — tree root, aggregates everything
    CATEGORY — abstract grouping ("Economic Impact", "Legal Framework")
    ENTITY   — named thing ("United States", "GPT-4", "TSMC")
    CLAIM    — verifiable leaf ("TSMC holds 60% foundry share")

Every node carries:
    source_section  — which ## heading it came from
    source_snippet  — the text that generated it
    cited_urls      — URLs from the report near this content
    depth           — 0 (root) to 10 (max leaf)

Leaf nodes additionally carry:
    claim_text       — the specific verifiable assertion
    search_query     — Google query to find evidence
    preferred_domain — most authoritative site for this claim
    verdict          — passed / failed / unverified / skipped
    evidence_urls    — URLs actually checked
    errors           — error codes
    critical         — whether load-bearing for the report

Parent nodes carry:
    aggregated_score — average of children (0.0-1.0)
    child_pass/fail/total — counts
    verdict          — passed (100%) / failed (0%) / partial (mixed)

Usage:
    # Build
    llm = create_llm_fn_anthropic(api_key="sk-...", model="claude-opus-4-6-20250219")
    builder = KGTaxonomyBuilder(llm_fn=llm, markdown=report, max_depth=10)
    kg = await builder.build()

    # Verify (optional — plug in your own verification pipeline)
    kg = await verify_knowledge_graph(kg, your_verify_fn)

    # Export
    kg.print_tree()                     # ASCII tree to stdout
    tree = kg.to_tree_dict()            # nested JSON for frontend
    flat = kg.to_dict()                 # flat dict with all nodes

    # CLI
    python knowledge_graph_taxonomy.py report.md --api-key sk-... --max-depth 6 --tree
"""

import json
import re
import logging
import asyncio
import hashlib
from typing import Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("kg_taxonomy")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NodeType(str, Enum):
    ROOT = "root"
    CATEGORY = "category"
    ENTITY = "entity"
    CLAIM = "claim"


class Verdict(str, Enum):
    UNVERIFIED = "unverified"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


LLMFn = Callable[[str, str], Awaitable[str]]
VerifyFn = Callable[..., Awaitable[dict]]


@dataclass
class KGNode:
    """A single node in the knowledge taxonomy tree."""
    id: str
    label: str
    node_type: NodeType
    depth: int = 0
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    # Provenance
    source_section: str = ""
    source_snippet: str = ""
    cited_urls: list[str] = field(default_factory=list)

    # Leaf verification
    claim_text: Optional[str] = None
    search_query: Optional[str] = None
    preferred_domain: str = ""
    verdict: Verdict = Verdict.UNVERIFIED
    evidence_urls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    critical: bool = False

    # Aggregation
    aggregated_score: Optional[float] = None
    child_pass: int = 0
    child_fail: int = 0
    child_total: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type.value,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "source_section": self.source_section,
            "source_snippet": self.source_snippet[:300] if self.source_snippet else "",
            "cited_urls": self.cited_urls,
            "claim_text": self.claim_text,
            "search_query": self.search_query,
            "preferred_domain": self.preferred_domain,
            "verdict": self.verdict.value,
            "evidence_urls": self.evidence_urls,
            "errors": self.errors,
            "critical": self.critical,
            "aggregated_score": self.aggregated_score,
            "child_pass": self.child_pass,
            "child_fail": self.child_fail,
            "child_total": self.child_total,
            "is_leaf": self.is_leaf,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Knowledge Graph Container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KnowledgeGraphTaxonomy:
    """
    Flat node store with tree structure via parent_id / children_ids.
    Supports bottom-up score aggregation and multiple export formats.
    """

    def __init__(self):
        self.nodes: dict[str, KGNode] = {}
        self.title: str = ""
        self.research_question: str = ""
        self.max_depth_allowed: int = 10
        self.nodes["root"] = KGNode(
            id="root", label="Knowledge Graph", node_type=NodeType.ROOT, depth=0
        )

    def add_node(self, node: KGNode) -> KGNode:
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
        return node

    def get_children(self, node_id: str) -> list[KGNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_leaves(self) -> list[KGNode]:
        return [n for n in self.nodes.values() if n.is_leaf and n.node_type != NodeType.ROOT]

    def get_nodes_at_depth(self, depth: int) -> list[KGNode]:
        return [n for n in self.nodes.values() if n.depth == depth]

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

    def aggregate_scores(self):
        """Bottom-up: leaf verdicts → parent scores.

        Uses Mind2Web 2 gate-then-average formula (§3.3):
          s(v) = 0                          if any critical child fails
          s(v) = avg(non-critical scores)   if all critical pass & non-critical exist
          s(v) = 1                          if ALL children are critical and all pass
        """
        for depth in range(self.max_depth, -1, -1):
            for node in self.get_nodes_at_depth(depth):
                if node.is_leaf:
                    if node.verdict == Verdict.PASSED:
                        node.aggregated_score = 1.0
                    elif node.verdict == Verdict.FAILED:
                        node.aggregated_score = 0.0
                    else:
                        node.aggregated_score = None
                else:
                    children = self.get_children(node.id)
                    scored = [c for c in children if c.aggregated_score is not None]
                    node.child_total = len(children)
                    node.child_pass = sum(
                        1 for c in children
                        if c.verdict == Verdict.PASSED
                        or (c.aggregated_score is not None and c.aggregated_score == 1.0)
                    )
                    node.child_fail = sum(
                        1 for c in children
                        if c.verdict == Verdict.FAILED
                        or (c.aggregated_score is not None and c.aggregated_score == 0.0)
                    )

                    if not scored:
                        node.aggregated_score = None
                        node.verdict = Verdict.UNVERIFIED
                        continue

                    # Partition into critical (K) and non-critical (N)
                    critical = [c for c in scored if c.critical]
                    non_critical = [c for c in scored if not c.critical]

                    # Gate: any critical child fails → parent = 0
                    any_critical_failed = any(
                        c.aggregated_score is not None and c.aggregated_score < 1.0
                        for c in critical
                    )
                    if any_critical_failed:
                        node.aggregated_score = 0.0
                        node.verdict = Verdict.FAILED
                    elif non_critical:
                        # All critical pass → score = average of non-critical
                        node.aggregated_score = sum(c.aggregated_score for c in non_critical) / len(non_critical)
                        if node.aggregated_score == 1.0:
                            node.verdict = Verdict.PASSED
                        elif node.aggregated_score == 0.0:
                            node.verdict = Verdict.FAILED
                        else:
                            node.verdict = Verdict.PARTIAL
                    else:
                        # All children are critical — all must pass
                        all_pass = all(c.aggregated_score == 1.0 for c in critical)
                        node.aggregated_score = 1.0 if all_pass else 0.0
                        node.verdict = Verdict.PASSED if all_pass else Verdict.FAILED

    def to_dict(self) -> dict:
        self.aggregate_scores()
        root = self.nodes.get("root")
        return {
            "title": self.title,
            "research_question": self.research_question,
            "max_depth": self.max_depth,
            "total_nodes": len(self.nodes),
            "total_leaves": len(self.get_leaves()),
            "overall_score": root.aggregated_score if root else None,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    def to_tree_dict(self, node_id: str = "root") -> dict:
        node = self.nodes.get(node_id)
        if not node:
            return {}
        result = node.to_dict()
        result["children"] = [self.to_tree_dict(cid) for cid in node.children_ids]
        return result

    def print_tree(self, node_id: str = "root", indent: int = 0):
        node = self.nodes.get(node_id)
        if not node:
            return
        icons = {"root": "◆", "category": "◇", "entity": "●", "claim": "◎"}
        marks = {"passed": "✓", "failed": "✗", "partial": "~", "unverified": "?", "skipped": "-"}
        icon = icons.get(node.node_type.value, "·")
        mark = marks.get(node.verdict.value, "?")
        score = f" [{node.aggregated_score:.0%}]" if node.aggregated_score is not None else ""
        prefix = "    " * indent
        connector = "├── " if indent > 0 else ""
        print(f"{prefix}{connector}{icon} {node.label} {mark}{score}")
        for cid in node.children_ids:
            self.print_tree(cid, indent + 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _node_id(label: str, parent_id: str) -> str:
    return hashlib.md5(f"{parent_id}/{label}".encode()).hexdigest()[:12]


def _extract_urls_near_text(markdown: str, text: str, window: int = 800) -> list[str]:
    words = {w.lower() for w in text.split() if len(w) > 3}
    all_urls = re.findall(r"https?://[^\s\)\]>]+", markdown)
    if not all_urls or not words:
        return []
    results, seen = [], set()
    for url in all_urls:
        uc = url.rstrip(".,;:'\"")
        if uc in seen:
            continue
        idx = markdown.find(url)
        if idx == -1:
            continue
        region = markdown[max(0, idx - window):idx + len(url) + window].lower()
        if sum(1 for w in words if w in region) >= 2:
            seen.add(uc)
            results.append(uc)
        if len(results) >= 5:
            break
    return results


def _find_section(markdown: str, text: str) -> str:
    words = [w for w in text.lower().split() if len(w) > 3]
    sections = {}
    current = ""
    for line in markdown.split("\n"):
        if line.startswith("## "):
            current = line.removeprefix("## ").strip()
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    best, best_score = "", 0
    for heading, content in sections.items():
        cl = content.lower()
        score = sum(1 for w in words if w in cl) + 2 * sum(1 for w in words if w in heading.lower())
        if score > best_score:
            best_score = score
            best = heading
    return best


def _parse_sections(md: str) -> dict[str, str]:
    sections = {}
    current = ""
    for line in md.split("\n"):
        if line.startswith("## "):
            current = line.removeprefix("## ").strip()
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    if not sections:
        sections["Full Document"] = md
    return sections


def _get_relevant_text(sections: dict[str, str], label: str, markdown: str, max_chars: int = 4000) -> str:
    words = [w for w in label.lower().split() if len(w) > 3]
    scored = []
    for heading, content in sections.items():
        cl = content.lower()
        s = sum(1 for w in words if w in cl) + 2 * sum(1 for w in words if w in heading.lower())
        scored.append((s, heading, content))
    scored.sort(reverse=True)
    result = ""
    for s, heading, content in scored[:3]:
        if s > 0:
            result += f"### {heading}\n{content[:max_chars // 3]}\n\n"
    return result or markdown[:max_chars]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = (
    "You are a knowledge taxonomy architect that structures documents into "
    "hierarchical knowledge graphs. You output only valid JSON, no markdown "
    "fences, no commentary."
)

PHASE1_PROMPT = """Analyze this research document and discover the top-level knowledge categories that organize the CORE factual information in it.

Rules:
- Output 3-5 categories covering the document's CENTRAL themes — skip tangential or background information
- Categories are ABSTRACT groupings — not specific entities or claims
- Good examples: "Countries Involved", "Key Technologies", "Timeline of Events", "Economic Indicators", "Legal Framework", "Key People and Organizations", "Methodologies Used", "Data Sources", "Risks and Challenges"
- Each category should be distinct — no overlapping scope
- ONLY include categories that are essential to the document's main argument or thesis
- estimated_depth: how many more levels you think this can decompose (1-9)

Output JSON array:
[
  {{
    "label": "Category Name",
    "description": "What knowledge this covers in the document",
    "can_decompose": true,
    "estimated_depth": 4
  }}
]

Document:
{document}"""

DECOMPOSE_PROMPT = """Recursively decompose this knowledge node into either SUBCATEGORIES or verifiable LEAF CLAIMS.

Context:
- Document: "{title}"
- Node: "{node_label}" (depth {depth} of max {max_depth})
- Path: {path}

Relevant text from the document:
{section_text}

Decision rules:
1. If this node is TANGENTIAL or only loosely related to the document's central thesis → output "skip" (no children, no claims)
2. If this node has meaningful, distinct subcategories that are CORE to the document → output "decompose" with children
3. If this node is specific enough to state verifiable facts → output "leaf" with claims
4. At depth {max_depth}, you MUST output claims or skip
5. Subcategories must not overlap
6. Claims must be: factual, specific, checkable by visiting a webpage
7. Aim for 2-4 children per decomposition, 1-3 claims per leaf
8. Be SELECTIVE — only keep subcategories and claims that are central to the document's argument. Omit background context, tangential references, and minor details

For skipping irrelevant nodes:
{{
  "action": "skip"
}}

For subcategories:
{{
  "action": "decompose",
  "children": [
    {{
      "label": "Subcategory or entity name",
      "description": "What this covers",
      "can_decompose": true
    }}
  ]
}}

For leaf claims:
{{
  "action": "leaf",
  "claims": [
    {{
      "label": "Short claim label",
      "claim_text": "Specific verifiable factual assertion from the document",
      "search_query": "Google search query to find evidence for or against this claim",
      "preferred_domain": "most-authoritative-site.com or empty string",
      "critical": true
    }}
  ]
}}"""

LEAF_PROMPT = """Extract 1-5 specific, verifiable factual claims about "{node_label}" from this document section.

Path in knowledge tree: {path}

For each claim provide:
- label: short name (3-6 words)
- claim_text: the specific verifiable fact stated in the document
- search_query: Google query to verify it
- preferred_domain: authoritative domain or ""
- critical: true if essential to the document's argument

Output JSON:
{{"claims": [{{"label": "...", "claim_text": "...", "search_query": "...", "preferred_domain": "", "critical": false}}]}}

Document section:
{section_text}"""

ENRICH_PROMPT = """Extract provenance for this knowledge graph node from the source document.

Node: "{node_label}" (depth {depth})
Path: {path}

Find:
1. source_snippet: the 1-3 sentence excerpt this node is based on (exact text from document)
2. cited_urls: any URLs in the document that relate to this node

Output JSON:
{{"source_snippet": "...", "cited_urls": ["..."]}}

Document excerpt:
{section_text}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KGTaxonomyBuilder:
    """
    Builds the knowledge graph using an LLM for all decomposition decisions.

    Args:
        llm_fn:       async (system_prompt: str, user_prompt: str) -> str
        markdown:     the full research document text (any length)
        title:        document title (auto-detected from # heading if empty)
        max_depth:    maximum tree depth (1-10, default 10)
        max_children: max subcategories per decomposition (default 4)
        max_claims:   max claims per leaf parent (default 3)
        on_node:      optional callback(KGNode) called when each node is created
        concurrency:  max parallel LLM calls per decomposition level (default 5)
    """

    def __init__(
        self,
        llm_fn: LLMFn,
        markdown: str,
        title: str = "",
        max_depth: int = 10,
        max_children: int = 4,
        max_claims: int = 3,
        on_node: Optional[Callable[[KGNode], None]] = None,
        concurrency: int = 5,
    ):
        self.llm = llm_fn
        self.markdown = markdown
        self.max_depth = min(max_depth, 10)
        self.max_children = max_children
        self.max_claims = max_claims
        self.on_node = on_node
        self.concurrency = concurrency

        self.title = title
        if not self.title:
            for line in markdown.split("\n"):
                if line.startswith("# "):
                    self.title = line.removeprefix("# ").strip()
                    break
            if not self.title:
                self.title = "Research Report"

        self.kg = KnowledgeGraphTaxonomy()
        self.kg.title = self.title
        self.kg.max_depth_allowed = self.max_depth
        self._sections = _parse_sections(markdown)

    async def _llm_json(self, user_prompt: str) -> Any:
        raw = await self.llm(SYSTEM_PROMPT, user_prompt)
        cleaned = re.sub(r"```json\s?|```", "", raw).strip()
        return json.loads(cleaned)

    def _node_path(self, node_id: str) -> str:
        path = []
        current = self.kg.nodes.get(node_id)
        while current and current.id != "root":
            path.append(current.label)
            current = self.kg.nodes.get(current.parent_id) if current.parent_id else None
        return " → ".join(reversed(path)) or "root"

    def _relevant_text(self, label: str, max_chars: int = 4000) -> str:
        return _get_relevant_text(self._sections, label, self.markdown, max_chars)

    def _emit(self, node: KGNode):
        if self.on_node:
            self.on_node(node)

    # Phase 1

    async def _discover_categories(self) -> list[dict]:
        logger.info("Phase 1: Discovering top-level categories...")
        prompt = PHASE1_PROMPT.format(document=self.markdown[:14000])
        try:
            cats = await self._llm_json(prompt)
            if isinstance(cats, list):
                return cats[:self.max_children]
        except Exception as e:
            logger.error(f"Category discovery failed: {e}")
        return [
            {"label": h, "description": h, "can_decompose": True, "estimated_depth": 3}
            for h in list(self._sections.keys())[:self.max_children]
        ]

    # Phase 2

    async def _decompose(self, node: KGNode):
        if node.depth >= self.max_depth:
            await self._extract_leaves(node)
            return

        section_text = self._relevant_text(node.label)
        path = self._node_path(node.id)
        prompt = DECOMPOSE_PROMPT.format(
            title=self.title, node_label=node.label, depth=node.depth,
            max_depth=self.max_depth, path=path, section_text=section_text,
        )

        try:
            result = await self._llm_json(prompt)
        except Exception as e:
            logger.warning(f"Decompose failed for '{node.label}': {e}")
            node.node_type = NodeType.CLAIM
            node.claim_text = f"The document discusses {node.label}"
            return

        action = result.get("action", "leaf")

        if action == "skip":
            node.node_type = NodeType.ENTITY
            return

        if action == "decompose" and result.get("children"):
            child_tasks = []
            for cd in result["children"][:self.max_children]:
                child_label = cd.get("label", "Unknown")
                child_id = _node_id(child_label, node.id)
                can_decompose = cd.get("can_decompose", True)
                child_node = KGNode(
                    id=child_id, label=child_label,
                    node_type=NodeType.CATEGORY if can_decompose else NodeType.ENTITY,
                    depth=node.depth + 1, parent_id=node.id,
                    source_section=_find_section(self.markdown, child_label),
                )
                self.kg.add_node(child_node)
                self._emit(child_node)
                if can_decompose and child_node.depth < self.max_depth:
                    child_tasks.append(self._decompose(child_node))
                else:
                    child_tasks.append(self._extract_leaves(child_node))

            sem = asyncio.Semaphore(self.concurrency)
            async def bounded(coro):
                async with sem:
                    return await coro
            await asyncio.gather(*[bounded(t) for t in child_tasks])

        elif action == "leaf" and result.get("claims"):
            self._add_claims(node, result["claims"])
        else:
            node.node_type = NodeType.CLAIM
            node.claim_text = f"The document discusses {node.label}"

    async def _extract_leaves(self, node: KGNode):
        section_text = self._relevant_text(node.label, max_chars=3000)
        path = self._node_path(node.id)
        prompt = LEAF_PROMPT.format(
            node_label=node.label, path=path, section_text=section_text,
        )
        try:
            result = await self._llm_json(prompt)
            claims = result.get("claims", [])
            if claims:
                self._add_claims(node, claims)
            else:
                node.node_type = NodeType.CLAIM
                node.claim_text = f"The document discusses {node.label}"
        except Exception as e:
            logger.warning(f"Leaf extraction failed for '{node.label}': {e}")
            node.node_type = NodeType.CLAIM
            node.claim_text = f"The document discusses {node.label}"

    def _add_claims(self, parent: KGNode, claims_data: list[dict]):
        for cd in claims_data[:self.max_claims]:
            label = cd.get("label", "Claim")
            cid = _node_id(label, parent.id)
            urls = _extract_urls_near_text(self.markdown, cd.get("claim_text", label))
            claim_node = KGNode(
                id=cid, label=label, node_type=NodeType.CLAIM,
                depth=parent.depth + 1, parent_id=parent.id,
                claim_text=cd.get("claim_text"),
                search_query=cd.get("search_query"),
                preferred_domain=cd.get("preferred_domain", ""),
                critical=cd.get("critical", False),
                cited_urls=urls,
                source_section=_find_section(self.markdown, cd.get("claim_text", "")),
            )
            self.kg.add_node(claim_node)
            self._emit(claim_node)

    # Phase 2B

    async def _enrich(self, node: KGNode):
        if node.source_snippet:
            return
        section_text = self._relevant_text(node.label, max_chars=2000)
        path = self._node_path(node.id)
        prompt = ENRICH_PROMPT.format(
            node_label=node.label, depth=node.depth, path=path, section_text=section_text,
        )
        try:
            result = await self._llm_json(prompt)
            node.source_snippet = result.get("source_snippet", "")
            for u in result.get("cited_urls", []):
                if u not in node.cited_urls:
                    node.cited_urls.append(u)
        except Exception:
            pass

    # Main

    async def build(self) -> KnowledgeGraphTaxonomy:
        """Execute full pipeline: discover → decompose → enrich → aggregate."""
        categories = await self._discover_categories()
        self.kg.research_question = self.title

        cat_nodes = []
        for cat in categories:
            cid = _node_id(cat["label"], "root")
            node = KGNode(
                id=cid, label=cat["label"], node_type=NodeType.CATEGORY,
                depth=1, parent_id="root",
                source_section=_find_section(self.markdown, cat["label"]),
            )
            self.kg.add_node(node)
            self._emit(node)
            cat_nodes.append(node)

        logger.info(f"Phase 1: {len(cat_nodes)} categories")
        await asyncio.gather(*[self._decompose(n) for n in cat_nodes])
        logger.info(f"Phase 2: {len(self.kg.nodes)} nodes, depth {self.kg.max_depth}, {len(self.kg.get_leaves())} leaves")

        non_leaves = [n for n in self.kg.nodes.values() if not n.is_leaf and n.node_type != NodeType.ROOT]
        if non_leaves:
            sem = asyncio.Semaphore(self.concurrency)
            async def be(n):
                async with sem:
                    return await self._enrich(n)
            await asyncio.gather(*[be(n) for n in non_leaves])

        self.kg.aggregate_scores()
        return self.kg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def verify_knowledge_graph(
    kg: KnowledgeGraphTaxonomy,
    verify_fn: VerifyFn,
    max_concurrent: int = 5,
) -> KnowledgeGraphTaxonomy:
    """
    Verify all leaf nodes.

    verify_fn(claim, urls, search_query, preferred_domain) -> {
        "verdict": "passed"|"failed"|"skipped",
        "evidence_urls": [...],
        "errors": [...]
    }
    """
    leaves = kg.get_leaves()
    logger.info(f"Verifying {len(leaves)} leaves...")
    sem = asyncio.Semaphore(max_concurrent)

    async def _verify_one(node: KGNode):
        if not node.claim_text:
            node.verdict = Verdict.SKIPPED
            return
        async with sem:
            try:
                result = await verify_fn(
                    claim=node.claim_text, urls=node.cited_urls,
                    search_query=node.search_query or "",
                    preferred_domain=node.preferred_domain,
                )
                node.verdict = Verdict(result.get("verdict", "failed"))
                node.evidence_urls = result.get("evidence_urls", [])
                node.errors = result.get("errors", [])
            except Exception as e:
                logger.error(f"Verify failed for '{node.label}': {e}")
                node.verdict = Verdict.FAILED
                node.errors = [str(e)]

    await asyncio.gather(*[_verify_one(l) for l in leaves])
    kg.aggregate_scores()
    passed = sum(1 for l in leaves if l.verdict == Verdict.PASSED)
    logger.info(f"Done: {passed}/{len(leaves)} passed")
    return kg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Function Factories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_llm_fn_anthropic(api_key: str, model: str = "claude-opus-4-6-20250219", max_tokens: int = 4096) -> LLMFn:
    """Anthropic Messages API."""
    import httpx
    async def call(system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model, "max_tokens": max_tokens, "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
    return call


def create_llm_fn_openai(base_url: str, api_key: str, model: str) -> LLMFn:
    """OpenAI-compatible endpoint."""
    import httpx
    async def call(system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    return call


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build knowledge graph taxonomy from a research document")
    parser.add_argument("file", help="Markdown file to process")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--model", default="claude-opus-4-6-20250219")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--output", default="kg_output.json")
    parser.add_argument("--tree", action="store_true", help="Print ASCII tree")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(args.file, "r", encoding="utf-8") as f:
        markdown = f.read()

    llm = create_llm_fn_anthropic(api_key=args.api_key, model=args.model)
    builder = KGTaxonomyBuilder(
        llm_fn=llm, markdown=markdown, max_depth=args.max_depth,
        on_node=lambda n: logger.info(f"  {'  ' * n.depth}+ {n.label} (d{n.depth})"),
    )
    kg = await builder.build()

    if args.tree:
        print()
        kg.print_tree()
        print()

    output = kg.to_tree_dict()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote {args.output}")

    leaves = kg.get_leaves()
    print(f"\nNodes: {len(kg.nodes)}  Leaves: {len(leaves)}  Max depth: {kg.max_depth}")


if __name__ == "__main__":
    asyncio.run(main())