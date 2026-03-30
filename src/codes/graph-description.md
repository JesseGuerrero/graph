# Knowledge Graph System Description

This document describes how the knowledge graph (KG) works in this project, for use as context by another LLM or Playwright-based verification system.

## Architecture Overview

```
Research Document (markdown with ## sections, URLs)
        |
  Phase 1: Category Discovery — LLM reads full doc, emits 3-5 top-level categories
        |
  Phase 2: Recursive Decomposition — each category splits into subcategories or leaf claims
        |
  Phase 2B: Provenance Enrichment — link each node to source text + cited URLs
        |
  Phase 3: Leaf Verification — each claim fact-checked against web sources
        |
  Phase 4: Bottom-Up Score Aggregation — leaf verdicts roll up as percentage scores
        |
  Verified Knowledge Graph (JSON tree rendered as interactive DOM tree)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/taxonomy.py` | KG builder, data structures, LLM prompts, verification, score aggregation |
| `web/app.py` | FastAPI server with SSE streaming endpoints for build/verify |
| `web/static/js/knowledge_graph.js` | Frontend tree rendering, verification animations, verdict propagation |
| `web/static/knowledge_graph.html` | KG page layout and CSS styling |
| `web/storm_runner.py` | STORM pipeline runner (generates research articles that feed into the KG) |

## Node Types and Tree Structure

The KG is a tree (directed parent-child) with four node types:

```
ROOT (depth 0, synthetic "Knowledge Graph" node)
 +-- CATEGORY (depth 1) — abstract groupings like "Countries Involved"
 |    +-- CATEGORY (depth 2) — subcategories like "Federal Policy"
 |    |    +-- ENTITY (depth 3) — named things like "United States"
 |    |    |    +-- CLAIM (leaf) — verifiable fact: "CHIPS Act allocated $52.7B"
 |    |    +-- CLAIM (leaf)
 |    +-- ENTITY (depth 2)
 |         +-- CLAIM (leaf)
 +-- CATEGORY (depth 1)
      +-- ...up to depth 10
```

## Data Structures (Python)

### KGNode — a single node in the tree

```python
# src/taxonomy.py lines 152-209
@dataclass
class KGNode:
    id: str                              # MD5(parent_id/label)[:12]
    label: str                           # Human-readable name
    node_type: NodeType                  # ROOT | CATEGORY | ENTITY | CLAIM
    depth: int = 0                       # 0 (root) to 10 (max)
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    # Provenance — where in the document this node came from
    source_section: str = ""             # Which ## heading
    source_snippet: str = ""             # Exact text excerpt
    cited_urls: list[str] = field(default_factory=list)

    # Leaf verification fields
    claim_text: Optional[str] = None     # The verifiable assertion
    search_query: Optional[str] = None   # Google query to find evidence
    preferred_domain: str = ""           # Most authoritative site
    verdict: Verdict = Verdict.UNVERIFIED  # passed | failed | partial | skipped
    evidence_urls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    critical: bool = False               # If True, failure zeros parent score

    # Aggregation
    aggregated_score: Optional[float] = None  # 0.0-1.0
    child_pass: int = 0
    child_fail: int = 0
    child_total: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0
```

### KnowledgeGraphTaxonomy — flat node store with tree operations

```python
# src/taxonomy.py lines 216-353
class KnowledgeGraphTaxonomy:
    def __init__(self):
        self.nodes: dict[str, KGNode] = {}
        self.nodes["root"] = KGNode(id="root", label="Knowledge Graph",
                                     node_type=NodeType.ROOT, depth=0)

    def add_node(self, node: KGNode) -> KGNode:
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
        return node

    def get_leaves(self) -> list[KGNode]:
        return [n for n in self.nodes.values()
                if n.is_leaf and n.node_type != NodeType.ROOT]

    def to_tree_dict(self, node_id: str = "root") -> dict:
        """Recursive nested JSON for the frontend."""
        node = self.nodes.get(node_id)
        if not node:
            return {}
        result = node.to_dict()
        result["children"] = [self.to_tree_dict(cid) for cid in node.children_ids]
        return result
```

## How the KG is Dynamically Built

### Phase 1: Category Discovery

The builder sends the first ~14KB of the document to the LLM and asks for 3-5 abstract categories:

```python
# src/taxonomy.py lines 624-638
async def _discover_categories(self) -> list[dict]:
    prompt = PHASE1_PROMPT.format(document=self.markdown[:14000])
    cats = await self._llm_json(prompt)
    return cats[:self.max_children]  # Capped at max_children (default 4)
```

LLM output format:
```json
[
  {"label": "Countries Involved", "description": "...", "can_decompose": true, "estimated_depth": 4},
  {"label": "Key Technologies", "description": "...", "can_decompose": true, "estimated_depth": 3}
]
```

### Phase 2: Recursive Decomposition

For each category, the LLM decides: decompose into subcategories, emit leaf claims, or skip:

```python
# src/taxonomy.py lines 642-697
async def _decompose(self, node: KGNode):
    if node.depth >= self.max_depth:
        await self._extract_leaves(node)
        return

    section_text = self._relevant_text(node.label)
    result = await self._llm_json(DECOMPOSE_PROMPT.format(
        title=self.title, node_label=node.label,
        depth=node.depth, max_depth=self.max_depth,
        path=self._node_path(node.id), section_text=section_text,
    ))

    action = result.get("action", "leaf")  # "decompose" | "leaf" | "skip"

    if action == "decompose" and result.get("children"):
        child_tasks = []
        for cd in result["children"][:self.max_children]:
            child_node = KGNode(
                id=_node_id(cd["label"], node.id),
                label=cd["label"],
                node_type=NodeType.CATEGORY if cd.get("can_decompose") else NodeType.ENTITY,
                depth=node.depth + 1,
                parent_id=node.id,
            )
            self.kg.add_node(child_node)
            self._emit(child_node)  # Triggers SSE event to frontend
            child_tasks.append(self._decompose(child_node))

        # Parallel execution with concurrency limit
        sem = asyncio.Semaphore(self.concurrency)
        async def bounded(coro):
            async with sem:
                return await coro
        await asyncio.gather(*[bounded(t) for t in child_tasks])

    elif action == "leaf" and result.get("claims"):
        self._add_claims(node, result["claims"])
```

Node IDs are deterministic hashes: `MD5(parent_id/label)[:12]`

```python
def _node_id(label: str, parent_id: str) -> str:
    return hashlib.md5(f"{parent_id}/{label}".encode()).hexdigest()[:12]
```

### Full Build Pipeline

```python
# src/taxonomy.py lines 757-787
async def build(self) -> KnowledgeGraphTaxonomy:
    # Phase 1: discover top-level categories
    categories = await self._discover_categories()
    for cat in categories:
        node = KGNode(id=_node_id(cat["label"], "root"), label=cat["label"],
                      node_type=NodeType.CATEGORY, depth=1, parent_id="root")
        self.kg.add_node(node)

    # Phase 2: recursive decomposition (parallel per level)
    await asyncio.gather(*[self._decompose(n) for n in cat_nodes])

    # Phase 2B: enrich non-leaf nodes with source snippets + URLs
    non_leaves = [n for n in self.kg.nodes.values()
                  if not n.is_leaf and n.node_type != NodeType.ROOT]
    await asyncio.gather(*[self._enrich(n) for n in non_leaves])

    # Phase 4: bottom-up score aggregation
    self.kg.aggregate_scores()
    return self.kg
```

## Score Aggregation (Mind2Web 2 Gate-Then-Average)

Scores propagate bottom-up from leaves to root:

```python
# src/taxonomy.py lines 255-317
def aggregate_scores(self):
    for depth in range(self.max_depth, -1, -1):
        for node in self.get_nodes_at_depth(depth):
            if node.is_leaf:
                # Leaf: 1.0 if passed, 0.0 if failed, None if unverified
                node.aggregated_score = {
                    Verdict.PASSED: 1.0,
                    Verdict.FAILED: 0.0,
                }.get(node.verdict)
            else:
                critical = [c for c in scored_children if c.critical]
                non_critical = [c for c in scored_children if not c.critical]

                # Gate: any critical child fails -> parent = 0
                if any(c.aggregated_score < 1.0 for c in critical):
                    node.aggregated_score = 0.0

                # Non-critical exist: average them
                elif non_critical:
                    node.aggregated_score = (
                        sum(c.aggregated_score for c in non_critical) / len(non_critical)
                    )

                # All children critical: AND gate (all must pass)
                else:
                    all_pass = all(c.aggregated_score == 1.0 for c in critical)
                    node.aggregated_score = 1.0 if all_pass else 0.0
```

## Backend API Endpoints

### GET `/api/articles/{id}/kg`
Returns cached `kg_taxonomy.json` or 404.

### POST `/api/articles/{id}/kg/build`
Streams SSE events as the KG is built:

```python
# web/app.py lines 337-418
@app.post("/api/articles/{article_id}/kg/build")
def build_kg(article_id: str):
    q = queue.Queue()

    def run_build():
        llm = create_llm_fn_openai(base_url=api_base, api_key=api_key, model=model)
        builder = KGTaxonomyBuilder(
            llm_fn=llm, markdown=article_text, title=topic,
            max_depth=6,
            on_node=lambda node: q.put({
                "type": "node",
                "data": {"label": node.label, "depth": node.depth,
                         "node_type": node.node_type.value}
            }),
        )
        kg = loop.run_until_complete(builder.build())
        tree = kg.to_tree_dict()
        json.dump(tree, open(kg_path, "w"), indent=2)  # Cache to disk
        q.put({"type": "done", "data": tree})

    threading.Thread(target=run_build, daemon=True).start()
    return StreamingResponse(stream_from_queue(q), media_type="text/event-stream")
```

SSE events emitted:
- `{type: "node", data: {label, depth, node_type}}` — each node as discovered
- `{type: "done", data: <full tree JSON>}` — build complete
- `{type: "error", data: {message}}` — on failure

### POST `/api/articles/{id}/kg/verify`
Verifies each leaf claim against web sources via browser + LLM judge. Streams:
- `{type: "claim_start", id, text}` — about to verify this claim
- `{type: "claim_result", id, verdict, evidence_urls, errors, reasoning}` — result
- `{type: "done", score, passed, failed, total_claims, error_summary}` — final

## Frontend Tree Rendering

The tree is rendered as pure DOM elements (no D3.js or graph library):

### Tree Data Mapping

```javascript
// web/static/js/knowledge_graph.js lines 189-216
function mapKGToTree(node, depth = 0) {
  const isClaim = node.type === 'claim' || (!node.children || node.children.length === 0);
  const hasChildren = node.children && node.children.length > 0;

  let treeType;
  if (depth === 0) treeType = 'root';          // Gold border
  else if (isClaim) treeType = 'leaf';          // Purple border
  else if (node.type === 'entity') treeType = 'check';  // Green border
  else treeType = 'section';                    // Blue border

  return {
    id: node.id,
    label: truncate(node.label, 40),
    type: treeType,
    critical: node.critical || false,
    collapsible: hasChildren && depth > 0,
    collapsed: hasChildren && depth >= 1,
    citedUrls: node.cited_urls || [],
    searchQuery: node.search_query || '',
    claimText: node.claim_text || '',
    children: hasChildren ? node.children.map(c => mapKGToTree(c, depth + 1)) : [],
  };
}
```

### DOM Tree Construction

```javascript
// web/static/js/knowledge_graph.js lines 223-265
function buildNodeEl(data) {
  const wrapper = el('div', `node type-${data.type} ${data.collapsible ? 'collapsible' : ''}`);

  const box = el('div', 'node-box');
  box.id = `node-${data.id}`;
  box.innerHTML = `<div class="label">${esc(data.label)}</div>` +
    (data.desc ? `<div class="desc">${esc(truncate(data.desc, 80))}</div>` : '');

  if (data.critical) box.innerHTML += '<span class="badge critical">CRITICAL</span>';
  box._nodeData = data;  // Stored for hover popups
  wrapper.appendChild(box);

  // Click to collapse/expand
  if (data.collapsible) {
    box.addEventListener('click', () => {
      wrapper.classList.toggle('collapsed');
      setTimeout(drawHLines, 50);  // Redraw connectors
    });
  }

  // Recursively build children
  if (data.children && data.children.length) {
    wrapper.appendChild(el('div', 'v-line'));       // Vertical connector
    const kids = el('div', 'children h-line-container');
    data.children.forEach(child => {
      const col = el('div', 'child-col');
      col.appendChild(el('div', 'v-line'));          // Vertical line to child
      col.appendChild(buildNodeEl(child));           // Recurse
      kids.appendChild(col);
    });
    wrapper.appendChild(kids);
  }
  return wrapper;
}
```

### Horizontal Connector Lines

```javascript
// web/static/js/knowledge_graph.js lines 267-280
function drawHLines() {
  document.querySelectorAll('.h-line-container').forEach(kids => {
    kids.querySelectorAll('.h-connector').forEach(h => h.remove());
    const cols = Array.from(kids.children)
      .filter(c => c.classList.contains('child-col') && c.offsetParent !== null);
    if (cols.length > 1) {
      const rects = cols.map(c => c.getBoundingClientRect());
      const kr = kids.getBoundingClientRect();
      const hl = document.createElement('div');
      hl.className = 'h-connector';
      hl.style.cssText = `position:absolute;top:0;` +
        `left:${rects[0].left - kr.left + rects[0].width / 2}px;` +
        `right:${kr.right - rects[rects.length-1].right + rects[rects.length-1].width/2}px;` +
        `height:2px;background:var(--border-active);`;
      kids.appendChild(hl);
    }
  });
}
```

## Verification UI Flow

During verification, the frontend animates each claim being checked:

```javascript
// web/static/js/knowledge_graph.js lines 340-413
// On claim_start: collapse tree, expand only path to this node, animate particle
if (event.type === 'claim_start') {
  collapseAllToDepth1();
  const nodeEl = document.getElementById(`node-${event.id}`);
  expandPathToNode(nodeEl);           // Uncollapse ancestors
  nodeEl.classList.add('pending');    // Pulsing glow animation

  // Highlight matching text in markdown panel
  const hl = highlightClaimInMarkdown(event.text, event.id);

  // Fly "Verify" particle from markdown to tree node (0.8s animation)
  await flyFactParticle(hl, nodeEl);
}

// On claim_result: apply verdict badge and propagate scores up
if (event.type === 'claim_result') {
  nodeEl.classList.add(event.verdict === 'passed' ? 'passed' : 'failed');

  // Verdict badge (checkmark or X)
  const badge = document.createElement('span');
  badge.className = `verdict-badge ${event.verdict === 'passed' ? 'pass' : 'fail'}`;
  badge.textContent = event.verdict === 'passed' ? '\u2713' : '\u2717';
  nodeEl.appendChild(badge);

  // Error type badges (INF, CRI, IAT, MAT, SYN, RET)
  addErrorBadges(nodeEl, event.errors);

  // Propagate verdict up to parent nodes (gate-then-average)
  propagateVerdict(nodeEl);
}
```

## JSON Tree Shape (for Playwright verification)

A built KG saved to `kg_taxonomy.json` has this recursive structure:

```json
{
  "id": "root",
  "label": "Knowledge Graph",
  "type": "root",
  "depth": 0,
  "children": [
    {
      "id": "a1b2c3d4e5f6",
      "label": "Countries Involved",
      "type": "category",
      "depth": 1,
      "source_section": "Global AI Governance",
      "source_snippet": "Multiple countries have enacted...",
      "cited_urls": ["https://example.com/source1"],
      "claim_text": null,
      "search_query": null,
      "verdict": "partial",
      "aggregated_score": 0.75,
      "critical": false,
      "children": [
        {
          "id": "f6e5d4c3b2a1",
          "label": "CHIPS Act funding",
          "type": "claim",
          "depth": 2,
          "claim_text": "The CHIPS Act allocated $52.7 billion to semiconductor manufacturing",
          "search_query": "CHIPS Act semiconductor funding amount",
          "preferred_domain": "congress.gov",
          "verdict": "passed",
          "evidence_urls": ["https://congress.gov/..."],
          "errors": [],
          "critical": true,
          "aggregated_score": 1.0,
          "children": []
        }
      ]
    }
  ]
}
```

## How to Verify the KG Tree with Playwright

The tree renders in the browser at `/static/knowledge_graph.html?id=<article_id>`. Key selectors for Playwright:

### Page Load and Tree Presence

```javascript
// Navigate to the KG page
await page.goto('http://127.0.0.1:8005/static/knowledge_graph.html?id=<article_id>');

// Wait for the tree to render (either from cache or after build)
await page.waitForSelector('#tree .node.type-root', { timeout: 120000 });

// Verify root node exists
const root = await page.locator('#tree .type-root .node-box');
await expect(root).toBeVisible();
```

### Tree Structure Verification

```javascript
// Count all node boxes in the tree
const allNodes = await page.locator('#tree .node-box').count();
console.log(`Total nodes rendered: ${allNodes}`);

// Count by type (CSS classes: type-root, type-section, type-check, type-leaf)
const categories = await page.locator('#tree .type-section').count();
const entities = await page.locator('#tree .type-check').count();
const claims = await page.locator('#tree .type-leaf').count();
console.log(`Categories: ${categories}, Entities: ${entities}, Claims: ${claims}`);

// Verify tree has depth (children containers exist)
const childContainers = await page.locator('#tree .children').count();
expect(childContainers).toBeGreaterThan(0);
```

### Expand Collapsed Nodes

```javascript
// All non-root nodes start collapsed (depth >= 1)
// Click a category node to expand it
await page.locator('.type-section .node-box').first().click();
await page.waitForTimeout(100);  // Wait for connector redraw

// Verify children are now visible
const firstSection = page.locator('.type-section').first();
const childNodes = firstSection.locator('.children .node-box');
await expect(childNodes.first()).toBeVisible();
```

### Read Node Content

```javascript
// Get the label text of all visible leaf nodes
const leafLabels = await page.locator('.type-leaf:not(.collapsed) .node-box .label')
  .allTextContents();
console.log('Leaf claims:', leafLabels);

// Get a specific node's data via its ID
const nodeBox = page.locator('#node-a1b2c3d4e5f6');
const label = await nodeBox.locator('.label').textContent();
const desc = await nodeBox.locator('.desc').textContent();
```

### Verify After Verification Run

```javascript
// After verification completes, check for verdict badges
const passedNodes = await page.locator('.node-box.passed').count();
const failedNodes = await page.locator('.node-box.failed').count();
console.log(`Passed: ${passedNodes}, Failed: ${failedNodes}`);

// Check verdict badges (checkmark/X overlays)
const passBadges = await page.locator('.verdict-badge.pass').count();
const failBadges = await page.locator('.verdict-badge.fail').count();

// Read the score dashboard
const scoreText = await page.locator('#processStatus').textContent();
// e.g. "Score: 75.0% — 15/20 claims passed"

// Read score cards
const overallScore = await page.locator('#scoreCards .score-card .value').first().textContent();
```

### Screenshot the Tree

```javascript
// Screenshot the full tree section
await page.locator('#tree-section').screenshot({ path: 'kg-tree.png' });

// Screenshot a specific node with its children expanded
await page.locator('.type-section').first().locator('.node-box').click();  // expand
await page.waitForTimeout(200);
await page.locator('.type-section').first().screenshot({ path: 'category-expanded.png' });

// Full page screenshot
await page.screenshot({ path: 'kg-full-page.png', fullPage: true });
```

### Check Error Badges

```javascript
// Error types: INF, CRI, IAT, MAT, SYN, RET
const activeErrors = await page.locator('.error-badge.active').allTextContents();
console.log('Active errors:', activeErrors);

// Check root error summary
const rootSummary = await page.locator('#node-root .error-summary .error-badge.active')
  .allTextContents();
```

### Hover Popup Inspection

```javascript
// Hover over a node to see its popup
await page.locator('#node-a1b2c3d4e5f6').hover();
await page.waitForSelector('.node-popup', { timeout: 1000 });

// Read popup content
const popupTitle = await page.locator('.node-popup .popup-title').textContent();
const popupClaim = await page.locator('.node-popup .popup-text').first().textContent();
const popupVerdict = await page.locator('.node-popup .popup-verdict').textContent();
```

## CSS Selectors Quick Reference

| What | Selector |
|------|----------|
| Root node box | `#node-root` or `.type-root .node-box` |
| Category nodes | `.type-section .node-box` |
| Entity nodes | `.type-check .node-box` |
| Leaf/claim nodes | `.type-leaf .node-box` |
| Any node by ID | `#node-{id}` |
| Passed nodes | `.node-box.passed` |
| Failed nodes | `.node-box.failed` |
| Pending (verifying) | `.node-box.pending` |
| Verdict badge | `.verdict-badge` (`.pass`, `.fail`, `.partial`) |
| Error badges | `.error-badge.active` |
| Critical badge | `.badge.critical` |
| Collapsed node | `.node.collapsed` |
| Children container | `.children.h-line-container` |
| Horizontal connector | `.h-connector` |
| Vertical connector | `.v-line` |
| Score dashboard | `#scoreDashboard` |
| Process status | `#processStatus` |
| Build button | `#build-btn` |
| Verify button | `#verifyBtn` |
| Markdown panel | `#md-panel` |
| Fact highlight | `.fact-highlight` |
| Node hover popup | `.node-popup` |

## Running the Server

```bash
cd web && uv run uvicorn app:app --host 127.0.0.1 --port 8005
```

The KG page is at: `http://127.0.0.1:8005/static/knowledge_graph.html?id=<article_id>`
