const params = new URLSearchParams(window.location.search);
const articleId = params.get('id');

const ERROR_TYPES = [
  {id:'inf', label:'INF', name:'Info Not Found', desc:'No evidence could be retrieved'},
  {id:'cri', label:'CRI', name:'Criteria Violation', desc:'Evidence contradicts the claim'},
  {id:'iat', label:'IAT', name:'Invalid Attribution', desc:'Cited URL is expired or dead'},
  {id:'mat', label:'MAT', name:'Missing Attribution', desc:'No source URL cited for this claim'},
  {id:'syn', label:'SYN', name:'Synthesis Error', desc:'Information incorrectly synthesized'},
  {id:'ret', label:'RET', name:'Retrieval Error', desc:'Source is irrelevant to the claim'},
];

let treeAnimating = false;
let verifyAbort = null;
let kgTreeData = null;
let articleMarkdown = '';

// ── Init ──

document.addEventListener('DOMContentLoaded', async () => {
  if (!articleId) return;
  document.getElementById('back-to-article').href = `/static/article.html?id=${articleId}`;
  loadSidebarHistory();

  // Load article markdown
  try {
    const artRes = await fetch(`/api/articles/${articleId}`);
    if (artRes.ok) {
      const art = await artRes.json();
      articleMarkdown = art.article_text || '';
    }
  } catch (e) {}

  // Try cached KG
  try {
    const res = await fetch(`/api/articles/${articleId}/kg`);
    if (res.ok) {
      kgTreeData = await res.json();
      showTree(kgTreeData);
      return;
    }
  } catch (e) {}

  showBuildPrompt();
});

async function loadSidebarHistory() {
  const container = document.getElementById('sidebar-history');
  if (!container) return;
  try {
    const res = await fetch('/api/articles');
    const articles = await res.json();
    articles.slice(0, 10).forEach(a => {
      const link = document.createElement('a');
      link.href = `/static/knowledge_graph.html?id=${a.id}`;
      link.textContent = a.topic;
      link.title = a.topic;
      container.appendChild(link);
    });
  } catch (e) {}
}

// ── Build ──

function showBuildPrompt() {
  const container = document.getElementById('kg-container');
  container.innerHTML = `
    <div class="build-prompt">
      <p>No verification tree built for this article yet.</p>
      <button class="action" id="build-btn">Build Verification Tree</button>
      <div class="status" id="build-status" style="display:none"></div>
      <div class="nodes-log" id="build-nodes" style="display:none"></div>
    </div>`;
  document.getElementById('build-btn').addEventListener('click', startBuild);
}

function startBuild() {
  const btn = document.getElementById('build-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Building...';

  const status = document.getElementById('build-status');
  const nodesDiv = document.getElementById('build-nodes');
  status.style.display = '';
  nodesDiv.style.display = '';
  status.textContent = 'Discovering categories...';

  fetch(`/api/articles/${articleId}/kg/build`, { method: 'POST' })
    .then(res => {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      function processBuffer(flush) {
        if (flush) buffer += '\n';
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === 'node') {
              const d = event.data;
              const indent = '\u00a0\u00a0'.repeat(d.depth);
              const icons = { root: '\u25c6', category: '\u25c7', entity: '\u25cf', claim: '\u25ce' };
              nodesDiv.innerHTML += `<div>${indent}${icons[d.node_type] || '\u00b7'} ${esc(d.label)}</div>`;
              nodesDiv.scrollTop = nodesDiv.scrollHeight;
              status.textContent = `Building... (${nodesDiv.children.length} nodes)`;
            } else if (event.type === 'done') {
              status.textContent = 'Done!';
              kgTreeData = event.data;
              setTimeout(() => showTree(kgTreeData), 500);
            } else if (event.type === 'error') {
              status.textContent = 'Error: ' + (event.data.message || 'Unknown');
              btn.disabled = false;
              btn.textContent = 'Retry';
            }
          } catch (e) { console.error('SSE parse error', e); }
        }
      }
      function read() {
        reader.read().then(({ done, value }) => {
          if (value) buffer += decoder.decode(value, { stream: true });
          processBuffer(done);
          if (!done) read();
        });
      }
      read();
    })
    .catch(err => {
      status.textContent = 'Error: ' + err.message;
      btn.disabled = false;
      btn.textContent = 'Retry';
    });
}

// ── Tree Rendering ──

function showTree(tree) {
  document.getElementById('kg-container').style.display = 'none';
  document.getElementById('tree-section').style.display = '';

  // Show markdown panel
  if (articleMarkdown) {
    document.getElementById('md-panel').style.display = '';
    document.getElementById('md-body').innerHTML =
      `<div class="rendered-md">${marked.parse(articleMarkdown)}</div>`;
  }

  const treeEl = document.getElementById('tree');
  treeEl.innerHTML = '';

  const treeData = mapKGToTree(tree);
  treeEl.appendChild(buildNodeEl(treeData));
  setTimeout(drawHLines, 100);

  // Count claims
  const claims = countClaims(tree);
  document.getElementById('processStatus').textContent = `${claims} verifiable claims`;
  document.getElementById('verifyBtn').disabled = false;

  // Wire controls
  document.getElementById('verifyBtn').onclick = startVerification;
  document.getElementById('killBtn').onclick = killVerification;
  document.getElementById('resetBtn').onclick = resetTree;
  document.getElementById('resetBtn').disabled = false;
}

function mapKGToTree(node, depth = 0) {
  const isRoot = depth === 0;
  const isClaim = node.type === 'claim' || (!node.children || node.children.length === 0);
  const isEntity = node.type === 'entity';
  const hasChildren = node.children && node.children.length > 0;

  let treeType;
  if (isRoot) treeType = 'root';
  else if (isClaim) treeType = 'leaf';
  else if (isEntity) treeType = 'check';
  else treeType = 'section';

  const result = {
    id: node.id || `node_${depth}_${Math.random().toString(36).slice(2, 6)}`,
    label: truncate(node.label || '', 40),
    desc: node.claim_text || node.source_snippet || '',
    type: treeType,
    strategy: isRoot ? 'SEQ' : 'PAR',
    critical: node.critical || false,
    collapsible: hasChildren && !isRoot,
    collapsed: hasChildren && depth >= 1,
    citedUrls: node.cited_urls || [],
    searchQuery: node.search_query || '',
    claimText: node.claim_text || '',
    children: hasChildren ? node.children.map(c => mapKGToTree(c, depth + 1)) : [],
  };
  return result;
}

function countClaims(node) {
  if (node.type === 'claim' || (!node.children || node.children.length === 0)) return 1;
  return (node.children || []).reduce((n, c) => n + countClaims(c), 0);
}

function buildNodeEl(data) {
  const classes = ['node', `type-${data.type}`];
  if (data.collapsible) classes.push('collapsible');
  if (data.collapsed) classes.push('collapsed');
  const wrapper = el('div', classes.join(' '));

  const box = el('div', 'node-box');
  box.id = `node-${data.id}`;
  box.innerHTML = `<div class="label">${esc(data.label)}</div>` +
    (data.desc ? `<div class="desc">${esc(truncate(data.desc, 80))}</div>` : '');

  if (data.critical) box.innerHTML += '<span class="badge critical">CRITICAL</span>';
  if (data.strategy) box.innerHTML += `<span class="badge strategy">${data.strategy}</span>`;

  if (data.citedUrls && data.citedUrls.length) {
    const urls = el('div', 'evidence-links');
    urls.style.marginTop = '4px';
    data.citedUrls.forEach(u => {
      const a = document.createElement('a');
      try { a.textContent = new URL(u).hostname.replace('www.', ''); } catch { a.textContent = u.substring(0, 30); }
      a.href = u; a.target = '_blank';
      urls.appendChild(a);
    });
    box.appendChild(urls);
  }

  wrapper.appendChild(box);

  if (data.collapsible) {
    box.addEventListener('click', () => {
      if (!treeAnimating) {
        wrapper.classList.toggle('collapsed');
        setTimeout(drawHLines, 50);
      }
    });
  }

  if (data.children && data.children.length) {
    wrapper.appendChild(el('div', 'v-line'));
    const kids = el('div', 'children h-line-container');
    data.children.forEach(child => {
      const col = el('div', 'child-col');
      col.appendChild(el('div', 'v-line'));
      col.appendChild(buildNodeEl(child));
      kids.appendChild(col);
    });
    wrapper.appendChild(kids);
  }

  return wrapper;
}

function drawHLines() {
  document.querySelectorAll('.h-line-container').forEach(kids => {
    kids.querySelectorAll('.h-connector').forEach(h => h.remove());
    const cols = Array.from(kids.children).filter(c => c.classList.contains('child-col') && c.offsetParent !== null);
    if (cols.length > 1) {
      const rects = cols.map(c => c.getBoundingClientRect());
      const kr = kids.getBoundingClientRect();
      const hl = document.createElement('div');
      hl.className = 'h-connector';
      hl.style.cssText = `position:absolute;top:0;left:${rects[0].left - kr.left + rects[0].width / 2}px;right:${kr.right - rects[rects.length - 1].right + rects[rects.length - 1].width / 2}px;height:2px;background:var(--border-active);`;
      kids.appendChild(hl);
    }
  });
}

// ── Tree path helpers ──

function collapseAllToDepth1() {
  document.querySelectorAll('#tree .collapsible').forEach(n => n.classList.add('collapsed'));
}

function expandPathToNode(nodeBox) {
  // Walk up from the node-box to the tree root, uncollapsing each ancestor .collapsible
  let el = nodeBox.closest('.node');
  while (el) {
    if (el.classList.contains('collapsible')) {
      el.classList.remove('collapsed');
    }
    const parentCol = el.parentElement; // .child-col
    if (!parentCol) break;
    const children = parentCol.parentElement; // .children
    if (!children) break;
    el = children.closest('.node');
  }
}

// ── Verification ──

async function startVerification() {
  treeAnimating = true;
  verifyAbort = new AbortController();
  document.getElementById('verifyBtn').disabled = true;
  document.getElementById('killBtn').disabled = false;
  document.getElementById('resetBtn').disabled = true;
  document.getElementById('processStatus').innerHTML = '<span class="spinner"></span>Verifying claims...';

  // Collapse everything to depth 1
  collapseAllToDepth1();
  setTimeout(drawHLines, 100);

  try {
    const resp = await fetch(`/api/articles/${articleId}/kg/verify`, {
      method: 'POST',
      signal: verifyAbort.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch { continue; }

        if (event.type === 'claim_start') {
          // Collapse tree to depth 1, then expand only the path to this leaf
          collapseAllToDepth1();
          const nodeEl = document.getElementById(`node-${event.id}`);
          if (nodeEl) {
            expandPathToNode(nodeEl);
            setTimeout(drawHLines, 50);
            nodeEl.classList.add('pending');
            nodeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
          showToast(`Verifying: ${truncate(event.text || event.id, 60)}`);

          // Highlight in markdown and fly particle
          const hl = highlightClaimInMarkdown(event.text || '', event.id);
          if (hl && nodeEl) {
            await flyFactParticle(hl, nodeEl);
          }
        }

        if (event.type === 'claim_result') {
          hideToast();
          const nodeEl = document.getElementById(`node-${event.id}`);
          if (!nodeEl) continue;

          nodeEl.classList.remove('pending');

          const noEvidence = !event.evidence_urls || event.evidence_urls.length === 0;
          let verdictType, verdictChar, nodeClass;
          if (noEvidence && event.verdict !== 'passed') {
            verdictType = 'inconclusive'; verdictChar = '\u2212'; nodeClass = 'failed';
          } else if (event.verdict === 'passed') {
            verdictType = 'pass'; verdictChar = '\u2713'; nodeClass = 'passed';
          } else {
            verdictType = 'fail'; verdictChar = '\u2717'; nodeClass = 'failed';
          }

          nodeEl.classList.add(nodeClass);

          // Mark markdown highlight
          const hlEl = document.querySelector(`.fact-highlight[data-claim-id="${event.id}"]`);
          if (hlEl) hlEl.classList.add(verdictType === 'pass' ? 'done' : 'fail-done');

          // Verdict badge
          const badge = document.createElement('span');
          badge.className = `verdict-badge ${verdictType}`;
          badge.textContent = verdictChar;
          badge.title = `${event.verdict.toUpperCase()}\n${event.reasoning || ''}`;
          nodeEl.appendChild(badge);

          // Error badges
          if (event.errors && event.errors.length) {
            addErrorBadges(nodeEl, event.errors);
          }

          // Evidence links
          if (event.evidence_urls && event.evidence_urls.length) {
            const links = el('div', 'evidence-links');
            event.evidence_urls.forEach(u => {
              const a = document.createElement('a');
              try { a.textContent = new URL(u).hostname.replace('www.', ''); } catch { a.textContent = u.substring(0, 30); }
              a.href = u; a.target = '_blank';
              links.appendChild(a);
            });
            nodeEl.appendChild(links);
          }

          // Propagate to parent
          propagateVerdict(nodeEl);
        }

        if (event.type === 'done') {
          hideToast();
          const score = event.score;
          const passed = score >= 0.5;
          const root = document.getElementById('node-root') || document.querySelector('.type-root .node-box');
          if (root) root.classList.add(passed ? 'passed' : 'failed');

          if (event.error_summary) addRootErrorSummary(event.error_summary);

          document.getElementById('processStatus').textContent =
            `Score: ${(score * 100).toFixed(1)}% \u2014 ${event.passed}/${event.total_claims} claims passed`;

          showScoreDashboard(event);
        }

        if (event.type === 'error') {
          hideToast();
          document.getElementById('processStatus').textContent = `Error: ${event.message}`;
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      document.getElementById('processStatus').textContent = 'Verification killed';
    } else {
      document.getElementById('processStatus').textContent = `Error: ${err.message}`;
    }
  }

  hideToast();
  treeAnimating = false;
  verifyAbort = null;
  document.getElementById('killBtn').disabled = true;
  document.getElementById('resetBtn').disabled = false;
}

function propagateVerdict(nodeEl) {
  // Walk up and mark parent nodes based on children
  const nodeWrapper = nodeEl.closest('.node');
  if (!nodeWrapper) return;
  const parentCol = nodeWrapper.closest('.child-col');
  if (!parentCol) return;
  const parentChildren = parentCol.closest('.children');
  if (!parentChildren) return;
  const parentNode = parentChildren.closest('.node');
  if (!parentNode) return;
  const parentBox = parentNode.querySelector(':scope > .node-box');
  if (!parentBox) return;

  // Count completed children
  const allLeaves = parentChildren.querySelectorAll('.node-box.passed, .node-box.failed');
  const totalLeaves = parentChildren.querySelectorAll('.type-leaf > .node-box, .type-check > .node-box').length;
  if (!totalLeaves) return;

  const failedCount = parentChildren.querySelectorAll('.node-box.failed').length;
  if (allLeaves.length >= totalLeaves) {
    parentBox.classList.add(failedCount === 0 ? 'passed' : 'failed');
  }
}

function killVerification() {
  if (verifyAbort) verifyAbort.abort();
}

async function resetTree() {
  if (verifyAbort) { verifyAbort.abort(); verifyAbort = null; }
  treeAnimating = false;

  // Delete cached KG from server
  try {
    await fetch(`/api/articles/${articleId}/kg`, { method: 'DELETE' });
  } catch (e) {}

  kgTreeData = null;

  // Hide tree and markdown panel, show build prompt
  document.getElementById('tree-section').style.display = 'none';
  document.getElementById('md-panel').style.display = 'none';
  document.getElementById('scoreDashboard').style.display = 'none';
  document.getElementById('tree').innerHTML = '';
  document.querySelectorAll('.fact-particle').forEach(el => el.remove());
  document.getElementById('kg-container').style.display = '';
  showBuildPrompt();
}

// ── Error Badges ──

function addErrorBadges(nodeEl, errors) {
  nodeEl.querySelectorAll('.error-badges').forEach(el => el.remove());
  const container = document.createElement('div');
  container.className = 'error-badges';
  ERROR_TYPES.forEach(et => {
    const badge = document.createElement('span');
    const active = errors.includes(et.id);
    badge.className = `error-badge ${active ? 'active' : 'inactive'}`;
    badge.textContent = et.label;
    badge.title = `${et.name}: ${et.desc}${active ? ' [DETECTED]' : ''}`;
    container.appendChild(badge);
  });
  nodeEl.appendChild(container);
}

function addRootErrorSummary(errorSummary) {
  const root = document.getElementById('node-root') || document.querySelector('.type-root .node-box');
  if (!root) return;
  root.querySelectorAll('.error-summary').forEach(el => el.remove());
  const container = document.createElement('div');
  container.className = 'error-summary';
  ERROR_TYPES.forEach(et => {
    const count = errorSummary[et.id] || 0;
    const badge = document.createElement('span');
    badge.className = `error-badge ${count > 0 ? 'active' : 'inactive'}`;
    badge.textContent = `${et.label}:${count}`;
    badge.title = `${et.name}: ${et.desc}\n${count} claim${count !== 1 ? 's' : ''} affected`;
    container.appendChild(badge);
  });
  root.appendChild(container);
}

// ── Score Dashboard ──

function showScoreDashboard(event) {
  const dashboard = document.getElementById('scoreDashboard');
  const score = event.score;
  const cls = score >= 0.7 ? 'score-good' : score >= 0.4 ? 'score-mid' : 'score-bad';
  document.getElementById('scoreCards').innerHTML = `
    <div class="score-card"><div class="value ${cls}">${(score * 100).toFixed(1)}%</div><div class="label-text">Overall Score</div></div>
    <div class="score-card"><div class="value score-good">${event.passed}</div><div class="label-text">Passed</div></div>
    <div class="score-card"><div class="value score-bad">${event.failed}</div><div class="label-text">Failed</div></div>
    <div class="score-card"><div class="value">${event.total_claims}</div><div class="label-text">Total Claims</div></div>
  `;
  dashboard.style.display = '';
}

// ── Markdown Claim Highlighting ──

function highlightClaimInMarkdown(claimText, claimId) {
  const mdEl = document.querySelector('#md-body .rendered-md');
  if (!mdEl) return null;

  const words = claimText.split(/\s+/).filter(w => w.length > 4).slice(0, 5);
  if (!words.length) return null;

  const walker = document.createTreeWalker(mdEl, NodeFilter.SHOW_TEXT);
  let bestNode = null, bestScore = 0;
  while (walker.nextNode()) {
    const text = walker.currentNode.textContent.toLowerCase();
    const score = words.filter(w => text.includes(w.toLowerCase())).length;
    if (score > bestScore) { bestScore = score; bestNode = walker.currentNode; }
  }
  if (!bestNode || bestScore < 1) return null;

  const range = document.createRange();
  range.selectNodeContents(bestNode);
  const hl = document.createElement('span');
  hl.className = 'fact-highlight';
  hl.dataset.claimId = claimId;
  range.surroundContents(hl);

  // Scroll markdown panel to highlight
  const container = hl.closest('.md-panel-body');
  if (container) {
    const cRect = container.getBoundingClientRect();
    const hRect = hl.getBoundingClientRect();
    container.scrollTop += hRect.top - cRect.top - cRect.height / 3;
  }
  return hl;
}

function flyFactParticle(sourceEl, targetEl) {
  return new Promise(resolve => {
    const sr = sourceEl.getBoundingClientRect();
    const tr = targetEl.getBoundingClientRect();
    const p = document.createElement('div');
    p.className = 'fact-particle';
    p.textContent = 'Verify';
    p.style.left = (sr.left + sr.width / 2) + 'px';
    p.style.top = (sr.top + sr.height / 2) + 'px';
    document.body.appendChild(p);
    requestAnimationFrame(() => {
      p.style.left = (tr.left + tr.width / 2 - p.offsetWidth / 2) + 'px';
      p.style.top = (tr.top + tr.height / 2 - p.offsetHeight / 2) + 'px';
    });
    setTimeout(() => {
      targetEl.classList.add('receiving');
      setTimeout(() => {
        targetEl.classList.remove('receiving');
        p.style.opacity = '0';
        setTimeout(() => { p.remove(); resolve(); }, 200);
      }, 350);
    }, 800);
  });
}

// ── Toast ──

let toastEl = null;
function showToast(msg) {
  hideToast();
  toastEl = document.createElement('div');
  toastEl.className = 'browser-toast';
  toastEl.innerHTML = `<div class="toast-dot"></div>${esc(msg)}`;
  document.body.appendChild(toastEl);
}
function hideToast() {
  if (!toastEl) return;
  toastEl.classList.add('hiding');
  const e = toastEl;
  toastEl = null;
  setTimeout(() => e.remove(), 300);
}

// ── Legend hover ──

document.addEventListener('mouseover', e => {
  const item = e.target.closest('.legend-item[data-highlight]');
  if (!item) return;
  document.querySelectorAll(`.type-${item.dataset.highlight} > .node-box`).forEach(box => {
    box.style.boxShadow = '0 0 18px rgba(93,173,226,0.7)';
  });
});
document.addEventListener('mouseout', e => {
  const item = e.target.closest('.legend-item[data-highlight]');
  if (!item) return;
  document.querySelectorAll(`.type-${item.dataset.highlight} > .node-box`).forEach(box => {
    box.style.boxShadow = '';
  });
});

// ── Helpers ──

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n - 1) + '\u2026' : s;
}
