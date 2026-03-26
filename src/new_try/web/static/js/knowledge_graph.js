const params = new URLSearchParams(window.location.search);
const articleId = params.get('id');

document.addEventListener('DOMContentLoaded', async () => {
    if (!articleId) return;
    document.getElementById('back-to-article').href = `/static/article.html?id=${articleId}`;

    // Try cached KG first
    try {
        const res = await fetch(`/api/articles/${articleId}/kg`);
        if (res.ok) {
            const tree = await res.json();
            document.getElementById('kg-title').textContent = tree.label || 'Knowledge Graph';
            showGraph(tree);
            return;
        }
    } catch (e) {}

    // No cached KG — show build button
    showBuildPrompt();
});

function showBuildPrompt() {
    const container = document.getElementById('kg-container');
    container.innerHTML = `
        <div class="flex flex-col items-center justify-center h-full gap-4">
            <p class="text-slate-500 text-sm">No knowledge graph built for this article yet.</p>
            <button id="build-btn" class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg text-sm font-medium transition">
                Build Knowledge Graph
            </button>
            <p id="build-status" class="text-xs text-slate-400 hidden"></p>
            <div id="build-nodes" class="text-xs text-slate-400 max-h-48 overflow-y-auto w-full max-w-md px-4 hidden"></div>
        </div>`;
    document.getElementById('build-btn').addEventListener('click', startBuild);
}

function startBuild() {
    const btn = document.getElementById('build-btn');
    btn.disabled = true;
    btn.textContent = 'Building...';
    btn.classList.add('opacity-50');

    const status = document.getElementById('build-status');
    const nodesDiv = document.getElementById('build-nodes');
    status.classList.remove('hidden');
    nodesDiv.classList.remove('hidden');
    status.textContent = 'Discovering categories...';

    let nodeCount = 0;
    const evtSource = new EventSource(`/api/articles/${articleId}/kg/build`);

    // EventSource only handles GET; we need POST. Use fetch + reader instead.
    evtSource.close();

    fetch(`/api/articles/${articleId}/kg/build`, { method: 'POST' })
        .then(res => {
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            function read() {
                reader.read().then(({ done, value }) => {
                    if (done) return;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        try {
                            const event = JSON.parse(line.slice(6));
                            handleBuildEvent(event, status, nodesDiv, () => nodeCount++);
                        } catch (e) {}
                    }
                    read();
                });
            }
            read();
        })
        .catch(err => {
            status.textContent = 'Error: ' + err.message;
            btn.disabled = false;
            btn.textContent = 'Retry';
            btn.classList.remove('opacity-50');
        });
}

function handleBuildEvent(event, status, nodesDiv, incCount) {
    if (event.type === 'node') {
        incCount();
        const d = event.data;
        const indent = '  '.repeat(d.depth);
        const icons = { root: '\u25c6', category: '\u25c7', entity: '\u25cf', claim: '\u25ce' };
        const icon = icons[d.node_type] || '\u00b7';
        nodesDiv.innerHTML += `<div>${indent}${icon} ${d.label}</div>`;
        nodesDiv.scrollTop = nodesDiv.scrollHeight;
        status.textContent = `Building... (${nodesDiv.children.length} nodes)`;
    } else if (event.type === 'done') {
        status.textContent = 'Done!';
        document.getElementById('kg-title').textContent = event.data.label || 'Knowledge Graph';
        setTimeout(() => showGraph(event.data), 500);
    } else if (event.type === 'error') {
        status.textContent = 'Error: ' + (event.data.message || 'Unknown error');
    }
}

// ─── D3 Radial Tree Visualization ────────────────────────────────────

function showGraph(tree) {
    const container = document.getElementById('kg-container');
    container.innerHTML = '';

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Flatten tree to d3 hierarchy
    const root = d3.hierarchy(tree, d => d.children);
    const totalNodes = root.descendants().length;

    // Use radial tree layout
    const radius = Math.min(width, height) / 2 - 80;
    const treeLayout = d3.tree()
        .size([2 * Math.PI, Math.min(radius, totalNodes * 12)])
        .separation((a, b) => (a.parent === b.parent ? 1 : 2) / a.depth || 1);
    treeLayout(root);

    const svg = d3.select(container).append('svg')
        .attr('width', width).attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`);

    // Zoom + pan
    svg.call(d3.zoom().scaleExtent([0.2, 4]).on('zoom', (e) => {
        g.attr('transform', `translate(${width / 2 + e.transform.x},${height / 2 + e.transform.y}) scale(${e.transform.k})`);
    }));

    const colors = { root: '#3b82f6', category: '#10b981', entity: '#8b5cf6', claim: '#f59e0b' };
    const radii = { root: 10, category: 6, entity: 5, claim: 4 };

    // Links
    g.selectAll('.link').data(root.links()).join('path')
        .attr('class', 'link')
        .attr('fill', 'none')
        .attr('stroke', '#cbd5e1')
        .attr('stroke-width', 1)
        .attr('d', d3.linkRadial().angle(d => d.x).radius(d => d.y));

    // Nodes
    const node = g.selectAll('.node').data(root.descendants()).join('g')
        .attr('class', 'node')
        .attr('transform', d => `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0)`);

    node.append('circle')
        .attr('r', d => radii[d.data.type] || 4)
        .attr('fill', d => colors[d.data.type] || '#94a3b8')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .style('cursor', 'pointer')
        .on('click', (e, d) => showNodeDetail(d.data));

    // Labels (only for depth <= 2 to avoid clutter)
    node.filter(d => d.depth <= 2)
        .append('text')
        .attr('dy', '0.31em')
        .attr('x', d => d.x < Math.PI === !d.children ? 8 : -8)
        .attr('text-anchor', d => d.x < Math.PI === !d.children ? 'start' : 'end')
        .attr('transform', d => d.x >= Math.PI ? 'rotate(180)' : null)
        .attr('font-size', d => d.depth === 0 ? '13px' : '10px')
        .attr('fill', '#334155')
        .text(d => {
            const l = d.data.label || '';
            return l.length > 35 ? l.slice(0, 33) + '...' : l;
        });

    // Tooltip on hover for deeper nodes
    node.filter(d => d.depth > 2)
        .append('title')
        .text(d => d.data.label);

    // Legend
    const legend = svg.append('g').attr('transform', `translate(16, ${height - 100})`);
    [['Root', '#3b82f6'], ['Category', '#10b981'], ['Entity', '#8b5cf6'], ['Claim', '#f59e0b']].forEach(([label, color], i) => {
        legend.append('circle').attr('cx', 0).attr('cy', i * 20).attr('r', 5).attr('fill', color);
        legend.append('text').attr('x', 12).attr('y', i * 20 + 4).text(label)
            .attr('font-size', '11px').attr('fill', '#64748b');
    });

    // Stats bar
    const leaves = root.leaves().length;
    const maxDepth = d3.max(root.descendants(), d => d.depth);
    const statsDiv = document.createElement('div');
    statsDiv.className = 'absolute top-3 right-3 text-xs text-slate-500 bg-white/80 px-3 py-1.5 rounded-lg';
    statsDiv.textContent = `${totalNodes} nodes \u00b7 ${leaves} claims \u00b7 depth ${maxDepth}`;
    container.style.position = 'relative';
    container.appendChild(statsDiv);
}

// ─── Node Detail Panel ───────────────────────────────────────────────

function showNodeDetail(data) {
    let panel = document.getElementById('node-detail');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'node-detail';
        panel.className = 'absolute top-3 left-3 bg-white border border-gray-200 rounded-xl shadow-lg p-4 max-w-sm max-h-[60vh] overflow-y-auto text-sm';
        document.getElementById('kg-container').appendChild(panel);
    }
    panel.classList.remove('hidden');

    const verdictColors = { passed: 'text-green-600', failed: 'text-red-600', partial: 'text-amber-600', unverified: 'text-slate-400' };
    const verdictLabel = data.verdict || 'unverified';
    const score = data.aggregated_score != null ? ` (${Math.round(data.aggregated_score * 100)}%)` : '';

    let html = `
        <div class="flex justify-between items-start mb-2">
            <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">${data.type || 'node'}</span>
            <button onclick="document.getElementById('node-detail').classList.add('hidden')" class="text-slate-400 hover:text-slate-600 text-lg leading-none">&times;</button>
        </div>
        <h4 class="font-semibold text-slate-800 mb-2">${data.label || ''}</h4>`;

    if (data.claim_text) {
        html += `<p class="text-slate-600 mb-2">${data.claim_text}</p>`;
    }
    if (data.source_snippet) {
        html += `<p class="text-xs text-slate-400 italic mb-2">"${data.source_snippet}"</p>`;
    }
    if (verdictLabel !== 'unverified' || score) {
        html += `<p class="${verdictColors[verdictLabel] || ''} font-medium">${verdictLabel}${score}</p>`;
    }
    if (data.cited_urls && data.cited_urls.length) {
        html += `<div class="mt-2 space-y-1">`;
        data.cited_urls.forEach(u => {
            html += `<a href="${u}" target="_blank" class="block text-xs text-blue-500 hover:underline truncate">${u}</a>`;
        });
        html += `</div>`;
    }
    if (data.source_section) {
        html += `<p class="mt-2 text-xs text-slate-400">Section: ${data.source_section}</p>`;
    }

    panel.innerHTML = html;
}
