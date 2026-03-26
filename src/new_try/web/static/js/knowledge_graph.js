const params = new URLSearchParams(window.location.search);
const articleId = params.get('id');

document.addEventListener('DOMContentLoaded', async () => {
    if (!articleId) return;

    document.getElementById('back-to-article').href = `/static/article.html?id=${articleId}`;

    try {
        const data = await api(`/api/articles/${articleId}`);
        document.getElementById('kg-title').textContent = data.topic;
        buildGraph(data);
    } catch (e) {
        document.getElementById('kg-container').innerHTML =
            '<p class="text-center text-red-500 pt-12">Failed to load article data.</p>';
    }
});

function buildGraph(data) {
    const container = document.getElementById('kg-container');
    container.innerHTML = '';
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Parse article sections from markdown headings
    const lines = (data.article_text || '').split('\n');
    const sections = [];
    lines.forEach(line => {
        const m = line.match(/^#{1,3}\s+(.+)/);
        if (m) sections.push(m[1].trim());
    });

    // Build nodes: topic center + sections + references
    const nodes = [{ id: 'topic', label: data.topic, type: 'topic' }];
    const links = [];

    sections.forEach((s, i) => {
        const id = `sec-${i}`;
        nodes.push({ id, label: s, type: 'section' });
        links.push({ source: 'topic', target: id });
    });

    // Add reference nodes (limit to 15 to keep it readable)
    const refs = Object.entries(data.citation_dict || {}).slice(0, 15);
    refs.forEach(([idx, ref]) => {
        const id = `ref-${idx}`;
        const label = ref.title || ref.url;
        nodes.push({ id, label, type: 'reference', url: ref.url });
        // Link reference to the topic
        links.push({ source: 'topic', target: id });
    });

    if (nodes.length <= 1) {
        container.innerHTML = '<p class="text-center text-slate-400 pt-12">No graph data available for this article.</p>';
        return;
    }

    const svg = d3.select(container).append('svg')
        .attr('width', width).attr('height', height);

    const g = svg.append('g');

    // Zoom
    svg.call(d3.zoom().scaleExtent([0.3, 3]).on('zoom', (e) => g.attr('transform', e.transform)));

    const colors = { topic: '#3b82f6', section: '#10b981', reference: '#f59e0b' };
    const radius = { topic: 24, section: 14, reference: 10 };

    const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(120))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => radius[d.type] + 10));

    const link = g.append('g').selectAll('line').data(links).join('line')
        .attr('stroke', '#cbd5e1').attr('stroke-width', 1.5);

    const node = g.append('g').selectAll('g').data(nodes).join('g')
        .call(d3.drag()
            .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
        );

    node.append('circle')
        .attr('r', d => radius[d.type])
        .attr('fill', d => colors[d.type])
        .attr('stroke', '#fff').attr('stroke-width', 2)
        .style('cursor', d => d.url ? 'pointer' : 'default')
        .on('click', (e, d) => { if (d.url) window.open(d.url, '_blank'); });

    node.append('text')
        .text(d => d.label.length > 30 ? d.label.slice(0, 28) + '...' : d.label)
        .attr('dx', d => radius[d.type] + 6).attr('dy', 4)
        .attr('font-size', d => d.type === 'topic' ? '13px' : '11px')
        .attr('fill', '#334155');

    // Legend
    const legend = svg.append('g').attr('transform', `translate(16, ${height - 80})`);
    [['Topic', '#3b82f6'], ['Section', '#10b981'], ['Reference', '#f59e0b']].forEach(([label, color], i) => {
        legend.append('circle').attr('cx', 0).attr('cy', i * 22).attr('r', 6).attr('fill', color);
        legend.append('text').attr('x', 14).attr('y', i * 22 + 4).text(label)
            .attr('font-size', '11px').attr('fill', '#64748b');
    });

    sim.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}
