document.addEventListener('DOMContentLoaded', async () => {
    const mdInput = document.getElementById('md-input');
    const titleInput = document.getElementById('title-input');
    const toc = document.getElementById('toc');
    const saveBtn = document.getElementById('save-btn');
    const toggleBtn = document.getElementById('toggle-preview');
    const preview = document.getElementById('md-preview');
    let showingPreview = false;

    const editId = new URLSearchParams(window.location.search).get('edit');

    // If editing, load existing article
    if (editId) {
        try {
            const data = await api(`/api/articles/${encodeURIComponent(editId)}`);
            titleInput.value = data.topic || '';
            mdInput.value = data.article_text || '';
            updateToc();
        } catch (e) {
            alert('Failed to load article: ' + e.message);
        }
    }

    // Dynamic TOC from headings
    mdInput.addEventListener('input', updateToc);

    function updateToc() {
        const lines = mdInput.value.split('\n');
        const headings = [];
        for (const line of lines) {
            const m = line.match(/^(#{1,6})\s+(.+)/);
            if (m) headings.push({ level: m[1].length, text: m[2].trim() });
        }
        if (!headings.length) {
            toc.innerHTML = '<p class="text-xs text-slate-300 italic">Headings will appear here...</p>';
            return;
        }
        // Auto-detect title from first h1 (only if title is empty)
        if (!titleInput.value && headings[0].level === 1) {
            titleInput.value = headings[0].text;
        }
        toc.innerHTML = headings.map(h => {
            const indent = (h.level - 1) * 12;
            const weight = h.level <= 2 ? 'font-medium text-slate-700' : 'text-slate-500';
            const size = h.level === 1 ? 'text-sm' : 'text-xs';
            return `<div class="${size} ${weight} py-0.5 truncate" style="padding-left:${indent}px" title="${h.text}">${h.text}</div>`;
        }).join('');
    }

    // Preview toggle
    toggleBtn.addEventListener('click', () => {
        showingPreview = !showingPreview;
        if (showingPreview) {
            mdInput.classList.add('hidden');
            preview.classList.remove('hidden');
            preview.innerHTML = renderMarkdown(mdInput.value);
            toggleBtn.textContent = 'Edit Markdown';
        } else {
            mdInput.classList.remove('hidden');
            preview.classList.add('hidden');
            toggleBtn.textContent = 'Show Preview';
        }
    });

    function renderMarkdown(md) {
        let html = md
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/^######\s+(.+)$/gm, '<h6 class="text-xs font-semibold text-slate-600 mt-3 mb-1">$1</h6>')
            .replace(/^#####\s+(.+)$/gm, '<h5 class="text-sm font-semibold text-slate-600 mt-3 mb-1">$1</h5>')
            .replace(/^####\s+(.+)$/gm, '<h4 class="text-sm font-semibold text-slate-700 mt-4 mb-1">$1</h4>')
            .replace(/^###\s+(.+)$/gm, '<h3 class="text-base font-semibold text-slate-700 mt-4 mb-2">$1</h3>')
            .replace(/^##\s+(.+)$/gm, '<h2 class="text-lg font-bold text-slate-800 mt-5 mb-2">$1</h2>')
            .replace(/^#\s+(.+)$/gm, '<h1 class="text-xl font-bold text-slate-800 mt-6 mb-3">$1</h1>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code class="bg-slate-100 px-1 rounded text-xs">$1</code>')
            .replace(/^\- (.+)$/gm, '<li class="ml-4 list-disc text-sm text-slate-600">$1</li>')
            .replace(/^\d+\.\s+(.+)$/gm, '<li class="ml-4 list-decimal text-sm text-slate-600">$1</li>')
            .replace(/\n\n/g, '</p><p class="text-sm text-slate-600 mb-2">');
        return '<p class="text-sm text-slate-600 mb-2">' + html + '</p>';
    }

    // Save
    saveBtn.addEventListener('click', async () => {
        const markdown = mdInput.value.trim();
        const title = titleInput.value.trim();
        if (!markdown) { alert('Please paste some markdown first.'); return; }
        if (!title) { alert('Please enter a title.'); return; }

        saveBtn.disabled = true;
        saveBtn.textContent = 'Saving...';
        try {
            if (editId) {
                // Update existing article
                await api(`/api/articles/${encodeURIComponent(editId)}`, {
                    method: 'PUT',
                    body: JSON.stringify({ markdown }),
                });
                window.location.href = `/static/article.html?id=${editId}`;
            } else {
                // Create new article
                const data = await api('/api/articles/import', {
                    method: 'POST',
                    body: JSON.stringify({ title, markdown }),
                });
                window.location.href = `/static/article.html?id=${data.id}`;
            }
        } catch (err) {
            alert('Failed to save: ' + err.message);
            saveBtn.disabled = false;
            saveBtn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg> Save Article';
        }
    });
});

// Deep Research with claude.ai
async function startDeepResearch() {
    const title = document.getElementById('title-input').value.trim();
    if (!title) { alert('Enter a topic/title first.'); return; }

    const btn = document.getElementById('deep-research-btn');
    const statusEl = document.getElementById('deep-research-status');
    const msgEl = document.getElementById('deep-research-msg');

    btn.disabled = true;
    btn.textContent = 'Researching...';
    statusEl.classList.remove('hidden');
    msgEl.textContent = 'Starting deep research with Claude...';

    try {
        const res = await api('/api/deep-research', {
            method: 'POST',
            body: JSON.stringify({ topic: title }),
        });

        // Stream status events
        const evtSource = new EventSource(`/api/run/${res.run_id}/events`);

        evtSource.onmessage = (e) => {
            const event = JSON.parse(e.data);

            if (event.type === 'status') {
                msgEl.textContent = event.data.message;
            }

            if (event.type === 'done') {
                evtSource.close();
                msgEl.textContent = `Done! (${event.data.chars} chars)`;
                statusEl.classList.remove('bg-purple-50', 'border-purple-200', 'text-purple-800');
                statusEl.classList.add('bg-green-50', 'border-green-200', 'text-green-800');

                // Redirect to the article
                setTimeout(() => {
                    window.location.href = `/static/article.html?id=${event.data.article_id}`;
                }, 1500);
            }

            if (event.type === 'error') {
                evtSource.close();
                msgEl.textContent = `Error: ${event.data.message}`;
                statusEl.classList.remove('bg-purple-50', 'border-purple-200', 'text-purple-800');
                statusEl.classList.add('bg-red-50', 'border-red-200', 'text-red-800');
                btn.disabled = false;
                btn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg> Deep Research';
            }
        };

        evtSource.onerror = () => {
            evtSource.close();
            msgEl.textContent = 'Connection lost';
            btn.disabled = false;
            btn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg> Deep Research';
        };
    } catch (err) {
        msgEl.textContent = `Failed: ${err.message}`;
        btn.disabled = false;
        btn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg> Deep Research';
    }
}
