const params = new URLSearchParams(window.location.search);
const runId = params.get('run_id');
const articleId = params.get('id');

let conversationLog = [];

document.addEventListener('DOMContentLoaded', () => {
    if (runId) {
        startProgress();
    } else if (articleId) {
        loadArticle(articleId);
    }

    document.getElementById('brainstorm-btn').addEventListener('click', () => {
        document.getElementById('brainstorm-modal').classList.remove('hidden');
    });
    document.getElementById('modal-close').addEventListener('click', () => {
        document.getElementById('brainstorm-modal').classList.add('hidden');
    });
    document.getElementById('brainstorm-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) {
            document.getElementById('brainstorm-modal').classList.add('hidden');
        }
    });
});

// --- Progress mode ---
const SPINNER_SVG = '<svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>';
const CHECK_SVG = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>';

function setStepState(stepName, state, detail) {
    const step = document.querySelector(`[data-step="${stepName}"]`);
    if (!step) return;
    const icon = step.querySelector('.step-icon');
    const detailEl = step.querySelector('.step-detail');

    icon.className = 'step-icon mt-0.5 ' + (state === 'active' ? 'step-active' : state === 'done' ? 'step-done' : 'step-pending');
    if (state === 'active') icon.innerHTML = SPINNER_SVG;
    else if (state === 'done') icon.innerHTML = CHECK_SVG;

    if (detail) detailEl.textContent = detail;
}

async function startProgress() {
    const progressView = document.getElementById('progress-view');
    progressView.classList.remove('hidden');

    // Fetch topic
    try {
        const status = await api(`/api/run/${runId}/status`);
        document.getElementById('progress-topic').textContent = status.topic;
        document.title = `STORM - ${status.topic}`;
    } catch (e) {}

    // SSE
    const evtSource = new EventSource(`/api/run/${runId}/events`);
    evtSource.onmessage = (e) => {
        const event = JSON.parse(e.data);
        handleEvent(event, evtSource);
    };
    evtSource.onerror = () => {
        evtSource.close();
    };
}

function handleEvent(event, evtSource) {
    const t = event.type;
    const d = event.data;

    switch (t) {
        case 'perspective_start':
            setStepState('perspective', 'active');
            break;
        case 'perspective_end':
            setStepState('perspective', 'done', `${d.perspectives?.length || 0} perspectives identified`);
            break;
        case 'gathering_start':
            setStepState('gathering', 'active');
            break;
        case 'dialogue_turn':
            setStepState('gathering', 'active', `Searched ${d.total_sources || 0} sources`);
            break;
        case 'gathering_end':
            setStepState('gathering', 'done', `Searched ${d.total_sources || 0} sources`);
            break;
        case 'organization_start':
            setStepState('outline', 'active');
            break;
        case 'outline_draft':
            setStepState('outline', 'active', 'Draft outline generated');
            break;
        case 'outline_refined':
            setStepState('outline', 'done', 'Outline refined');
            break;
        case 'writing_start':
            setStepState('writing', 'active');
            break;
        case 'polishing_start':
            setStepState('writing', 'done');
            setStepState('polishing', 'active');
            break;
        case 'done':
            setStepState('polishing', 'done');
            evtSource.close();
            if (d.article_dir) {
                setTimeout(() => loadArticle(d.article_dir), 500);
            }
            break;
        case 'error':
            evtSource.close();
            document.getElementById('error-msg').textContent = d.message || 'An error occurred';
            document.getElementById('error-msg').classList.remove('hidden');
            break;
    }
}

// --- Article mode ---
async function loadArticle(id) {
    document.getElementById('progress-view').classList.add('hidden');
    document.getElementById('article-view').classList.remove('hidden');

    try {
        const data = await api(`/api/articles/${encodeURIComponent(id)}`);
        document.getElementById('article-title').textContent = data.topic;
        document.title = `STORM - ${data.topic}`;

        // Render article
        const body = document.getElementById('article-body');
        body.innerHTML = data.article_html;

        // Build TOC from article headings
        buildTOC(body);

        // References
        if (data.references && Object.keys(data.references).length > 0) {
            renderReferences(data.references);
        }

        // Conversation log for brainstorming
        if (data.conversation_log && data.conversation_log.length > 0) {
            conversationLog = data.conversation_log;
            buildBrainstormModal(data.conversation_log);
        } else {
            document.getElementById('brainstorm-btn').classList.add('hidden');
        }
    } catch (e) {
        document.getElementById('article-body').innerHTML = `<p class="text-red-500">Failed to load article: ${e.message}</p>`;
    }
}

function buildTOC(container) {
    const headings = container.querySelectorAll('h1, h2, h3');
    const tocList = document.getElementById('toc-list');
    if (headings.length === 0) return;

    document.getElementById('toc').classList.remove('hidden');
    tocList.innerHTML = '';

    headings.forEach((h, i) => {
        const id = `heading-${i}`;
        h.id = id;
        const level = parseInt(h.tagName[1]);
        const pl = level === 1 ? '' : level === 2 ? 'pl-3' : 'pl-6';
        const a = document.createElement('a');
        a.href = `#${id}`;
        a.className = `block text-sm text-slate-500 hover:text-blue-600 py-0.5 ${pl} truncate`;
        a.textContent = h.textContent;
        tocList.appendChild(a);
    });
}

function renderReferences(refs) {
    const section = document.getElementById('references-section');
    const list = document.getElementById('references-list');
    section.classList.remove('hidden');

    const entries = Object.entries(refs);
    list.innerHTML = entries.map(([url, info], i) => {
        const title = info.title || info.snippet?.slice(0, 80) || url;
        return `<div class="flex gap-2" id="ref-${i + 1}">
            <span class="text-slate-400 shrink-0">[${i + 1}]</span>
            <a href="${escapeAttr(url)}" target="_blank" class="text-blue-600 hover:underline break-all">${escapeHtml(title)}</a>
        </div>`;
    }).join('');
}

function buildBrainstormModal(log) {
    const tabs = document.getElementById('persona-tabs');
    const content = document.getElementById('persona-content');

    tabs.innerHTML = log.map((conv, i) => {
        const name = conv.perspective || conv.persona || `Persona ${i + 1}`;
        return `<button class="persona-tab px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap ${i === 0 ? 'active' : ''}" data-index="${i}">${escapeHtml(name)}</button>`;
    }).join('');

    // Show first persona
    showPersona(0);

    tabs.addEventListener('click', (e) => {
        const btn = e.target.closest('.persona-tab');
        if (!btn) return;
        tabs.querySelectorAll('.persona-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        showPersona(parseInt(btn.dataset.index));
    });
}

function showPersona(index) {
    const content = document.getElementById('persona-content');
    const conv = conversationLog[index];
    if (!conv) return;

    const description = conv.perspective || conv.persona || '';
    const turns = conv.dlg_turns || conv.dialogue_turns || [];

    let html = '';
    if (description) {
        html += `<div class="bg-blue-50 border border-blue-100 rounded-lg p-3 text-sm text-blue-800">${escapeHtml(description)}</div>`;
    }

    turns.forEach(turn => {
        const queries = turn.queries || turn.search_queries || [];
        const utterance = turn.utterance || turn.content || '';
        const role = turn.role || 'user';

        if (role === 'user' || turn.agent_utterance === undefined) {
            html += `<div class="flex gap-3 justify-end">
                <div class="chat-bubble-user px-4 py-2.5 max-w-lg text-sm">${escapeHtml(utterance)}</div>
                <div class="w-8 h-8 bg-slate-200 rounded-full flex items-center justify-center shrink-0">
                    <svg class="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
                </div>
            </div>`;
        }
        if (turn.agent_utterance !== undefined) {
            const agentText = turn.agent_utterance || '';
            html += `<div class="flex gap-3">
                <div class="w-8 h-8 bg-slate-800 rounded-full flex items-center justify-center shrink-0">
                    <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                </div>
                <div class="chat-bubble-agent px-4 py-2.5 max-w-lg text-sm">${escapeHtml(agentText)}</div>
            </div>`;
        }
    });

    content.innerHTML = html || '<p class="text-slate-400 text-center py-8">No conversation data available</p>';
}

function escapeHtml(s) {
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function escapeAttr(s) {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
