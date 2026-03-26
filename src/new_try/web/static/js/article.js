const params = new URLSearchParams(window.location.search);
const runId = params.get('run_id');
const articleId = params.get('id');

let conversationLog = [];
let citationDict = {};

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

    document.getElementById('refs-btn').addEventListener('click', () => {
        document.getElementById('refs-modal').classList.remove('hidden');
    });
    document.getElementById('refs-modal-close').addEventListener('click', () => {
        document.getElementById('refs-modal').classList.add('hidden');
    });
    document.getElementById('refs-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) {
            document.getElementById('refs-modal').classList.add('hidden');
        }
    });

    // Knowledge Graph button
    const kgBtn = document.getElementById('btn-knowledge-graph');
    if (kgBtn) {
        kgBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const id = articleId || params.get('run_id');
            if (id) window.location.href = `/static/knowledge_graph.html?id=${id}`;
        });
    }
});

// --- Progress mode ---
const SPINNER_SVG = '<svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>';
const CHECK_SVG = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>';
const ERROR_SVG = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>';

function setStepState(stepName, state, detail) {
    const step = document.querySelector(`[data-step="${stepName}"]`);
    if (!step) return;
    const icon = step.querySelector('.step-icon');
    const detailEl = step.querySelector('.step-detail');

    icon.className = 'step-icon mt-0.5 ' + (state === 'active' ? 'step-active' : state === 'done' ? 'step-done' : state === 'error' ? 'step-error' : 'step-pending');
    if (state === 'active') icon.innerHTML = SPINNER_SVG;
    else if (state === 'done') icon.innerHTML = CHECK_SVG;
    else if (state === 'error') icon.innerHTML = ERROR_SVG;

    if (detail) detailEl.textContent = detail;
}

async function startProgress() {
    const progressView = document.getElementById('progress-view');
    progressView.classList.remove('hidden');

    try {
        const status = await api(`/api/run/${runId}/status`);
        document.getElementById('progress-topic').textContent = status.topic;
        document.title = `STORM - ${status.topic}`;
    } catch (e) {}

    const evtSource = new EventSource(`/api/run/${runId}/events`);
    evtSource.onmessage = (e) => {
        const event = JSON.parse(e.data);
        handleEvent(event, evtSource);
    };
    evtSource.onerror = () => {
        evtSource.close();
    };
}

let isSerper = false;

function gatheringSummary(d) {
    if (d.is_serper) isSerper = true;
    let text = `Searched ${d.total_sources || 0} sources`;
    if (isSerper) {
        const cached = d.total_cached || 0;
        const newQ = d.total_new || 0;
        if (cached || newQ) {
            text += ` | ${d.total_queries || 0} queries, ${cached} cached, ${newQ} new`;
            if (newQ > 0) {
                const cost = (newQ * 0.001).toFixed(3);
                text += ` - Cost $${cost}`;
            }
        }
    }
    return text;
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
            if (d.is_serper) isSerper = true;
            break;
        case 'dialogue_turn':
            setStepState('gathering', 'active', gatheringSummary(d));
            if (d.query_details && d.query_details.length > 0) {
                showQueryGroup(d.query_details, d.is_serper);
            }
            break;
        case 'gathering_end':
            setStepState('gathering', 'done', gatheringSummary(d));
            if (d.failed_urls && Object.keys(d.failed_urls).length > 0) {
                showFailedUrls(d.failed_urls);
            }
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
            // Mark the currently active step as errored
            document.querySelectorAll('.step').forEach(step => {
                if (step.querySelector('.step-active')) {
                    const name = step.dataset.step;
                    setStepState(name, 'error');
                }
            });
            document.getElementById('error-msg').textContent = d.message || 'An error occurred';
            document.getElementById('error-msg').classList.remove('hidden');
            break;
    }
}

const CLOUD_SVG = '<svg class="w-3 h-3 inline-block text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"/></svg>';

function showQueryGroup(queryDetails, isSerperProvider) {
    const step = document.querySelector('[data-step="gathering"]');
    if (!step) return;
    let container = step.querySelector('.url-list');
    if (!container) {
        container = document.createElement('div');
        container.className = 'url-list mt-1 space-y-2';
        step.querySelector('div').appendChild(container);
    }
    queryDetails.forEach(qd => {
        const group = document.createElement('div');
        group.className = 'text-xs';
        const cacheIcon = (isSerperProvider && qd.cached) ? ` ${CLOUD_SVG}` : '';
        const cacheLabel = (isSerperProvider && qd.cached) ? ' <span class="text-blue-400">cached</span>' : '';
        let html = `<div class="text-slate-500 font-medium">${escapeHtml(qd.query)}${cacheIcon}${cacheLabel}</div>`;
        (qd.urls || []).forEach(url => {
            const short = url.length > 70 ? url.slice(0, 70) + '...' : url;
            html += `<div class="text-slate-400 pl-3 truncate"><a href="${escapeAttr(url)}" target="_blank" class="hover:underline">${escapeHtml(short)}</a></div>`;
        });
        group.innerHTML = html;
        container.appendChild(group);
    });
}

function showFailedUrls(failedUrls) {
    const step = document.querySelector('[data-step="gathering"]');
    if (!step) return;
    const parent = step.querySelector('div');
    Object.entries(failedUrls).forEach(([url, reason]) => {
        const el = document.createElement('div');
        el.className = 'text-xs mt-1 px-2 py-1 rounded bg-amber-50 border border-amber-200 text-amber-700';
        const shortUrl = url.length > 70 ? url.slice(0, 70) + '...' : url;
        el.innerHTML = `<a href="${escapeAttr(url)}" target="_blank" class="text-amber-700 hover:underline">${escapeHtml(shortUrl)}</a> — ${escapeHtml(reason)}`;
        parent.appendChild(el);
    });
}

// --- Article mode ---
async function loadArticle(id) {
    document.getElementById('progress-view').classList.add('hidden');
    document.getElementById('article-view').classList.remove('hidden');

    try {
        const data = await api(`/api/articles/${encodeURIComponent(id)}`);
        document.getElementById('article-title').textContent = data.topic;
        document.title = `STORM - ${data.topic}`;

        citationDict = data.citation_dict || {};

        // Process raw article text like Streamlit does
        let articleText = data.article_text || '';
        articleText = cleanArticleText(articleText);
        articleText = addInlineCitationLinks(articleText, citationDict);

        // Render processed markdown to HTML
        const body = document.getElementById('article-body');
        body.innerHTML = marked.parse(articleText);

        // Inject narrative storyline above the article content
        loadNarrative(id, body);

        buildTOC(body);

        // References
        if (Object.keys(citationDict).length > 0) {
            renderReferences(citationDict);
            buildRefsModal(citationDict);
            document.getElementById('refs-btn').classList.remove('hidden');
        }

        // Conversation log
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

async function loadNarrative(articleId, body) {
    try {
        const res = await fetch(`/api/articles/${encodeURIComponent(articleId)}/narrative`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.html) {
            const div = document.createElement('div');
            body.insertBefore(div, body.firstChild);
            // Use Range to parse HTML and execute scripts
            const range = document.createRange();
            range.selectNode(div);
            const fragment = range.createContextualFragment(data.html);
            div.appendChild(fragment);
        }
    } catch (e) { console.error('Narrative load error:', e); }
}

// Clean article text like Streamlit's _display_main_article_text
function cleanArticleText(text) {
    // Remove "Write the lead section:" prefix
    const leadIdx = text.indexOf('Write the lead section:');
    if (leadIdx !== -1) {
        text = text.substring(leadIdx + 'Write the lead section:'.length);
    }
    // Remove first heading line (the title is shown separately)
    if (text.trimStart().startsWith('#')) {
        const lines = text.split('\n');
        const firstNonEmpty = lines.findIndex(l => l.trim().length > 0);
        if (firstNonEmpty >= 0 && lines[firstNonEmpty].trimStart().startsWith('#')) {
            lines.splice(firstNonEmpty, 1);
        }
        text = lines.join('\n');
    }
    // Fix markdown citation links: ]: "title" http -> ]: http
    text = text.replace(/\]:\s+".*?"\s+http/g, ']: http');
    // Rename "summary" heading to "Report Summary"
    text = text.replace(/^(#+)\s*summary\s*$/im, '$1 Report Summary');
    // Clean up image markdown: collapse multi-line alt text, truncate long captions
    text = text.replace(/!\[([^\]]*?)\]\(([^)]+)\)/gs, (match, alt, url) => {
        alt = alt.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
        if (alt.length > 80) alt = alt.substring(0, 80).trim() + '...';
        return `![${alt}](${url})`;
    });
    // Break up long paragraphs (more than 4 sentences) into shorter ones
    text = breakLongParagraphs(text);
    return text;
}

function breakLongParagraphs(text) {
    // Split into blocks by double newline (markdown paragraphs)
    const blocks = text.split(/\n\n+/);
    const result = [];
    for (const block of blocks) {
        // Skip headings, lists, blockquotes, empty lines
        if (!block.trim() || /^[#\-\*>|]/.test(block.trim()) || block.trim().startsWith('[')) {
            result.push(block);
            continue;
        }
        // Split into sentences: period/!/? followed by space and uppercase or citation
        const sentences = block.split(/(?<=\.(?:\[\d+\])*|[!?])\s+(?=[A-Z""])/);
        if (sentences.length <= 4) {
            result.push(block);
            continue;
        }
        // Group into chunks of 3 sentences
        const chunks = [];
        for (let i = 0; i < sentences.length; i += 3) {
            chunks.push(sentences.slice(i, i + 3).join(' '));
        }
        result.push(chunks.join('\n\n'));
    }
    return result.join('\n\n');
}

// Convert [i] citations to clickable links like Streamlit's add_inline_citation_link
function addInlineCitationLinks(text, citations) {
    return text.replace(/\[(\d+)\]/g, (match, num) => {
        const ref = citations[num];
        const url = ref ? ref.url : '#ref-' + num;
        return `[[${num}]](${url})`;
    });
}

// Title case: capitalize important words, lowercase minor ones
function toTitleCase(str) {
    const minor = new Set(['a','an','the','and','but','or','nor','for','yet','so',
        'in','on','at','to','by','of','up','as','is','it','if','no','not','with','from']);
    return str.replace(/\w\S*/g, (word, idx) => {
        if (idx === 0 || !minor.has(word.toLowerCase())) {
            return word.charAt(0).toUpperCase() + word.slice(1);
        }
        return word.toLowerCase();
    });
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
        const titleText = toTitleCase(h.textContent);
        // Also update the heading in the article body
        h.textContent = titleText;

        const item = document.createElement('div');
        item.className = 'toc-item';

        if (level === 1) {
            // Top-level: bold, no indent
            item.innerHTML = `<a href="#${id}" data-heading="${id}" class="toc-link toc-h1 block text-sm font-semibold text-slate-700 hover:text-blue-600 py-1 transition-colors">${escapeHtml(titleText)}</a>`;
        } else if (level === 2) {
            // Subsection: indented with left border line
            item.innerHTML = `<a href="#${id}" data-heading="${id}" class="toc-link toc-h2 block text-sm text-slate-400 hover:text-blue-600 py-0.5 ml-3 pl-3 border-l border-slate-200 transition-colors">${escapeHtml(titleText)}</a>`;
        } else {
            // Sub-subsection: deeper indent with nested border
            item.innerHTML = `<a href="#${id}" data-heading="${id}" class="toc-link toc-h3 block text-xs text-slate-400 hover:text-blue-600 py-0.5 ml-6 pl-3 border-l border-slate-200 transition-colors">${escapeHtml(titleText)}</a>`;
        }

        tocList.appendChild(item);
    });

    // Track which section is in view and highlight its TOC link
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                tocList.querySelectorAll('.toc-link').forEach(l => {
                    if (l.classList.contains('toc-h1')) {
                        l.classList.remove('text-slate-800');
                        l.classList.add('text-slate-700');
                    } else {
                        l.classList.remove('text-slate-600');
                        l.classList.add('text-slate-400');
                    }
                });
                const active = tocList.querySelector(`[data-heading="${entry.target.id}"]`);
                if (active) {
                    if (active.classList.contains('toc-h1')) {
                        active.classList.remove('text-slate-700');
                        active.classList.add('text-slate-800');
                    } else {
                        active.classList.remove('text-slate-400');
                        active.classList.add('text-slate-600');
                    }
                }
            }
        });
    }, { rootMargin: '0px 0px -70% 0px', threshold: 0 });

    headings.forEach(h => observer.observe(h));
}

function renderReferences(citations) {
    const section = document.getElementById('references-section');
    const list = document.getElementById('references-list');
    section.classList.remove('hidden');

    // Sort by citation index
    const sorted = Object.entries(citations).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));

    list.innerHTML = sorted.map(([index, info]) => {
        const snippetHtml = (info.snippets || []).map(s =>
            `<p class="text-xs text-slate-500 mt-1 leading-relaxed">${escapeHtml(s).substring(0, 200)}${s.length > 200 ? '...' : ''}</p>`
        ).join('');
        return `<div class="border-b border-gray-100 pb-2 mb-2" id="ref-${index}">
            <div class="flex gap-2 items-start">
                <span class="text-slate-400 shrink-0 font-mono text-xs mt-0.5">[${index}]</span>
                <div class="min-w-0">
                    <a href="${escapeAttr(info.url)}" target="_blank" class="text-blue-600 hover:underline text-sm font-medium break-all">${escapeHtml(info.title || info.url)}</a>
                    <p class="text-xs text-slate-400 break-all">${escapeHtml(info.url)}</p>
                    ${snippetHtml}
                </div>
            </div>
        </div>`;
    }).join('');
}

function buildRefsModal(citations) {
    const list = document.getElementById('refs-modal-list');
    const sorted = Object.entries(citations).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
    list.innerHTML = sorted.map(([index, info]) =>
        `<div class="flex gap-2 items-start py-1">
            <span class="text-slate-400 shrink-0 font-mono text-xs mt-0.5">[${index}]</span>
            <a href="${escapeAttr(info.url)}" target="_blank" class="text-blue-600 hover:underline break-all">${escapeHtml(info.url)}</a>
        </div>`
    ).join('');
}

function buildBrainstormModal(log) {
    const tabs = document.getElementById('persona-tabs');
    const content = document.getElementById('persona-content');

    // Parse persona names like Streamlit: split by ": " or "- "
    const parsed = log.map(conv => {
        const perspective = conv.perspective || '';
        let name, description;
        if (perspective.includes(': ')) {
            [name, ...description] = perspective.split(': ');
            description = description.join(': ');
        } else if (perspective.includes('- ')) {
            [name, ...description] = perspective.split('- ');
            description = description.join('- ');
        } else {
            name = '';
            description = perspective;
        }
        return { name: name.trim(), description: description.trim(), turns: conv.dlg_turns || [] };
    });

    tabs.innerHTML = parsed.map((p, i) => {
        const label = p.name || `Persona ${i + 1}`;
        return `<button class="persona-tab px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap ${i === 0 ? 'active' : ''}" data-index="${i}">${escapeHtml(label)}</button>`;
    }).join('');

    // Store parsed data for showPersona
    conversationLog = parsed;
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
    const persona = conversationLog[index];
    if (!persona) return;

    let html = '';

    // Show persona description
    if (persona.description) {
        html += `<div class="bg-blue-50 border border-blue-100 rounded-lg p-3 text-sm text-blue-800">${escapeHtml(persona.description)}</div>`;
    }

    // Render dialogue turns using Streamlit's format: user_utterance / agent_utterance
    (persona.turns || []).forEach(turn => {
        if (turn.user_utterance) {
            html += `<div class="flex gap-3 justify-end">
                <div class="chat-bubble-user px-4 py-2.5 max-w-lg text-sm font-medium chat-md">${marked.parse(turn.user_utterance)}</div>
                <div class="w-8 h-8 bg-slate-200 rounded-full flex items-center justify-center shrink-0">
                    <svg class="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
                </div>
            </div>`;
        }
        if (turn.agent_utterance !== undefined) {
            // Strip citations from agent utterance like Streamlit does
            const agentText = removeCitations(turn.agent_utterance || '');
            html += `<div class="flex gap-3">
                <div class="w-8 h-8 bg-slate-800 rounded-full flex items-center justify-center shrink-0">
                    <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                </div>
                <div class="chat-bubble-agent px-4 py-2.5 max-w-lg text-sm chat-md">${marked.parse(agentText)}</div>
            </div>`;
        }
    });

    content.innerHTML = html || '<p class="text-slate-400 text-center py-8">No conversation data available</p>';
}

// Remove citations like Streamlit's remove_citations
function removeCitations(text) {
    return text.replace(/ ?\[\d+/g, '').replace(/\|/g, '').replace(/\]/g, '');
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
