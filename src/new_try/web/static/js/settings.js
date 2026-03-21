const DEFAULTS = {
    openai_url: '',
    openai_key: '',
    openai_model: '',
    search_provider: 'searxng',
    searxng_url: '',
    serper_key: '',
    serper_cache: true,
    brave_key: '',
    xai_enabled: false,
    max_perspective: 5,
    max_conv_turn: 5,
    search_top_k: 5,
    retrieve_top_k: 10,
    article_gen_tokens: 4000,
    article_polish_tokens: 8000,
    outline_gen_tokens: 1000,
    chunk_size: 1000,
};

// Map between server keys (cfg_*) and frontend keys
const KEY_MAP = {
    cfg_llm_url: 'openai_url',
    cfg_llm_key: 'openai_key',
    cfg_llm_model: 'openai_model',
    cfg_search_radio: 'search_provider',
    cfg_searxng_url: 'searxng_url',
    cfg_serper_key: 'serper_key',
    cfg_serper_cache: 'serper_cache',
    cfg_chunk_size: 'chunk_size',
    cfg_max_perspective: 'max_perspective',
    cfg_max_conv_turn: 'max_conv_turn',
    cfg_search_top_k: 'search_top_k',
    cfg_retrieve_top_k: 'retrieve_top_k',
    cfg_article_gen_tokens: 'article_gen_tokens',
    cfg_article_polish_tokens: 'article_polish_tokens',
    cfg_outline_gen_tokens: 'outline_gen_tokens',
};

const PROVIDER_MAP = {
    'SearXNG (self-hosted)': 'searxng',
    'Serper (Google via API)': 'serper',
    'DuckDuckGo (snippets only)': 'duckduckgo',
    'Brave Search (coming soon)': 'brave',
};
const PROVIDER_MAP_REV = Object.fromEntries(Object.entries(PROVIDER_MAP).map(([k, v]) => [v, k]));

let _cachedSettings = null;

async function loadSettings() {
    if (_cachedSettings) return _cachedSettings;
    try {
        const res = await fetch('/api/settings');
        if (res.ok) {
            const server = await res.json();
            const mapped = { ...DEFAULTS };
            for (const [skey, fkey] of Object.entries(KEY_MAP)) {
                if (server[skey] !== undefined) {
                    let val = server[skey];
                    if (skey === 'cfg_search_radio') val = PROVIDER_MAP[val] || val;
                    mapped[fkey] = val;
                }
            }
            _cachedSettings = mapped;
            return mapped;
        }
    } catch {}
    return { ...DEFAULTS };
}

async function saveSettings() {
    const s = {
        openai_url: document.getElementById('set-openai-url').value.trim(),
        openai_key: document.getElementById('set-openai-key').value.trim(),
        openai_model: document.getElementById('set-openai-model').value.trim(),
        search_provider: document.querySelector('input[name="search-provider"]:checked')?.value || 'searxng',
        searxng_url: document.getElementById('set-searxng-url').value.trim(),
        serper_key: document.getElementById('set-serper-key').value.trim(),
        serper_cache: document.getElementById('set-serper-cache').checked,
        brave_key: '',
        xai_enabled: false,
        max_perspective: parseInt(document.getElementById('set-max-perspective').value) || 5,
        max_conv_turn: parseInt(document.getElementById('set-max-conv-turn').value) || 5,
        search_top_k: parseInt(document.getElementById('set-search-top-k').value) || 5,
        retrieve_top_k: parseInt(document.getElementById('set-retrieve-top-k').value) || 10,
        article_gen_tokens: parseInt(document.getElementById('set-article-gen-tokens').value) || 4000,
        article_polish_tokens: parseInt(document.getElementById('set-article-polish-tokens').value) || 8000,
        outline_gen_tokens: parseInt(document.getElementById('set-outline-gen-tokens').value) || 1000,
        chunk_size: parseInt(document.getElementById('set-chunk-size').value) || 1000,
    };
    _cachedSettings = s;

    // Save to server (shared with Streamlit)
    const serverData = {};
    for (const [skey, fkey] of Object.entries(KEY_MAP)) {
        let val = s[fkey];
        if (skey === 'cfg_search_radio') val = PROVIDER_MAP_REV[val] || val;
        serverData[skey] = val;
    }
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(serverData),
        });
    } catch {}

    closeSettings();
}

function getSettings() {
    return _cachedSettings || { ...DEFAULTS };
}

async function openSettings() {
    const s = await loadSettings();
    document.getElementById('set-openai-url').value = s.openai_url;
    document.getElementById('set-openai-key').value = s.openai_key;
    document.getElementById('set-openai-model').value = s.openai_model;
    document.getElementById('set-searxng-url').value = s.searxng_url;
    document.getElementById('set-serper-key').value = s.serper_key;
    document.getElementById('set-serper-cache').checked = s.serper_cache;
    const radio = document.querySelector(`input[name="search-provider"][value="${s.search_provider}"]`);
    if (radio) radio.checked = true;
    document.getElementById('set-max-perspective').value = s.max_perspective;
    document.getElementById('set-max-conv-turn').value = s.max_conv_turn;
    document.getElementById('set-search-top-k').value = s.search_top_k;
    document.getElementById('set-retrieve-top-k').value = s.retrieve_top_k;
    document.getElementById('set-article-gen-tokens').value = s.article_gen_tokens;
    document.getElementById('set-article-polish-tokens').value = s.article_polish_tokens;
    document.getElementById('set-outline-gen-tokens').value = s.outline_gen_tokens;
    document.getElementById('set-chunk-size').value = s.chunk_size;
    document.getElementById('settings-modal').classList.remove('hidden');
}

function closeSettings() {
    document.getElementById('settings-modal').classList.add('hidden');
}

function toggleSection(id) {
    const el = document.getElementById(id);
    const arrow = document.getElementById(id + '-arrow');
    el.classList.toggle('hidden');
    arrow.textContent = el.classList.contains('hidden') ? '+' : '\u2212';
}

// Load settings on page load so getSettings() works synchronously for form submit
document.addEventListener('DOMContentLoaded', async () => {
    await loadSettings();
    const modal = document.getElementById('settings-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === e.currentTarget) closeSettings();
        });
    }
});
