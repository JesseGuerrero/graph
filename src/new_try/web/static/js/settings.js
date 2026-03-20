const SETTINGS_KEY = 'storm_settings';
const DEFAULTS = {
    openai_url: '',
    openai_key: '',
    openai_model: '',
    search_provider: 'searxng',
    searxng_url: '',
    brave_key: '',
    xai_enabled: false,
    max_perspective: 5,
    max_conv_turn: 5,
    search_top_k: 5,
    retrieve_top_k: 10,
    article_gen_tokens: 4000,
    article_polish_tokens: 8000,
    outline_gen_tokens: 1000,
};

function loadSettings() {
    try {
        return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY)) };
    } catch { return { ...DEFAULTS }; }
}

function saveSettings() {
    const s = {
        openai_url: document.getElementById('set-openai-url').value.trim(),
        openai_key: document.getElementById('set-openai-key').value.trim(),
        openai_model: document.getElementById('set-openai-model').value.trim(),
        search_provider: document.querySelector('input[name="search-provider"]:checked')?.value || 'searxng',
        searxng_url: document.getElementById('set-searxng-url').value.trim(),
        brave_key: '',
        xai_enabled: false,
        max_perspective: parseInt(document.getElementById('set-max-perspective').value) || 5,
        max_conv_turn: parseInt(document.getElementById('set-max-conv-turn').value) || 5,
        search_top_k: parseInt(document.getElementById('set-search-top-k').value) || 5,
        retrieve_top_k: parseInt(document.getElementById('set-retrieve-top-k').value) || 10,
        article_gen_tokens: parseInt(document.getElementById('set-article-gen-tokens').value) || 4000,
        article_polish_tokens: parseInt(document.getElementById('set-article-polish-tokens').value) || 8000,
        outline_gen_tokens: parseInt(document.getElementById('set-outline-gen-tokens').value) || 1000,
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    closeSettings();
}

function getSettings() {
    const s = loadSettings();
    const out = {};
    if (s.openai_url) out.openai_url = s.openai_url;
    if (s.openai_key) out.openai_key = s.openai_key;
    if (s.openai_model) out.openai_model = s.openai_model;
    out.search_provider = s.search_provider;
    if (s.searxng_url) out.searxng_url = s.searxng_url;
    out.max_perspective = s.max_perspective;
    out.max_conv_turn = s.max_conv_turn;
    out.search_top_k = s.search_top_k;
    out.retrieve_top_k = s.retrieve_top_k;
    out.article_gen_tokens = s.article_gen_tokens;
    out.article_polish_tokens = s.article_polish_tokens;
    out.outline_gen_tokens = s.outline_gen_tokens;
    return out;
}

function openSettings() {
    const s = loadSettings();
    document.getElementById('set-openai-url').value = s.openai_url;
    document.getElementById('set-openai-key').value = s.openai_key;
    document.getElementById('set-openai-model').value = s.openai_model;
    document.getElementById('set-searxng-url').value = s.searxng_url;
    const radio = document.querySelector(`input[name="search-provider"][value="${s.search_provider}"]`);
    if (radio) radio.checked = true;
    document.getElementById('set-max-perspective').value = s.max_perspective;
    document.getElementById('set-max-conv-turn').value = s.max_conv_turn;
    document.getElementById('set-search-top-k').value = s.search_top_k;
    document.getElementById('set-retrieve-top-k').value = s.retrieve_top_k;
    document.getElementById('set-article-gen-tokens').value = s.article_gen_tokens;
    document.getElementById('set-article-polish-tokens').value = s.article_polish_tokens;
    document.getElementById('set-outline-gen-tokens').value = s.outline_gen_tokens;
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

document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('settings-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === e.currentTarget) closeSettings();
        });
    }
});
