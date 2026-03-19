let articles = [];
let filter = 'all';

document.addEventListener('DOMContentLoaded', async () => {
    await loadArticles();

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('border-blue-600', 'text-blue-600');
                b.classList.add('border-transparent', 'text-slate-500');
            });
            btn.classList.add('border-blue-600', 'text-blue-600');
            btn.classList.remove('border-transparent', 'text-slate-500');
            filter = btn.dataset.filter;
            render();
        });
    });

    // Search
    document.getElementById('search-input').addEventListener('input', () => render());
});

async function loadArticles() {
    try {
        articles = await api('/api/articles');
    } catch (e) {
        articles = [];
    }
    render();
}

function render() {
    const search = document.getElementById('search-input').value.toLowerCase();
    const list = document.getElementById('article-list');
    const empty = document.getElementById('empty-state');

    let filtered = articles;
    if (filter !== 'all') {
        filtered = filtered.filter(a => a.type === filter);
    }
    if (search) {
        filtered = filtered.filter(a => a.topic.toLowerCase().includes(search));
    }

    if (filtered.length === 0) {
        list.innerHTML = '';
        empty.classList.remove('hidden');
        return;
    }

    empty.classList.add('hidden');
    list.innerHTML = filtered.map(a => `
        <a href="/static/article.html?id=${encodeURIComponent(a.id)}"
           class="block bg-white border border-gray-200 rounded-xl p-5 hover:border-blue-300 hover:shadow-sm transition">
            <div class="flex items-start justify-between">
                <div class="flex-1">
                    <h3 class="font-semibold text-slate-800 mb-1">${escapeHtml(a.topic)}</h3>
                    <p class="text-sm text-slate-500">${timeAgo(a.created_at)}</p>
                </div>
                <span class="text-xs bg-blue-50 text-blue-600 px-2.5 py-1 rounded-full font-medium">STORM Article</span>
            </div>
        </a>
    `).join('');
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}
