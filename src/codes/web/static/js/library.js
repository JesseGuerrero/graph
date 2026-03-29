let articles = [];
let currentPage = 1;
let pageSize = 24;

document.addEventListener('DOMContentLoaded', async () => {
    await loadArticles();

    document.getElementById('search-input').addEventListener('input', () => {
        currentPage = 1;
        render();
    });

    document.getElementById('page-size').addEventListener('change', (e) => {
        pageSize = parseInt(e.target.value);
        currentPage = 1;
        render();
    });

    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) { currentPage--; render(); }
    });

    document.getElementById('next-page').addEventListener('click', () => {
        const filtered = getFiltered();
        const totalPages = Math.ceil(filtered.length / pageSize) || 1;
        if (currentPage < totalPages) { currentPage++; render(); }
    });
});

async function loadArticles() {
    try {
        articles = await api('/api/articles');
    } catch (e) {
        articles = [];
    }
    render();
}

function getFiltered() {
    const search = document.getElementById('search-input').value.toLowerCase();
    let filtered = articles;
    if (search) {
        filtered = filtered.filter(a => a.topic.toLowerCase().includes(search));
    }
    return filtered;
}

function render() {
    const filtered = getFiltered();
    const grid = document.getElementById('article-grid');
    const empty = document.getElementById('empty-state');
    const pagination = document.getElementById('pagination');

    if (filtered.length === 0) {
        grid.innerHTML = '';
        pagination.classList.add('hidden');
        empty.classList.remove('hidden');
        return;
    }

    empty.classList.add('hidden');

    // Pagination
    const totalPages = Math.ceil(filtered.length / pageSize) || 1;
    if (currentPage > totalPages) currentPage = totalPages;
    const start = (currentPage - 1) * pageSize;
    const end = Math.min(start + pageSize, filtered.length);
    const page = filtered.slice(start, end);

    // Render 3-column card grid like Streamlit
    grid.innerHTML = page.map(a => `
        <a href="/static/article.html?id=${encodeURIComponent(a.id)}"
           class="article-card block bg-white border border-gray-200 rounded-xl p-5 hover:border-blue-300 hover:shadow-sm transition"
           style="border-left: 0.3rem solid #9AD8E1;">
            <span class="text-xs text-slate-400 block mb-2">My Article</span>
            <h3 class="font-semibold text-slate-800 text-sm truncate" title="${escapeAttr(a.topic)}">${escapeHtml(a.topic)}</h3>
            <p class="text-xs text-slate-500 mt-2">${timeAgo(a.created_at)}</p>
        </a>
    `).join('');

    // Pagination controls
    if (filtered.length > pageSize) {
        pagination.classList.remove('hidden');
        document.getElementById('page-info').textContent = `Page ${currentPage} of ${totalPages}`;
        document.getElementById('prev-page').disabled = currentPage <= 1;
        document.getElementById('next-page').disabled = currentPage >= totalPages;
    } else {
        pagination.classList.add('hidden');
    }
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function escapeAttr(s) {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
