// Sidebar toggle
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('-translate-x-full');
}

// Highlight active nav link
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    document.querySelectorAll('.sidebar-link').forEach(link => {
        if (link.getAttribute('href') === path || link.getAttribute('href') === path.replace('/static/', '/static/')) {
            link.classList.add('active');
        }
    });
});

// Fetch wrapper
async function api(url, options = {}) {
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

// Load sidebar history (last 10 articles)
document.addEventListener('DOMContentLoaded', async () => {
    const container = document.getElementById('sidebar-history');
    if (!container) return;
    try {
        const articles = await api('/api/articles');
        const recent = articles.slice(0, 10);
        recent.forEach(a => {
            const link = document.createElement('a');
            link.href = `/static/article.html?id=${a.id}`;
            link.className = 'block px-3 py-1.5 text-xs text-slate-400 hover:text-white truncate transition';
            link.textContent = a.topic;
            link.title = a.topic;
            container.appendChild(link);
        });
    } catch (e) {}
});

// Date formatting
function formatDate(ts) {
    const d = new Date(ts * 1000);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function timeAgo(ts) {
    const seconds = Math.floor(Date.now() / 1000 - ts);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return formatDate(ts);
}
