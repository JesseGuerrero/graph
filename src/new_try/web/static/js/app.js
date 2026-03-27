// Sidebar toggle (mobile hamburger)
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('-translate-x-full');
}

// Highlight active nav link + add collapse button
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    document.querySelectorAll('.sidebar-link').forEach(link => {
        if (link.getAttribute('href') === path || link.getAttribute('href') === path.replace('/static/', '/static/')) {
            link.classList.add('active');
        }
    });

    // Add collapse toggle to sidebar
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    sidebar.style.position = 'relative';
    sidebar.style.transition = 'width 0.2s ease, min-width 0.2s ease';

    const btn = document.createElement('button');
    btn.className = 'sidebar-collapse-btn';
    btn.title = 'Toggle sidebar';
    btn.textContent = '\u2039';
    sidebar.appendChild(btn);

    const collapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (collapsed) collapseSidebar(sidebar, btn);

    btn.addEventListener('click', () => {
        if (sidebar.dataset.collapsed === '1') {
            expandSidebar(sidebar, btn);
            localStorage.setItem('sidebarCollapsed', 'false');
        } else {
            collapseSidebar(sidebar, btn);
            localStorage.setItem('sidebarCollapsed', 'true');
        }
    });
});

function collapseSidebar(sidebar, btn) {
    sidebar.dataset.collapsed = '1';
    sidebar.style.width = '0';
    sidebar.style.minWidth = '0';
    sidebar.style.overflow = 'hidden';
    sidebar.style.borderRight = 'none';
    sidebar.style.padding = '0';
    btn.textContent = '\u203a';
}

function expandSidebar(sidebar, btn) {
    sidebar.dataset.collapsed = '0';
    sidebar.style.width = '';
    sidebar.style.minWidth = '';
    sidebar.style.overflow = '';
    sidebar.style.borderRight = '';
    sidebar.style.padding = '';
    btn.textContent = '\u2039';
}

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
    const isKgPage = window.location.pathname.includes('knowledge_graph');
    try {
        const articles = await api('/api/articles');
        const recent = articles.slice(0, 10);
        recent.forEach(a => {
            const link = document.createElement('a');
            link.href = isKgPage
                ? `/static/knowledge_graph.html?id=${a.id}`
                : `/static/article.html?id=${a.id}`;
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
