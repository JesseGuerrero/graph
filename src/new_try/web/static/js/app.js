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

    // Add collapse toggle to sidebar — button lives on document.body so
    // it remains visible even when sidebar is collapsed to 0 width
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    sidebar.style.transition = 'width 0.2s ease, min-width 0.2s ease';

    const btn = document.createElement('button');
    btn.className = 'sidebar-collapse-btn';
    btn.title = 'Toggle sidebar';
    document.body.appendChild(btn);

    function positionBtn() {
        btn.style.left = sidebar.getBoundingClientRect().right + 'px';
    }

    const collapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (collapsed) collapseSidebar(sidebar, btn);
    else positionBtn();

    btn.addEventListener('click', () => {
        if (sidebar.dataset.collapsed === '1') {
            expandSidebar(sidebar, btn);
            localStorage.setItem('sidebarCollapsed', 'false');
        } else {
            collapseSidebar(sidebar, btn);
            localStorage.setItem('sidebarCollapsed', 'true');
        }
    });

    function collapseSidebar(sb, b) {
        sb.dataset.collapsed = '1';
        sb.style.width = '0';
        sb.style.minWidth = '0';
        sb.style.overflow = 'hidden';
        sb.style.borderRight = 'none';
        sb.style.padding = '0';
        b.textContent = '\u203a';
        b.style.left = '0px';
    }

    function expandSidebar(sb, b) {
        sb.dataset.collapsed = '0';
        sb.style.width = '';
        sb.style.minWidth = '';
        sb.style.overflow = '';
        sb.style.borderRight = '';
        sb.style.padding = '';
        b.textContent = '\u2039';
        // Wait for transition to finish to read final width
        setTimeout(() => { b.style.left = sb.getBoundingClientRect().right + 'px'; }, 220);
    }
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
