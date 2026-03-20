document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('topic-form');
    const input = document.getElementById('topic-input');

    // Example cards fill input and submit
    document.querySelectorAll('.example-card').forEach(card => {
        card.addEventListener('click', () => {
            input.value = card.querySelector('h3').textContent;
            form.dispatchEvent(new Event('submit'));
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const topic = input.value.trim();
        if (!topic) return;

        input.disabled = true;
        try {
            const data = await api('/api/run', {
                method: 'POST',
                body: JSON.stringify({ topic, settings: getSettings() }),
            });
            window.location.href = `/static/article.html?run_id=${data.run_id}`;
        } catch (err) {
            alert('Failed to start: ' + err.message);
            input.disabled = false;
        }
    });
});
