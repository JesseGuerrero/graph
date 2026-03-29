"""Run Mind2Web-2 dev set tasks through claude.ai Research mode.

Opens a persistent Playwright browser, activates Research mode,
submits each task sequentially, extracts the markdown response,
and writes results to answers.csv.
"""

import asyncio
import csv
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("run_research")

PROFILE_DIR = Path("C:/Users/jesse/Desktop/demo/claude_profile")
PROFILE_DIR.mkdir(exist_ok=True)
DEV_CSV = Path(__file__).parent / "Mind2Web-2" / "dev_set.csv"
OUT_CSV = Path(__file__).parent / "answers.csv"
CLAUDE_URL = "https://claude.ai"

SYSTEM_SUFFIX = (
    "\n\nIMPORTANT formatting requirements for your response:\n"
    "- Write the full response in rich Markdown\n"
    "- Use headers (##), bold, tables, and bullet lists\n"
    "- Include inline hyperlinks in Markdown format: [text](url)\n"
    "- Cite every factual claim with a source URL inline\n"
    "- End with a ## References section listing all URLs used\n"
    "- Be comprehensive and thorough"
)


async def wait_for_chat_input(page, timeout=120):
    """Wait for claude.ai's chat input to be ready."""
    for _ in range(timeout // 2):
        el = await page.query_selector(
            'div.ProseMirror[contenteditable="true"], '
            'div[contenteditable="true"][translate="no"], '
            'fieldset div[contenteditable="true"]'
        )
        if el:
            return el
        await page.wait_for_timeout(2000)
    raise TimeoutError("Chat input not found")


async def click_new_chat(page):
    """Start a fresh conversation."""
    # Try the "new chat" button / link
    for sel in [
        'a[href="/new"]',
        'button[data-testid="new-chat-button"]',
        'a[aria-label="New chat"]',
        'button[aria-label="New chat"]',
    ]:
        btn = await page.query_selector(sel)
        if btn:
            await btn.click()
            await page.wait_for_timeout(2000)
            return
    # Fallback: navigate directly
    await page.goto(f"{CLAUDE_URL}/new", wait_until="domcontentloaded")
    await page.wait_for_timeout(2000)


async def activate_research_mode(page):
    """Try to enable Research mode / extended thinking in claude.ai."""
    # Look for a Research toggle, model selector, or similar UI element
    for sel in [
        'button:has-text("Research")',
        'button[aria-label="Research"]',
        '[data-testid="research-toggle"]',
        'button:has-text("Extended")',
    ]:
        try:
            btn = await page.query_selector(sel)
            if btn:
                await btn.click()
                await page.wait_for_timeout(1000)
                logger.info("Activated research mode via: %s", sel)
                return True
        except Exception:
            continue
    logger.warning("Could not find Research mode toggle — using default mode")
    return False


async def submit_and_wait(page, prompt, task_id, max_wait=900):
    """Type prompt, submit, wait for full response, return markdown text."""
    editor = await wait_for_chat_input(page)
    await editor.click()
    await page.wait_for_timeout(300)

    # Clear any existing text
    await page.keyboard.press("Control+a")
    await page.keyboard.press("Backspace")
    await page.wait_for_timeout(200)

    # Type the prompt
    await page.keyboard.type(prompt, delay=3)
    await page.wait_for_timeout(500)

    # Submit
    send_btn = await page.query_selector(
        'button[aria-label="Send Message"], '
        'button[data-testid="send-button"], '
        'button[type="submit"]'
    )
    if send_btn:
        await send_btn.click()
    else:
        await page.keyboard.press("Enter")

    logger.info("[%s] Prompt submitted, waiting for response...", task_id)
    await page.wait_for_timeout(3000)

    # Poll until response is complete
    elapsed = 0
    poll = 5
    last_len = 0
    stable = 0

    while elapsed < max_wait:
        await page.wait_for_timeout(poll * 1000)
        elapsed += poll

        # Check for stop button = still generating
        stop = await page.query_selector(
            'button[aria-label="Stop Response"], '
            'button[data-testid="stop-button"], '
            'button:has-text("Stop")'
        )
        if stop:
            stable = 0
            if elapsed % 30 == 0:
                logger.info("[%s] Still generating... %ds", task_id, elapsed)
            continue

        # Content stable check
        text = await extract_response(page)
        cur_len = len(text) if text else 0
        if cur_len > 100 and cur_len == last_len:
            stable += 1
            if stable >= 3:
                logger.info("[%s] Response complete: %d chars in %ds", task_id, cur_len, elapsed)
                return text
        else:
            stable = 0
            last_len = cur_len

    # Timeout — return whatever we have
    text = await extract_response(page)
    logger.warning("[%s] Timed out at %ds, got %d chars", task_id, max_wait, len(text) if text else 0)
    return text or ""


async def extract_response(page):
    """Extract the last assistant response as text."""
    # Try specific selectors first
    for sel in [
        '[data-testid="assistant-message"]:last-of-type',
        '.font-claude-message:last-of-type',
        '[class*="agent-turn"]:last-of-type',
    ]:
        try:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                if text and len(text) > 50:
                    return text.strip()
        except Exception:
            continue

    # Fallback: JS extraction
    try:
        return await page.evaluate("""
            () => {
                const msgs = document.querySelectorAll(
                    '[data-testid*="message"], [class*="message"], [class*="response"]'
                );
                if (!msgs.length) return '';
                // Find the last long message (assistant response)
                let best = '';
                for (const m of msgs) {
                    const t = m.innerText || '';
                    if (t.length > best.length) best = t;
                }
                return best.trim();
            }
        """)
    except Exception:
        return ""


async def main():
    from patchright.async_api import async_playwright

    # Load tasks
    tasks = []
    with open(DEV_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    logger.info("Loaded %d tasks from %s", len(tasks), DEV_CSV)

    # Load existing answers to allow resuming
    done_ids = set()
    existing_rows = []
    if OUT_CSV.exists():
        with open(OUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_ids.add(row["task_id"])
                existing_rows.append(row)
        logger.info("Found %d existing answers, will skip those", len(done_ids))

    # Launch browser
    logger.info("Launching browser...")
    pw = await async_playwright().start()
    browser = await pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=False,
        viewport={"width": 1400, "height": 900},
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )
    page = browser.pages[0] if browser.pages else await browser.new_page()

    try:
        # Navigate to claude.ai
        logger.info("Opening claude.ai...")
        await page.goto(CLAUDE_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)

        # Check login
        try:
            await wait_for_chat_input(page, timeout=15)
            logger.info("Logged in.")
        except TimeoutError:
            logger.info("NOT LOGGED IN — please log in in the browser window.")
            await wait_for_chat_input(page, timeout=300)
            logger.info("Login detected.")

        # Process each task
        results = list(existing_rows)
        for i, task in enumerate(tasks):
            tid = task["task_id"]
            if tid in done_ids:
                logger.info("[%d/%d] %s — already done, skipping", i + 1, len(tasks), tid)
                continue

            logger.info("[%d/%d] Starting task: %s", i + 1, len(tasks), tid)

            # New chat for each task
            await click_new_chat(page)
            await page.wait_for_timeout(1500)

            # Try to activate research mode
            await activate_research_mode(page)

            # Build prompt
            prompt = task["task_description"] + SYSTEM_SUFFIX

            # Submit and wait
            answer = await submit_and_wait(page, prompt, tid)

            if not answer or len(answer) < 50:
                logger.error("[%s] Got empty/short response, saving anyway", tid)

            # Save result
            results.append({
                "task_id": tid,
                "task_description": task["task_description"],
                "answer": answer,
            })

            # Write CSV after each task (crash-safe)
            with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["task_id", "task_description", "answer"])
                writer.writeheader()
                writer.writerows(results)

            logger.info("[%s] Saved. %d/%d complete.", tid, len(results), len(tasks))

            # Brief pause between tasks
            await page.wait_for_timeout(3000)

    finally:
        await browser.close()
        await pw.stop()

    logger.info("All done! Results in %s", OUT_CSV)


if __name__ == "__main__":
    asyncio.run(main())
