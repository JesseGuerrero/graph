"""Automate claude.ai Deep Research via Playwright.

Opens a visible browser with a persistent profile (user stays logged in),
submits a research prompt, waits for the full response, and extracts
the resulting markdown.
"""

import asyncio
import logging
import time
from pathlib import Path

logger = logging.getLogger("claude_research")

CLAUDE_URL = "https://claude.ai"
PROFILE_DIR = Path("C:/Users/jesse/Desktop/demo/claude_profile")
PROFILE_DIR.mkdir(exist_ok=True)


async def run_claude_research(topic: str, on_status=None):
    """
    Open claude.ai, submit a research prompt, and extract the response markdown.

    Args:
        topic: The research topic/question
        on_status: Optional callback fn(status_str) for progress updates

    Returns:
        dict with {markdown, title, model}
    """
    from patchright.async_api import async_playwright

    def status(msg):
        logger.info(msg)
        if on_status:
            on_status(msg)

    status("Launching browser...")
    pw = await async_playwright().start()
    browser = await pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=False,
        viewport={"width": 1280, "height": 900},
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ],
    )

    page = browser.pages[0] if browser.pages else await browser.new_page()

    try:
        # Navigate to claude.ai
        status("Opening claude.ai...")
        await page.goto(CLAUDE_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        # Check if logged in — look for the chat input
        status("Checking login status...")
        try:
            await page.wait_for_selector(
                '[contenteditable="true"], div.ProseMirror, textarea[placeholder]',
                timeout=10000,
            )
            status("Logged in.")
        except Exception:
            status("NOT LOGGED IN — please log in to claude.ai in the browser window. Waiting...")
            # Wait for the user to log in manually — poll for chat input
            for _ in range(120):  # Wait up to 4 minutes
                await page.wait_for_timeout(2000)
                el = await page.query_selector(
                    '[contenteditable="true"], div.ProseMirror, textarea[placeholder]'
                )
                if el:
                    status("Login detected. Continuing...")
                    break
            else:
                raise TimeoutError("Timed out waiting for login")

        # Click "new chat" if there's an existing conversation
        status("Starting new chat...")
        try:
            new_chat_btn = await page.query_selector(
                'a[href="/new"], button[data-testid="new-chat-button"], '
                'a[aria-label="New chat"], button[aria-label="New chat"]'
            )
            if new_chat_btn:
                await new_chat_btn.click()
                await page.wait_for_timeout(1500)
        except Exception:
            pass

        # Build the research prompt
        research_prompt = (
            f"Please conduct deep, thorough research on the following topic and "
            f"produce a comprehensive, well-cited markdown report:\n\n"
            f"**{topic}**\n\n"
            f"Requirements:\n"
            f"- Use web search to find current, accurate information\n"
            f"- Include inline citations with source URLs\n"
            f"- Structure with clear headings and sections\n"
            f"- Be comprehensive — cover all major aspects\n"
            f"- Include a references section at the end"
        )

        # Type the prompt into the chat input
        status("Typing research prompt...")
        editor = await page.wait_for_selector(
            '[contenteditable="true"], div.ProseMirror',
            timeout=10000,
        )
        await editor.click()
        await page.wait_for_timeout(300)

        # Use keyboard to type (more reliable than fill for contenteditable)
        await page.keyboard.type(research_prompt, delay=5)
        await page.wait_for_timeout(500)

        # Submit — press Enter or click send button
        status("Submitting prompt...")
        send_btn = await page.query_selector(
            'button[aria-label="Send Message"], '
            'button[data-testid="send-button"], '
            'button[type="submit"]'
        )
        if send_btn:
            await send_btn.click()
        else:
            await page.keyboard.press("Enter")

        await page.wait_for_timeout(2000)

        # Wait for response to complete — poll for the stop/typing indicator to disappear
        status("Waiting for Claude to respond (this may take a few minutes)...")
        max_wait = 600  # 10 minutes max
        poll_interval = 3
        elapsed = 0
        last_len = 0
        stable_count = 0

        while elapsed < max_wait:
            await page.wait_for_timeout(poll_interval * 1000)
            elapsed += poll_interval

            # Check if there's a stop button (means still generating)
            stop_btn = await page.query_selector(
                'button[aria-label="Stop Response"], '
                'button[data-testid="stop-button"], '
                'button:has-text("Stop")'
            )

            if stop_btn:
                # Still generating
                stable_count = 0
                if elapsed % 15 == 0:
                    status(f"Still generating... ({elapsed}s)")
                continue

            # No stop button — check if content has stabilized
            content = await _extract_last_response(page)
            cur_len = len(content) if content else 0

            if cur_len > 0 and cur_len == last_len:
                stable_count += 1
                if stable_count >= 3:
                    status(f"Response complete ({cur_len} chars, {elapsed}s)")
                    break
            else:
                stable_count = 0
                last_len = cur_len

        # Extract the final response
        status("Extracting response markdown...")
        markdown = await _extract_last_response(page)

        if not markdown or len(markdown) < 100:
            raise ValueError(f"Response too short ({len(markdown) if markdown else 0} chars)")

        status(f"Done! Got {len(markdown)} chars")
        return {
            "markdown": markdown,
            "title": topic,
            "model": "claude-deep-research",
        }

    finally:
        await browser.close()
        await pw.stop()


async def _extract_last_response(page):
    """Extract the last assistant message from the claude.ai chat."""
    # Try multiple selectors for the response container
    selectors = [
        # Claude's response blocks
        '[data-testid="assistant-message"]:last-of-type',
        '.font-claude-message:last-of-type',
        '[class*="agent-turn"]:last-of-type .markdown',
        '[class*="response"]:last-of-type',
        # Generic — last message block that's not the user's
        'div[class*="message"]:not([class*="user"]):last-of-type',
    ]

    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                if text and len(text) > 50:
                    return text.strip()
        except Exception:
            continue

    # Fallback: get all text from message containers
    try:
        # Get the last large text block on the page
        result = await page.evaluate("""
            () => {
                // Find all message-like containers
                const blocks = document.querySelectorAll(
                    '[data-testid*="message"], [class*="message"], [class*="response"], .markdown'
                );
                if (blocks.length === 0) return '';
                // Return the last one's text
                const last = blocks[blocks.length - 1];
                return last.innerText || '';
            }
        """)
        if result and len(result) > 50:
            return result.strip()
    except Exception:
        pass

    return ""
