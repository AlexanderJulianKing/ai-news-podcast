"""Read front pages the way a person's browser shows them.

A normal, visible Chromium window runs on a virtual screen (Xvfb) on the Pi, loads the
page, and waits for it to finish. The rendered text and a screenshot are read through
Chromium's DevTools connection. Nothing is disguised: the browser reports its own
built-in user agent. (Headless Chromium announces itself as "HeadlessChrome", which
Cloudflare blocks on apnews.com; a visible window does not, as of 2026-09-23.)

Then GPT-6 Luna turns the rendered text into the event list the show uses, in the order
the page shows the stories. On 2026-09-23, checked against Alex's own screenshots, this
matched NPR's and Democracy Now's top stories where Gemini's search-grounded scrape
mostly returned other outlets' stories, and it caught AP's lead story that Gemini's URL
reader skipped.

Any failure (no Chromium or Xvfb, a bot check, a thin page, a timeout) raises
RenderError so the caller can fall back to the older scrape.
"""
import asyncio
import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime

import newscaster.config as _config

CHALLENGE_MARKERS = ("security verification", "just a moment", "verify you are human", "access denied",
                     "attention required", "enable javascript and cookies")


class RenderError(RuntimeError):
    pass


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _read_page(ws_url, want_screenshot):
    import websockets  # installed in the Pi's venv; imported here so tests need not have it

    async with websockets.connect(ws_url, max_size=60_000_000) as ws:
        counter = {"n": 0}

        async def send(method, params=None):
            counter["n"] += 1
            call_id = counter["n"]
            await ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == call_id:
                    return msg.get("result", {})

        async def evaluate(expression):
            result = await send("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            return (result.get("result") or {}).get("value")

        text = await evaluate("document.body ? document.body.innerText : ''") or ""
        title = await evaluate("document.title") or ""
        shot = (await send("Page.captureScreenshot", {"format": "png"})).get("data") if want_screenshot else None
        return title, text, shot


PROFILE_PREFIX = "newscaster_chromium_"


def reap_orphans():
    """Kill browsers a previous run left behind (if its Python process died mid-page) and
    delete their temp profiles. Only touches processes started with our profile prefix."""
    try:
        found = subprocess.run(["pgrep", "-f", "user-data-dir=.*" + PROFILE_PREFIX],
                               capture_output=True, text=True, timeout=10).stdout.split()
    except Exception:
        found = []
    own = os.getpgid(0)
    for pid in found:
        try:
            group = os.getpgid(int(pid))
            if group != own:
                os.killpg(group, signal.SIGKILL)
        except Exception:
            pass
    root = tempfile.gettempdir()
    for name in os.listdir(root):
        if name.startswith(PROFILE_PREFIX):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def check_page(text):
    """Raise RenderError for a bot-check page or one with too little text to be a front page."""
    words = len((text or "").split())
    minimum = getattr(_config, "BROWSER_MIN_WORDS", 300)
    if words < minimum and any(m in (text or "").lower() for m in CHALLENGE_MARKERS):
        raise RenderError("bot check page: {!r}".format((text or "")[:120]))
    if words < minimum:
        raise RenderError("page too thin ({} words)".format(words))


def render_page(url, *, wait_seconds=None, screenshot_path=None):
    """Load `url` in a visible Chromium on a virtual screen. Returns {'title', 'text'}.

    Raises RenderError on any failure, a bot check, or a page with too little text.
    """
    chromium = shutil.which(getattr(_config, "BROWSER_BINARY", "chromium"))
    xvfb_run = shutil.which("xvfb-run")
    if not chromium or not xvfb_run:
        raise RenderError("chromium or xvfb-run is not installed here")
    wait = wait_seconds if wait_seconds is not None else getattr(_config, "BROWSER_WAIT_SECONDS", 20)
    width, height = getattr(_config, "BROWSER_WINDOW", (1366, 3000))
    reap_orphans()
    port = _free_port()
    profile = tempfile.mkdtemp(prefix=PROFILE_PREFIX)
    proc = subprocess.Popen(
        [xvfb_run, "-a", "-s", "-screen 0 {}x{}x24".format(width, height), chromium,
         "--no-first-run", "--no-default-browser-check", "--disable-gpu",
         "--window-size={},{}".format(width, height), "--remote-debugging-port={}".format(port),
         "--remote-allow-origins=*", "--user-data-dir=" + profile, url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        time.sleep(wait)
        try:
            tabs = json.load(urllib.request.urlopen("http://127.0.0.1:{}/json".format(port), timeout=10))
            page = next(t for t in tabs if t.get("type") == "page")
        except Exception as e:
            raise RenderError("could not reach the browser: {}".format(e))
        try:
            title, text, shot = asyncio.run(asyncio.wait_for(
                _read_page(page["webSocketDebuggerUrl"], bool(screenshot_path)), timeout=45))
        except Exception as e:
            raise RenderError("could not read the page: {}".format(e))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        shutil.rmtree(profile, ignore_errors=True)

    check_page(text)
    if screenshot_path and shot:
        os.makedirs(os.path.dirname(screenshot_path) or ".", exist_ok=True)
        with open(screenshot_path, "wb") as f:
            f.write(base64.b64decode(shot))
    return {"title": title, "text": text}


def prune_screenshots(folder, keep_days=14):
    """Delete screenshots older than `keep_days` so they cannot fill the SD card."""
    if not os.path.isdir(folder):
        return
    cutoff = time.time() - keep_days * 86400
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if name.endswith(".png") and os.path.getmtime(path) < cutoff:
            os.remove(path)


ORDER_NOTE = ("List the items in the order the page shows them, starting with the most prominent story at the top "
              "of the page. Skip navigation, ads, newsletter prompts, and evergreen features.")


def scrape_rendered(url, label, event_prompt, timestamp_rules, ask, *, screenshot_dir=None, date_key=None):
    """Render `url` and turn its text into the event list with `ask(user_prompt)`.

    Raises RenderError when the page cannot be rendered, and RuntimeError on an empty list.
    """
    shot = None
    if screenshot_dir:
        shot = os.path.join(screenshot_dir, "{}_{}.png".format(date_key or datetime.now().strftime("%Y_%m_%d"), label))
    page = render_page(url, screenshot_path=shot)
    limit = getattr(_config, "BROWSER_TEXT_CHARS", 60000)
    prompt = "{}{}{}\n\nRENDERED FRONT PAGE TEXT ({}):\n{}".format(
        event_prompt, timestamp_rules, ORDER_NOTE, url, page["text"][:limit])
    items = (ask(prompt) or "").strip()
    if not items:
        raise RuntimeError("empty list from the model for {}".format(url))
    return items
