"""Does Gemini's front-page scrape match the actual front page? And could a non-Google
model reading the rendered page do as well?

Reference: the page as headless Chromium on the Pi rendered it (text extracted from the
DOM), fetched minutes before the scrapes. For each site: the show's exact Gemini scrape
(grounding or URL reader, as in topic_finder) and GPT-6 Luna given the rendered page text
with the same instructions. Opus 5.5 audits each list against the page text.
"""
import json
import os
import random
import re
import sys
from datetime import datetime

from bs4 import BeautifulSoup

from benchmarks.tier_swap.run_tier_swap import OUT_DIR, call, cost as call_cost

DATE = "September 23, 2026"
D = sys.argv[1]
SITES = {  # name: (production mode, how the show asks)
    "npr": ("grounding", "NPR's morning news brief and homepage (npr.org)"),
    "dn": ("grounding", "Democracy Now (https://www.democracynow.org)"),
    "pp": ("url", "https://www.propublica.org"),
    "riverside": ("url_riverside", "https://www.riversideca.gov/media"),
}
AUDIT = (
    "You are auditing a list of news items that a scraper produced from one outlet's front page. You get the scraper's "
    "instructions, the text of that front page as a real browser rendered it minutes earlier, and two item lists labeled "
    "A and B in random order. For EACH list, find: (1) items not on the page at all; (2) items on the page but described "
    "wrongly (wrong actor, number, date, or claim); (3) news items the page shows that the instructions ask for but the "
    "list missed, counting only real news stories, not navigation, ads, or evergreen features. Page text may be cut off "
    "near the end; do not count misses from the part you cannot see.\n\nReply with JSON only: {\"A\": {\"not_on_page\": "
    "[...], \"wrong\": [...], \"missed\": [...]}, \"B\": {...}, \"note\": \"one sentence\"}"
)


def page_text(name):
    soup = BeautifulSoup(open(os.path.join(D, name + ".html"), errors="ignore").read(), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text("\n").split("\n")]
    return "\n".join(l for l in lines if len(l) > 3)


def prompts_for(name):
    from newscaster import prompts
    event = prompts.EVENT_SCRAPER_PROMPT.format(date=DATE, max_items=20)
    ts = prompts.EVENT_SCRAPER_TIMESTAMP_RULES.format(date=DATE)
    kind, src = SITES[name]
    if kind == "grounding":
        return event + prompts.EVENT_SCRAPER_GROUNDED_TAIL.format(source=src), dict(grounding=True)
    if kind == "url":
        return event + ts + src, dict(url_context=True)
    return ("What are the latest headlines here released in the past two days? Today is {}. If there are none released "
            "today, then say that there are none released from the news source today or yesterday. And mention the news "
            "source. Do not give anything else. {}".format(DATE, src), dict(url_context=True, mode="standard"))


def main():
    import newscaster.config as config
    config.init()
    config.LLM_AUDIT_LOG_ENABLED = False
    from newscaster.llm import get_llm_response
    luna = {"provider": "openrouter", "model": "openai/gpt-6-luna", "name": "GPT-6 Luna", "reasoning": "low"}
    judge = {"provider": "anthropic", "model": "claude-opus-5-5"}
    rows = []
    for name in SITES:
        text = page_text(name)
        prompt, kw = prompts_for(name)
        gemini = get_llm_response(prompt, **kw)
        # Luna reads the rendered page instead of fetching it itself.
        luna_prompt = re.sub(r"\s*(Source: .*|https?://\S+)\s*$", "", prompt).strip()
        luna_out, _s, lu = call(luna, luna_prompt + "\n\nRENDERED FRONT PAGE TEXT ({}):\n{}".format(SITES[name][1], text[:60000]),
                                "You are an intelligent assistant.")
        gem_first = random.Random(name).random() < 0.5
        a, b = ("gemini", "luna") if gem_first else ("luna", "gemini")
        lists = {"gemini": gemini, "luna": luna_out}
        user = "SCRAPER INSTRUCTIONS:\n{}\n\nFRONT PAGE TEXT (as rendered):\n{}\n\n=== LIST A ===\n{}\n\n=== LIST B ===\n{}".format(
            luna_prompt, text[:60000], lists[a], lists[b])
        jt, _s, ju = call(judge, user, AUDIT)
        v = json.loads(re.search(r"\{.*\}", jt, re.S).group(0))
        row = {"site": name, "mode": SITES[name][0], "page_words": len(text.split()), "gemini": gemini, "luna": luna_out,
               "audit": {a: v.get("A"), b: v.get("B")}, "note": v.get("note"), "judge_cost": call_cost(ju, judge["model"])}
        rows.append(row)
        s = lambda k: {x: len(row["audit"][k].get(x) or []) for x in ("not_on_page", "wrong", "missed")}
        print("{:<10} {:<13} page {:>5} words | gemini {} lines {} | luna {} lines {}".format(
            name, row["mode"], row["page_words"], len([l for l in gemini.splitlines() if l.strip()]), s("gemini"),
            len([l for l in luna_out.splitlines() if l.strip()]), s("luna")))
    out = os.path.join(OUT_DIR, "scrape_check_{}.json".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
    print("wrote", out, "| judge $%.2f" % sum(r["judge_cost"] for r in rows))


if __name__ == "__main__":
    main()
