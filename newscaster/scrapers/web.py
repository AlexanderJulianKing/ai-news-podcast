import requests
from bs4 import BeautifulSoup

from newscaster.logging import print_and_write


# Selectors to try in order when looking for the article body. The first match
# whose extracted text is "substantial" (more than _MIN_BODY_CHARS) wins.
_ARTICLE_BODY_SELECTORS = (
    "article",
    "main",
    "[role=main]",
    ".entry-content",
    ".post-content",
    ".article-content",
    ".article-body",
    ".story-body",
)

_MIN_BODY_CHARS = 50


def _normalize_text(raw):
    """Collapse whitespace the same way the rest of the codebase does."""
    lines = (line.strip() for line in raw.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)


def _extract_article_body(soup):
    """Try common article-container patterns and return the first substantial match.

    Returns the normalized text of the first matching element whose text is
    longer than _MIN_BODY_CHARS, or None if no recognized container has enough
    content to be considered the article body.
    """
    for selector in _ARTICLE_BODY_SELECTORS:
        for element in soup.select(selector):
            text = _normalize_text(element.get_text())
            if len(text) >= _MIN_BODY_CHARS:
                return text
    return None


def scrape_text(url, max_chars=20000):
    """Scrape text from a webpage.

    Strips <script>/<style>/<nav>/<header>/<footer>/<aside> chrome, then tries
    to return only the article body (via <article>, <main>, etc.) if one can
    be identified. Falls back to the whole-document text if no recognized
    container is found. The returned text is truncated to ``max_chars`` so
    callers don't pass unbounded blobs to the LLM.
    """
    if not url.startswith('http'):
        return "Error: Invalid URL"

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}, timeout=10)
    except requests.exceptions.RequestException as e:
        return "Error: " + str(e)

    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove chrome elements unconditionally — even when we fall back to the
    # whole document we don't want nav menus and ads stuffed into the LLM.
    for chrome_tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        chrome_tag.extract()

    article_text = _extract_article_body(soup)
    text = article_text if article_text is not None else _normalize_text(soup.get_text())

    if len(text) > max_chars:
        print_and_write(
            f'scrape_text: truncating {url} from {len(text)} to {max_chars} chars'
        )
        text = text[:max_chars]

    return text
