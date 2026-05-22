"""Tests for newscaster.scrapers.web — article body extraction and scrape_text."""
from unittest.mock import patch, MagicMock

from bs4 import BeautifulSoup

from newscaster.scrapers.web import _extract_article_body, scrape_text


def _soup(html):
    return BeautifulSoup(html, "html.parser")


def test_extract_article_body_finds_article_tag():
    """When the page has an <article> element, its text is returned."""
    html = """
    <html><body>
      <nav>Nav menu junk</nav>
      <article>
        <h1>Real Headline</h1>
        <p>This is the article body that we care about.</p>
      </article>
      <footer>Footer junk</footer>
    </body></html>
    """
    body = _extract_article_body(_soup(html))
    assert body is not None
    assert "Real Headline" in body
    assert "article body that we care about" in body
    assert "Nav menu junk" not in body
    assert "Footer junk" not in body


def test_extract_article_body_falls_back_to_main():
    """When there is no <article>, the <main> element is used."""
    html = """
    <html><body>
      <nav>Nav menu junk</nav>
      <main>
        <p>Main content paragraph one.</p>
        <p>Main content paragraph two.</p>
      </main>
    </body></html>
    """
    body = _extract_article_body(_soup(html))
    assert body is not None
    assert "Main content paragraph one" in body
    assert "Nav menu junk" not in body


def test_extract_article_body_falls_back_to_role_main():
    """When neither <article> nor <main>, an element with role=main is used."""
    html = """
    <html><body>
      <nav>Nav menu junk</nav>
      <div role="main">
        <p>Role-main content here that is substantial enough to count as a real
           article body and not get rejected as a near-empty container.</p>
      </div>
    </body></html>
    """
    body = _extract_article_body(_soup(html))
    assert body is not None
    assert "Role-main content here" in body
    assert "Nav menu junk" not in body


def test_extract_article_body_uses_entry_content_class():
    """When no semantic tags exist, common WordPress class names are tried."""
    html = """
    <html><body>
      <nav>Nav menu junk</nav>
      <div class="entry-content">
        <p>WordPress-style article body with enough text to clear the minimum
           length threshold for a real article container.</p>
      </div>
    </body></html>
    """
    body = _extract_article_body(_soup(html))
    assert body is not None
    assert "WordPress-style article body" in body
    assert "Nav menu junk" not in body


def test_extract_article_body_returns_none_when_no_match():
    """When no recognized container is present, returns None to signal fallback."""
    html = """
    <html><body>
      <nav>just a nav</nav>
      <div>some unrelated content</div>
    </body></html>
    """
    body = _extract_article_body(_soup(html))
    assert body is None


def test_extract_article_body_skips_empty_article():
    """An <article> tag with only whitespace should not count as a match."""
    html = """
    <html><body>
      <article>   </article>
      <main>
        <p>This is the real content of the page, with enough words to clear
           the minimum-length threshold and beat the empty article tag.</p>
      </main>
    </body></html>
    """
    body = _extract_article_body(_soup(html))
    assert body is not None
    assert "This is the real content" in body


def test_scrape_text_uses_article_body_when_present():
    """scrape_text returns only the article body when one is found."""
    html = """
    <html><body>
      <nav><a href="/donate">Donate</a><a href="/about">About Us</a></nav>
      <header>Site Title</header>
      <article>
        <h1>Real Headline</h1>
        <p>This is the article body.</p>
      </article>
      <footer>Copyright junk</footer>
    </body></html>
    """
    response = MagicMock(status_code=200, text=html)
    with patch("newscaster.scrapers.web.requests.get", return_value=response):
        text = scrape_text("https://example.com/article")
    assert "Real Headline" in text
    assert "This is the article body" in text
    assert "Donate" not in text
    assert "About Us" not in text
    assert "Copyright junk" not in text


def test_scrape_text_falls_back_to_whole_page_when_no_article_tag():
    """When no semantic article container exists, the whole page text is returned."""
    html = """
    <html><body>
      <div class="weird-custom-layout">
        <p>Article-ish content with no standard container.</p>
      </div>
    </body></html>
    """
    response = MagicMock(status_code=200, text=html)
    with patch("newscaster.scrapers.web.requests.get", return_value=response):
        text = scrape_text("https://example.com/weird")
    assert "Article-ish content with no standard container" in text


def test_scrape_text_strips_script_and_style():
    """Scripts and styles are removed regardless of whether article body is used."""
    html = """
    <html><body>
      <script>var leaked = true;</script>
      <style>.x { color: red; }</style>
      <article>
        <p>The article text.</p>
      </article>
    </body></html>
    """
    response = MagicMock(status_code=200, text=html)
    with patch("newscaster.scrapers.web.requests.get", return_value=response):
        text = scrape_text("https://example.com/with-scripts")
    assert "The article text" in text
    assert "leaked" not in text
    assert "color: red" not in text


def test_scrape_text_truncates_when_fallback_text_exceeds_max_chars():
    """When no article container is found and the whole-page text exceeds
    max_chars, the result is truncated rather than left huge or rejected."""
    huge_body = "blob " * 10000  # 50,000 chars, no recognized container
    html = f"<html><body><div>{huge_body}</div></body></html>"
    response = MagicMock(status_code=200, text=html)
    with patch("newscaster.scrapers.web.requests.get", return_value=response):
        text = scrape_text("https://example.com/huge", max_chars=20000)
    assert len(text) == 20000
    assert text.startswith("blob")


def test_scrape_text_does_not_truncate_short_pages():
    """Pages under max_chars are returned in full."""
    html = "<html><body><article><p>Short article content here, well under 20K chars.</p></article></body></html>"
    response = MagicMock(status_code=200, text=html)
    with patch("newscaster.scrapers.web.requests.get", return_value=response):
        text = scrape_text("https://example.com/short", max_chars=20000)
    assert len(text) < 200
    assert "Short article content" in text


def test_scrape_text_returns_error_on_bad_url():
    """Non-http URLs return the Error: sentinel string (existing contract)."""
    text = scrape_text("not-a-url")
    assert text.startswith("Error: ")


def test_scrape_text_returns_error_on_http_failure():
    """HTTP 4xx/5xx responses return the Error: sentinel string."""
    response = MagicMock(status_code=404)
    with patch("newscaster.scrapers.web.requests.get", return_value=response):
        text = scrape_text("https://example.com/missing")
    assert text.startswith("Error: ")
    assert "404" in text
