import requests
from bs4 import BeautifulSoup

from newscaster.logging import print_and_write


def scrape_text(url):
    """Scrape text from a webpage"""
    if not url.startswith('http'):
        return "Error: Invalid URL"

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}, timeout=10)
    except requests.exceptions.RequestException as e:
        return "Error: " + str(e)
    except requests.exceptions.Timeout:
        print_and_write("The request timed out.")

    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text
