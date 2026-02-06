import random
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from newscaster.logging import print_and_write


def calmatters_scraper():
    today = datetime.now().strftime('%B %e, %Y')
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_formatted = yesterday.strftime('%B %e, %Y')

    headlines = 'Calmatters, the news source, has released the following headlines today, ' + today + ':\n'
    calmatters_categories = ['politics', 'justice', 'economy', 'education', 'housing', 'environment', 'health']
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CalMattersScraper/1.0)"})

    for category in calmatters_categories:
        print_and_write(category)
        url = f'https://calmatters.org/category/{category}/'
        response = None
        for attempt in range(3):
            try:
                response = session.get(url, timeout=(5, 20))
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                wait = (attempt + 1) * 2
                print_and_write('CalMatters fetch failed', category, str(exc), f'Retrying in {wait}s')
                time.sleep(wait)
        if response is None or not response.ok:
            print_and_write('Skipping CalMatters category after repeated failures', category)
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.find_all('article', class_='archive-stories archive-main-sect')
        if not articles:
            print_and_write('No CalMatters articles found for category', category)
            continue

        for article in articles:
            meta = article.find('div', class_='story-meta')
            if not meta:
                continue
            post_date = meta.get_text(strip=True)
            parts = post_date.split('\u2022')
            if len(parts) < 2:
                continue
            date_string = parts[1].strip()
            headline_tag = article.find('a', class_='story-title')
            if not headline_tag:
                continue
            headline = headline_tag.get_text(strip=True)

            if today in date_string or yesterday_formatted in date_string:
                headlines = headlines + headline + '\n'

        time.sleep(random.uniform(2, 5))

    headlines = headlines + '\n\n'
    session.close()
    return headlines
