import time
from urllib.parse import urlparse

import newscaster.config as _config
from newscaster.logging import print_and_write
from newscaster.llm import call_with_default


# Hosts that block scraping or only host snippets, not real articles.
_BLOCKED_HOST_SUFFIXES = (
    'facebook.com',
    'fb.com',
    'twitter.com',
    'x.com',
    't.co',
)


def _is_blocked_url(url):
    try:
        host = (urlparse(url).hostname or '').lower()
    except ValueError:
        return False
    if not host:
        return False
    return any(host == suffix or host.endswith('.' + suffix) for suffix in _BLOCKED_HOST_SUFFIXES)


def google_official_search(query, num_results=3, days_prior=1):
    """Return the results of a google search using the official Google API"""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    search_results_list = []

    try:
        for i in range(5):
            api_key = _config.GOOGLE_SEARCH_API_KEY
            custom_search_engine_id = _config.GOOGLE_CSE_ID
            print_and_write('query:', query)

            service = build("customsearch", "v1", developerKey=api_key)

            date_restriction = f'd{max(1, int(days_prior))}'
            result = service.cse().list(q=query, cx=custom_search_engine_id, num=num_results, dateRestrict=date_restriction).execute()

            search_results = result.get("items", [])

            for item in search_results:
                url = item.get("link", "")
                if _is_blocked_url(url):
                    print_and_write('FILTERED social/blocked URL:', url)
                    continue
                search_result_dict = {
                    "headline": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("snippet", "")
                }
                print_and_write(search_result_dict['headline'])
                search_results_list.append(search_result_dict)
            if len(search_results_list) == 0:
                print_and_write('SEARCH BREAK', i)
                query_prompt = 'Please change the following query into a rephrased google search query that can get more relevant results. Please only give the new query and do not give quotation marks'
                query = call_with_default(
                    query, query, system_prompt=query_prompt, mode='light',
                    _log_label='search-query-rephrase',
                )
                query = query.strip("\"").strip('\n')
                time.sleep(5)

            else:
                return search_results_list

    except HttpError as e:
        raise RuntimeError(f"Google search HTTP error: {e}")

    return search_results_list
