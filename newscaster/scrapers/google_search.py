import time
from datetime import datetime, timedelta

from newscaster.config import GOOGLE_SEARCH_API_KEY, GOOGLE_CSE_ID
from newscaster.logging import print_and_write


def google_official_search(query, num_results=3, days_prior=1):
    """Return the results of a google search using the official Google API"""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import json

    search_results_list = []
    query = query
    query_record = query

    try:
        for i in range(5):
            api_key = GOOGLE_SEARCH_API_KEY
            custom_search_engine_id = GOOGLE_CSE_ID
            print_and_write('query:', query)

            service = build("customsearch", "v1", developerKey=api_key)

            today = datetime.today()
            past_day = today - timedelta(days=days_prior)
            date_restriction = f"date:r:{past_day.isoformat()}"
            date_restriction = 'd1'
            result = service.cse().list(q=query, cx=custom_search_engine_id, num=num_results, dateRestrict=date_restriction).execute()

            search_results = result.get("items", [])

            for item in search_results:
                search_result_dict = {
                    "headline": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                }
                print_and_write(search_result_dict['headline'])
                search_results_list.append(search_result_dict)
            if len(search_results_list) == 0:
                print_and_write('SEARCH BREAK', i)
                query_prompt = 'Please change the following query into a rephrased google search query that can get more relevant results. Please only give the new query and do not give quotation marks'
                query = query_prompt(query, system_prompt=query_prompt)
                query = query.strip("\"").strip('\n')
                time.sleep(5)

            else:
                return search_results_list

    except HttpError as e:
        error_details = json.loads(e.content.decode())

        if error_details.get("error", {}).get("code") == 403 and "invalid API key" in error_details.get("error", {}).get("message", ""):
            return "Error: The provided Google API key is invalid or missing."
        else:
            return f"Error: {e}"

    return search_results_list
