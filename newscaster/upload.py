import os
import random
import sys
import time
from datetime import date
from typing import Dict

import google.auth.exceptions
import google_auth_httplib2
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import httplib2
import argparse

from newscaster.llm import get_llm_response
from newscaster.logging import print_and_write

httplib2.RETRIES = 1

MAX_RETRIES = 10
UPLOAD_HTTP_TIMEOUT_SECONDS = 180

RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError, ConnectionResetError,
                        ConnectionAbortedError, ConnectionRefusedError)

RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

MISSING_CLIENT_SECRETS_MESSAGE = f"""
WARNING: Please configure OAuth 2.0

To make this sample run you will need to populate the client_secrets.json file
found at:

   {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', CLIENT_SECRETS_FILE))}

with information from the API Console
https://console.cloud.google.com/
"""

VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")

YOUTUBE_TITLE_LIMIT = 100  # YouTube rejects titles longer than this (HTTP 400 invalidTitle)


def get_authenticated_service():
    creds = None
    token_file = "uploader2-token.json"

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except google.auth.exceptions.RefreshError:
                creds = None
        if not creds:
            if not os.path.exists(CLIENT_SECRETS_FILE):
                sys.exit(MISSING_CLIENT_SECRETS_MESSAGE)
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, 'w') as token:
            token.write(creds.to_json())

    authed_http = google_auth_httplib2.AuthorizedHttp(
        creds, http=httplib2.Http(timeout=UPLOAD_HTTP_TIMEOUT_SECONDS)
    )
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, http=authed_http)


def fit_title_to_limit(title, limit=YOUTUBE_TITLE_LIMIT, max_attempts=3):
    """Keep a YouTube title within `limit` characters.

    YouTube rejects titles over 100 chars (HTTP 400 invalidTitle), and a failed
    upload is otherwise silent, so a long lead headline can quietly stop an
    episode from airing. If the built title is too long, ask the LLM for a
    shorter one, showing the too-long title as an example of what to avoid, and
    retry a few times. As a last resort, hard-truncate at a word boundary so the
    episode still airs.
    """
    if len(title) <= limit:
        return title

    too_long = title
    for _ in range(max_attempts):
        prompt = (
            f"A YouTube video title must be {limit} characters or fewer. "
            f"This title is too long at {len(too_long)} characters:\n\"{too_long}\"\n\n"
            f"Write a new title for the same news episode that is {limit} characters or fewer. "
            f"Keep it accurate and keep the leading date if it still fits. "
            f"Return only the new title, with no quotes or extra text."
        )
        try:
            candidate = get_llm_response(
                prompt, system_prompt="You are a concise news headline editor.", mode="light")
        except Exception as exc:
            print_and_write(f"Title-shortening LLM call failed ({exc}); falling back to truncation.")
            break
        candidate = (candidate or "").strip()
        if candidate:
            candidate = candidate.splitlines()[0].strip().strip('"').strip("'").strip()
        if candidate and len(candidate) <= limit:
            print_and_write(
                f"Title was {len(title)} chars (over {limit}); shortened to {len(candidate)}: {candidate!r}")
            return candidate
        if candidate:
            too_long = candidate  # show the latest still-too-long attempt on the next try

    cut = title[:limit].rsplit(" ", 1)[0].rstrip() or title[:limit]
    print_and_write(f"Could not shorten title under {limit} via LLM; truncating to: {cut!r}")
    return cut


def initialize_upload(youtube, options):
    tags = None
    if options.keywords:
        tags = options.keywords.split(",")

    today = date.today()
    formatted_date = today.strftime("%B %d, %Y")
    formatted_date2 = today.strftime("%Y_%m_%d")

    episode_file = f'episode_titles/{formatted_date2}.txt'
    if not os.path.exists(episode_file):
        sys.exit(f"Episode titles file not found: {episode_file}")

    titles = read_episode_titles(episode_file).split(',')

    if not titles:
        sys.exit("No titles found in the episode titles file.")

    first_title = titles[0].strip()
    real_title = f"{formatted_date} - {first_title}"
    real_title = fit_title_to_limit(real_title)
    print(titles)
    print('real_title', real_title, type(real_title))

    titles_content = read_episode_titles(episode_file)

    real_description = (
        f"{titles_content}.\nDaily News. Always ad-free. \nMade by Alex using Claude Opus 5.5, GPT-6 Luna, GPT-6 Sol, Gemini, and Google Cloud Text to speech. \n\n"
        "I made this podcast to give me the news each morning as I get ready for the day. I wanted to stay informed about the world, but there are a lot of stories highlighted by media outlets that aren\'t important to the world, relevant to my life, or interesting. In the interest of making money, these media outlets will sensationalize stories and amplify negativity, which is what keeps eyeballs and therefore ad revenue.\n"
        "So I made a news podcast that circumvents these perverse incentives and distills out the important stories. \n"
        "Every day, an LLM will pick out news stories: one it deems the most important to the american people, and one it deems the most important to the average Californian. Additionally, it briefly goes over other stories that may be of general interest. \n\n"
    )
    print('real_description', type(real_description), real_description)

    tags_prompt = "Based on this description, please make 6 tags for a YouTube video with a mix of broad tags and specific tags. For example: Broad tags: 'fitness', 'workouts' Specific tags: 'how to do a pushup', 'proper pushup form'. Only give the tags and separate by comma.\n\nDescription: " + real_description
    tags = get_llm_response(tags_prompt, system_prompt='You are an intelligent assistant.', mode='light')
    tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

    body = {
        "snippet": {
            "title": real_title,
            "description": real_description,
            "tags": tags,
            "categoryId": options.category
        },
        "status": {
            "privacyStatus": options.privacyStatus
        }
    }
    print(body)
    print('about to insert_request')
    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=MediaFileUpload(options.file, chunksize=-1, resumable=True)
    )

    video_id = resumable_upload(insert_request)
    write_upload_marker(formatted_date2, video_id)
    return video_id


def write_upload_marker(formatted_date, video_id, directory="output_scripts"):
    """Atomically record a completed upload so the scheduler can verify it."""
    os.makedirs(directory, exist_ok=True)
    marker = os.path.join(directory, f"{formatted_date}_UPLOAD_COMPLETE.flag")
    temporary = f"{marker}.tmp.{os.getpid()}"
    with open(temporary, "w") as outfile:
        outfile.write(f"{video_id}\n")
    os.replace(temporary, marker)
    print(f"Upload completion marker written: {marker}")
    return marker


def resumable_upload(insert_request):
    response = None
    retry = 0
    print(insert_request)

    print("Inspecting insert_request:")
    print("  Request URL:", insert_request.uri)
    print("  Request Headers:", insert_request.headers)
    print("  Request Body:", insert_request.body)
    while response is None:
        error = None
        try:
            print("Uploading file...")
            print('inserting chunk')
            status, response = insert_request.next_chunk()
            print('inserted chunk')
            if response:
                if 'id' in response:
                    print(f"Video id '{response['id']}' was successfully uploaded.")
                    return response['id']
                else:
                    raise RuntimeError(f"The upload failed with an unexpected response: {response}")
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
            else:
                raise
        except RETRIABLE_EXCEPTIONS as e:
            error = f"A retriable error occurred: {e}"

        if error:
            print(error)
            retry += 1
            if retry > MAX_RETRIES:
                sys.exit("No longer attempting to retry.")

            max_sleep = 2 ** retry
            sleep_seconds = random.random() * max_sleep
            print(f"Sleeping {sleep_seconds:.2f} seconds and then retrying...")
            time.sleep(sleep_seconds)


def read_episode_titles(path):
    """The day's episode titles, always read as UTF-8.

    The Pi's system locale is en_US (Latin-1), so a plain open() turned the curly
    apostrophe in "Iran\u2019s" into "Iran\u00e2" plus two invisible characters in
    the YouTube titles of 2026-09-24 and 09-25.
    """
    with open(path, 'r', encoding='utf-8') as infile:
        return infile.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Video file to upload")
    parser.add_argument("--title", help="Video title", default="Test Title")
    parser.add_argument("--description", help="Video description", default="Test Description")
    parser.add_argument("--category", default="22",
                        help="Numeric video category. See https://developers.google.com/youtube/v3/docs/videoCategories/list")
    parser.add_argument("--keywords", help="Video keywords, comma separated", default="")
    parser.add_argument("--privacyStatus", choices=VALID_PRIVACY_STATUSES,
                        default=VALID_PRIVACY_STATUSES[0], help="Video privacy status.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        sys.exit("Please specify a valid file using the --file= parameter.")

    youtube = get_authenticated_service()
    try:
        initialize_upload(youtube, args)
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}", file=sys.stderr)
        raise SystemExit(1)
