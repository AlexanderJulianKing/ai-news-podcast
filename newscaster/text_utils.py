import json
import re

from newscaster.logging import print_and_write, write_jsonl_log

_NOTE_RE = re.compile(r"[ \t]*\[\[\s*NOTE\s*:(.*?)\]\][ \t]*\n?", re.S | re.I)


def extract_producer_notes(text, where=""):
    """Remove [[NOTE: ...]] remarks a writer left for the producers, and log them (logs/producer_notes.jsonl).

    2026-10-24: given "earlier coverage" of a different story, the script writer explained itself in a
    line of the script ("Note: The update context describes ... I've written this as a standalone
    report") and Ethan read it on air. The writers now have this channel for such remarks instead.
    """
    notes = []

    def take(m):
        notes.append(" ".join(m.group(1).split()))
        return ""
    clean = _NOTE_RE.sub(take, text or "")
    for note in notes:
        print_and_write(f"PRODUCER NOTE [{where}]: {note}")
        write_jsonl_log("producer_notes", {"where": where, "note": note})
    return clean.strip()



def text_cleaner(text):
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", " - ")
    text = text.replace("\u2026", "...")

    def _swap_double_quotes(match: re.Match) -> str:
        inner = match.group(1)
        return f"'{inner}'"

    text = re.sub(r'"([^"\n]*?)"', _swap_double_quotes, text)
    if '"' in text:
        text = text.replace('"', "'")
    text = re.sub(r"'{2,}", "'", text)

    text = text.replace('quote quote', 'quote')
    text = text.replace('quote, quote', 'quote')
    text = re.sub('former president donald trump', 'President Donald Trump', text, flags=re.IGNORECASE)
    text = re.sub('former president trump', 'President Trump', text, flags=re.IGNORECASE)
    text = re.sub('former US president trump', 'President Trump', text, flags=re.IGNORECASE)
    text = re.sub('former US president donald trump', 'President Trump', text, flags=re.IGNORECASE)

    text = text.replace("\u200B", "").replace("\u00A0", " ").replace("\u00AD", "")

    # Google Cloud TTS reads "P." as "page" and "A." as a spelled-out letter,
    # so "5 P.M." comes out as "five page M". Normalize the time suffixes.
    text = re.sub(r'\b([PpAa])\.\s*([Mm])\.?', lambda m: m.group(1).upper() + m.group(2).upper(), text)

    return text


def find_quoted_strings(input_string):
    pattern = r'"(.*?)"|\'(.*?)\''
    matches = re.findall(pattern, input_string)
    return [match[0] or match[1] for match in matches]


GROUNDING_FALSE_NEGATIVE_PHRASES = (
    "no indication that",
    "no evidence that",
    "no such announcement",
    "has not issued",
    "there is no indication",
    "there have been no reports",
)


def _grounded_response_needs_retry(response_text: str) -> bool:
    if not response_text:
        return True
    text = response_text.strip()
    if not text:
        return True
    if text.upper().startswith('UNVERIFIED'):
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in GROUNDING_FALSE_NEGATIVE_PHRASES)


def extract_json(text: str, want: type = dict):
    """Extract a JSON value from possibly-fenced LLM output.

    Strips a leading ``` / ```json fence, then parses the substring spanning the
    first opening to the last closing bracket of the wanted kind (``list`` -> [],
    anything else -> {}). Raises ``ValueError`` when no such bracket span exists;
    ``json.JSONDecodeError`` propagates when the span is not valid JSON. Callers
    decide how to handle failure (return [] / None) and check the parsed type.
    """
    opener, closer = ("[", "]") if want is list else ("{", "}")
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find(opener)
    end = raw.rfind(closer)
    if start == -1 or end == -1 or end < start:
        kind = "array" if want is list else "object"
        raise ValueError(f"text did not contain a JSON {kind}")
    return json.loads(raw[start:end + 1])
