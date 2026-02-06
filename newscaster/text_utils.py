import re


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
