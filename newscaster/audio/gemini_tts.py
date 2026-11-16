"""The opinion voice: Gemini TTS (Schedar), read with a style direction (2026-10-23).

Found while auditioning (2026-10-22/23):
- A direction placed before the text is read aloud. It has to sit under "### DIRECTOR'S NOTES",
  with the words under "### TRANSCRIPT".
- Gemini 3.8 returns a whole WAV file: header, samples, then a 6 KB "C2PA" content-credentials
  block. Read as raw samples, the header aired as a click and the C2PA block as a ~125 ms burst of
  noise after every paragraph (Alex heard them). decode() reads only the samples.
- About one paragraph in ten comes out garbled (its transcript matched its text at 0.39-0.67 in
  the rehearsal). Each render is transcribed and compared with its text, and redone if it's off.
- 2026-11-15: the voice also ad-libs. "So, Watanabe filed a complaint." aired after a paragraph that
  ends "stop being a cry baby"; a whole-text match of 0.91 passed it. In 24 test renders, 3 added a
  sentence (match 0.85-0.87, all passing) and 4 read the paragraph twice (0.63-0.67). So a render
  that adds 3 or more words in one place is rejected too. Numbers are left out of the comparison,
  since the transcriber writes "2024" for "twenty twenty-four".
"""
import difflib
import io
import re
import time
import wave

import numpy as np

from newscaster import config as _config
from newscaster.logging import print_and_write

_RATE = 24000          # Gemini returns 16-bit mono PCM at 24 kHz
_OUT_RATE = 48000      # what the Chirp3-HD lines use
_MAX_BYTES = 3500      # the API takes at most 4,000 bytes of text per request
_MIN_DIRECTED_CHARS = 200


def decode(data, mime_type=""):
    """(samples as floats in [-1, 1], sample rate) from Gemini's audio: a WAV file, or raw 16-bit PCM."""
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data)) as w:            # reads the "data" chunk only
            rate, frames = w.getframerate(), w.readframes(w.getnframes())
    else:
        m = re.search(r"rate=(\d+)", mime_type or "")
        rate, frames = (int(m.group(1)) if m else _RATE), data
    return np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768, rate


def _chunks(text):
    if len(text.encode("utf-8")) <= _MAX_BYTES:
        return [text]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if cur and len((cur + " " + sentence).encode("utf-8")) > _MAX_BYTES:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    return out + ([cur] if cur else [])


def _client():
    from google import genai
    from newscaster.llm.google_client import client_kwargs
    return genai.Client(**client_kwargs())


def _render(client, text):
    from google.genai import types
    # On a very short line the notes outweigh the text and get read aloud ("Vote no on Proposition
    # thirty-seven." came back as 19 seconds, notes included, on 3 of 3 tries); those lines go plain.
    if len(text) < _MIN_DIRECTED_CHARS:
        contents = text
    else:
        contents = f"### DIRECTOR'S NOTES\n{_config.OPINION_TTS_DIRECTION}\n\n### TRANSCRIPT\n{text}"
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=_config.OPINION_VOICE))))
    last = None
    for attempt in range(3):
        try:
            r = client.models.generate_content(model=_config.OPINION_TTS_MODEL, contents=contents, config=config)
            audio = r.candidates[0].content.parts[0].inline_data
            um = getattr(r, "usage_metadata", None)
            from newscaster import costs
            costs.record("tts", "Gemini TTS", (getattr(um, "candidates_token_count", 0) or 0)
                         * getattr(_config, "OPINION_TTS_USD_PER_M_AUDIO_TOKENS", 6.0) / 1e6)
            if audio.data:
                return decode(audio.data, audio.mime_type)
        except Exception as exc:
            last = exc
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini TTS failed: {last!r:.200}")


def _wav_bytes(x, sr):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


_NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                    "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                    "hundred thousand million billion trillion point percent oh".split())


def _words(s):
    return [w for w in re.findall(r"[a-z0-9']+", (s or "").lower().replace("’", "'"))
            if w not in _NUMBER_WORDS and not re.fullmatch(r"\d+(?:st|nd|rd|th|s)?", w)]


def check_transcript(text, heard):
    """(ok, match, added words) for a transcript of a render. ok: it matches the text at 0.85 or better
    (1.0 = word for word) and adds no run of 3+ words the text doesn't have."""
    a, b = _words(text), _words(heard)
    sm = difflib.SequenceMatcher(None, a, b)
    # the longest stretch the transcript has beyond the text (a replace counts only its surplus words)
    extra, added = 0, []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("insert", "replace") and (j2 - j1) - (i2 - i1) > extra:
            extra, added = (j2 - j1) - (i2 - i1), b[j1:j2]
    return sm.ratio() >= 0.85 and extra < 3, sm.ratio(), added if extra >= 3 else []


def _transcribe(client, x, rate=_RATE):
    from google.genai import types
    for attempt in range(3):
        try:
            return client.models.generate_content(model=_config.OPINION_TTS_CHECK_MODEL, contents=[
                types.Part.from_bytes(data=_wav_bytes(x, rate), mime_type="audio/wav"),
                "Transcribe this audio verbatim. Write numbers as words. Output only the transcript."]).text
        except Exception as exc:     # Vertex AI answers 429 when busy (2026-11-15); wait and try again
            if attempt == 2 or "429" not in str(exc):
                raise
            time.sleep(15 * (attempt + 1))


def speak(text, filename):
    """Render text in the opinion voice to a 48 kHz WAV. Raises if Gemini fails, or if every try at a
    passage comes out garbled, so the caller can fall back to Chirp3-HD."""
    client = _client()
    parts = []
    for chunk in _chunks(text):
        best = None
        for attempt in range(4):
            x, rate = _render(client, chunk)
            try:
                heard = _transcribe(client, x, rate)
            except Exception as exc:
                print_and_write(f"Opinion voice: transcript check skipped ({exc!r:.120})")
                best = (x, rate)
                break
            ok, score, added = check_transcript(chunk, heard)
            if ok:
                best = (x, rate)
                break
            print_and_write(f"Opinion voice: try {attempt + 1} rejected, match {score:.2f}"
                            + (f", added {' '.join(added[:12])!r}" if added else "") + f" ({chunk[:60]!r}...)")
        if best is None:
            raise RuntimeError("Gemini misread a passage on every try")
        x, rate = best
        # to 48 kHz by linear interpolation, so these lines splice cleanly with the Chirp3-HD ones
        t = np.arange(int(len(x) * _OUT_RATE / rate)) * (rate / _OUT_RATE)
        parts.append(np.interp(t, np.arange(len(x)), x))
        parts.append(np.zeros(int(0.3 * _OUT_RATE)))
    with open(filename, "wb") as f:
        f.write(_wav_bytes(np.concatenate(parts[:-1]), _OUT_RATE))
