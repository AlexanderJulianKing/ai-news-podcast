"""The opinion voice: Gemini TTS (Schedar), read with a style direction (2026-10-23).

Found while auditioning (2026-10-22/23):
- A direction placed before the text is read aloud. It has to sit under "### DIRECTOR'S NOTES",
  with the words under "### TRANSCRIPT".
- Gemini 3.8 returns a whole WAV file: header, samples, then a 6 KB "C2PA" content-credentials
  block. Read as raw samples, the header aired as a click and the C2PA block as a ~125 ms burst of
  noise after every paragraph (Alex heard them). decode() reads only the samples.
- About one paragraph in ten comes out garbled (its transcript matched its text at 0.39-0.67 in
  the rehearsal). Each render is transcribed and compared with its text, and redone if it's off.
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
    return genai.Client(api_key=_config.GOOGLE_GENAI_API_KEY)


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


def _match(client, x, text, rate=_RATE):
    """How closely a transcript of the render matches its text (1.0 = word for word)."""
    from google.genai import types
    reply = client.models.generate_content(model=_config.OPINION_TTS_CHECK_MODEL, contents=[
        types.Part.from_bytes(data=_wav_bytes(x, rate), mime_type="audio/wav"),
        "Transcribe this audio verbatim. Write numbers as words. Output only the transcript."])
    words = lambda s: re.findall(r"[a-z']+", (s or "").lower().replace("’", "'"))
    return difflib.SequenceMatcher(None, words(text), words(reply.text)).ratio()


def speak(text, filename):
    """Render text in the opinion voice to a 48 kHz WAV. Raises if Gemini fails, or if every try at a
    passage comes out garbled, so the caller can fall back to Chirp3-HD."""
    client = _client()
    parts = []
    for chunk in _chunks(text):
        best, best_score = None, -1.0
        for attempt in range(3):
            x, rate = _render(client, chunk)
            try:
                score = _match(client, x, chunk, rate)
            except Exception as exc:
                print_and_write(f"Opinion voice: transcript check skipped ({exc!r:.120})")
                best, best_score = (x, rate), 1.0
                break
            if score > best_score:
                best, best_score = (x, rate), score
            if score >= 0.85:
                break
            print_and_write(f"Opinion voice: transcript matched {score:.2f} on try {attempt + 1} "
                            f"({chunk[:60]!r}...)")
        if best_score < 0.75:
            raise RuntimeError(f"Gemini garbled a passage on every try (best match {best_score:.2f})")
        x, rate = best
        # to 48 kHz by linear interpolation, so these lines splice cleanly with the Chirp3-HD ones
        t = np.arange(int(len(x) * _OUT_RATE / rate)) * (rate / _OUT_RATE)
        parts.append(np.interp(t, np.arange(len(x)), x))
        parts.append(np.zeros(int(0.3 * _OUT_RATE)))
    with open(filename, "wb") as f:
        f.write(_wav_bytes(np.concatenate(parts[:-1]), _OUT_RATE))
