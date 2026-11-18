import os
import struct
import time
import wave

from google.cloud import texttospeech
from pydub import AudioSegment

from newscaster import config as _config
from newscaster.audio.loudness import measure_wav, plan_gain
from newscaster.config import _SECOND
from newscaster.logging import print_and_write
from newscaster.text_utils import text_cleaner
from newscaster.llm import get_llm_response

_CLIP_THRESHOLD = 0.99   # fraction of max sample value
_CLIP_PCT_LIMIT = 0.05   # flag if more than 0.05% of samples are clipped
_MAX_REGEN_ATTEMPTS = 2   # retry TTS this many times before normalizing


def check_clipping(filename):
    """Return (is_clipped, clip_pct) for a 16-bit WAV file."""
    with wave.open(filename, 'r') as w:
        frames = w.readframes(w.getnframes())
        sampwidth = w.getsampwidth()
    if sampwidth != 2:
        return False, 0.0
    n_samples = len(frames) // 2
    if n_samples == 0:
        return False, 0.0
    samples = struct.unpack('<' + 'h' * n_samples, frames)
    max_val = 32767
    threshold = _CLIP_THRESHOLD * max_val
    clipped = sum(1 for s in samples if abs(s) >= threshold)
    clip_pct = clipped / n_samples * 100
    return clip_pct > _CLIP_PCT_LIMIT, clip_pct


_BURST_MAX_S = 0.6        # a glitch longer than this is left to the gain fallback
_BURST_OVER_SPEECH_DB = 12
_BURST_MIN_FLATNESS = 0.15   # static is spectrally flat; voiced speech sits near 0.02-0.05


def silence_burst(filename):
    """Mute a short burst of clipped static, if that is what the clipping is.

    On 2026-11-17 (pm) Chirp rendered 0.3 s of full-scale static before one of Ethan's lines on all
    three tries, and turning the whole line down 3 dB left it on air, 20 dB above his voice. Silences
    the clipped stretch (widened to where the level falls back toward speech) only when it is short,
    far louder than the line's speech, and noise-like rather than voiced. Returns (start_s, end_s)
    of the muted stretch, or None when the file was left alone."""
    import numpy as np
    with wave.open(filename, 'r') as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            return None
        sr = w.getframerate()
        params = w.getparams()
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64) / 32768
    hop = int(0.05 * sr)
    n = len(x) // hop
    if n < 10:
        return None
    clipped = np.where(np.abs(x) >= _CLIP_THRESHOLD)[0]
    if len(clipped) == 0:
        return None
    level = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    voiced = level[level > -45]
    if len(voiced) == 0:
        return None
    speech = float(np.median(voiced))
    a, b = clipped[0] // hop, min(clipped[-1] // hop, n - 1)
    while a > 0 and level[a - 1] > speech + 6:
        a -= 1
    while b < n - 1 and level[b + 1] > speech + 6:
        b += 1
    start, end = a * hop, (b + 1) * hop
    if (end - start) / sr > _BURST_MAX_S or level[a:b + 1].max() < speech + _BURST_OVER_SPEECH_DB:
        return None
    spectrum = np.abs(np.fft.rfft(x[start:end] * np.hanning(end - start))) + 1e-12
    if np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum) < _BURST_MIN_FLATNESS:
        return None
    fade = min(int(0.01 * sr), start, len(x) - end)
    y = x.copy()
    y[start:end] = 0
    if fade:
        y[start - fade:start] *= np.linspace(1, 0, fade)
        y[end:end + fade] *= np.linspace(0, 1, fade)
    with wave.open(filename, 'w') as w:
        w.setparams(params)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype('<i2').tobytes())
    return start / sr, end / sr


def normalize_audio(filename, headroom_db=3.0):
    """Reduce peak level of a WAV file to avoid clipping."""
    audio = AudioSegment.from_wav(filename)
    peak_db = audio.max_dBFS
    if peak_db > -headroom_db:
        reduction = peak_db + headroom_db
        audio = audio - reduction
        audio.export(filename, format="wav")
        print_and_write(f"Normalized {filename} by -{reduction:.1f} dB")
    return filename


def normalize_loudness(filename):
    """Pull a WAV to the configured loudness target, within the peak ceiling.

    Runs on every synthesised line so the voices match each other. The gain is
    clamped two ways: never more than LOUDNESS_MAX_GAIN_DB in either direction,
    so a degenerate render is not amplified into the mix, and never enough to
    push the peak above LOUDNESS_PEAK_CEILING_DB. Returns the gain applied in
    dB, or None when nothing was changed.
    """
    if not getattr(_config, 'LOUDNESS_NORMALIZE_ENABLED', False):
        return None
    try:
        loudness, peak_db = measure_wav(filename)
    except Exception as e:
        print_and_write(f"Loudness measurement failed for {filename}: {e}; leaving level alone")
        return None
    if loudness is None:
        print_and_write(f"Loudness not measurable for {filename}; leaving level alone")
        return None

    gain = plan_gain(
        loudness,
        peak_db,
        getattr(_config, 'LOUDNESS_TARGET_LUFS', -23.0),
        getattr(_config, 'LOUDNESS_PEAK_CEILING_DB', -1.0),
        getattr(_config, 'LOUDNESS_MAX_GAIN_DB', 12.0),
    )
    if gain is None or abs(gain) < getattr(_config, 'LOUDNESS_MIN_GAIN_DB', 0.1):
        return None

    try:
        audio = AudioSegment.from_wav(filename)
        audio = audio + gain
        audio.export(filename, format="wav")
    except Exception as e:
        print_and_write(f"Loudness normalization failed for {filename}: {e}; leaving level alone")
        return None

    print_and_write(
        f"Loudness {filename}: {loudness:.1f} LUFS -> target, applied {gain:+.1f} dB")
    return gain


def google_speak(name, text, filename):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "newscaster1-03dc16232821.json"

    text = text_cleaner(text)

    if name == 'Opinion':
        # The Sunday opinion voice is Gemini TTS; Chirp3-HD Schedar steps in if it fails.
        try:
            from newscaster.audio import gemini_tts
            gemini_tts.speak(text, filename)
            normalize_loudness(filename)
            return
        except Exception as e:
            print_and_write(f"Opinion voice (Gemini) failed: {e!r:.200}; using {_config.OPINION_FALLBACK_VOICE}")

    client = texttospeech.TextToSpeechClient()
    if name == 'Elias':
        robo_name = "en-US-Studio-M"
    elif name == 'Chloe':
        robo_name = 'en-US-Chirp3-HD-Leda'
    elif name == 'Grace' or name == 'Grace2':
        robo_name = 'en-US-Chirp3-HD-Aoede'
    elif name == 'Ethan':
        robo_name = 'en-US-Chirp3-HD-Fenrir'
    elif name == 'Opinion':
        robo_name = _config.OPINION_FALLBACK_VOICE

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=robo_name
    )

    if name == 'Grace2':
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000)
    elif name == 'Grace':
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000
        )
    else:
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000
        )
    complete = False
    i = 1
    while complete == False:
        try:
            print_and_write('speaking')
            response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
            print_and_write(voice, text)
            complete = True
        except Exception as e:
            if i % 3 == 0:
                lower_than_5000 = False
                while lower_than_5000 == False:
                    print_and_write('trying to say', text)
                    text = get_llm_response(text, system_prompt='Please modify the text to fit these requirements: \n1: The Entirety of the text is no greater than 5000 characters. \n2: Each individual sentence is no greater than 700 characters long. \n3: Do not refer to Donald Trump as the former president; he is the current president. \n Give no other text.', mode='light')
                    text = text_cleaner(text)
                    while 'quote quote' in text:
                        text = text.replace('quote quote', 'quote')
                    while 'quote, quote' in text:
                        text = text.replace('quote, quote', 'quote')
                    if len(text) < 5000:
                        input_text = texttospeech.SynthesisInput(text=text)
                        lower_than_5000 = True

            i += 1
            time.sleep(10 * i)
            print_and_write('failure, wating', str(10 * i), 'seconds:', e)

    with open(filename, "wb") as out_file:
        out_file.write(response.audio_content)
    from newscaster import costs
    costs.record("tts", "Chirp3-HD", len(text) * 30e-6, units=len(text), unit="chars")

    # --- clipping detection with retry ---
    is_clipped, clip_pct = check_clipping(filename)
    if is_clipped:
        print_and_write(f"WARNING: clipping detected in {filename} ({clip_pct:.2f}% samples clipped)")
        for attempt in range(1, _MAX_REGEN_ATTEMPTS + 1):
            print_and_write(f"Re-synthesizing {filename} (attempt {attempt}/{_MAX_REGEN_ATTEMPTS})")
            try:
                response = client.synthesize_speech(
                    input=input_text, voice=voice, audio_config=audio_config)
                with open(filename, "wb") as out_file:
                    out_file.write(response.audio_content)
                is_clipped, clip_pct = check_clipping(filename)
                if not is_clipped:
                    print_and_write(f"Re-synthesis resolved clipping in {filename}")
                    break
            except Exception as e:
                print_and_write(f"Re-synthesis attempt {attempt} failed: {e}")
                time.sleep(5)
        if is_clipped:
            burst = silence_burst(filename)
            if burst:
                print_and_write(f"Silenced a burst of static at {burst[0]:.2f}-{burst[1]:.2f}s in {filename}")
                is_clipped, clip_pct = check_clipping(filename)
        if is_clipped:
            print_and_write(f"Clipping persists after retries, normalizing {filename}")
            normalize_audio(filename)

    # --- loudness match across voices (last, so it sees the final waveform) ---
    normalize_loudness(filename)

    return


def split_speaker(line, voices_list, current, strict=False):
    """("Ethan: Good morning." -> ("Ethan", "Good morning.")). A line without a known speaker keeps
    the current one and all of its words, or, when strict, returns None: segment scripts are always
    labeled, and on 2026-10-24 an unlabeled writer's note ("Note: The update context ...") went to
    Ethan's voice. Before 2026-10-23 the line was split on every colon and anything after the second
    was dropped, so "a name only Congress could love: PASTEUR" lost words."""
    if ':' in line:
        head, rest = line.split(':', 1)
        if head.strip().capitalize() in voices_list:
            return head.strip().capitalize(), rest.strip()
    return None if strict else (current, line.strip())


def text2speech(date_str, voices_list):
    voices_list = list(voices_list) + ['Connie', 'Opinion']
    OUTPUT_PATH = 'segment_audio/' + date_str

    speaker = 'Grace'
    for i in range(4):
        if os.path.exists('output_scripts/{}_segment_{}.txt'.format(date_str, i)):
            if os.path.exists('{}_segment_{}.mp3'.format(OUTPUT_PATH, i)):
                print_and_write(f"The file '{'{}_segment_{}.mp3'.format(OUTPUT_PATH, i)}' exists.")
            else:
                print_and_write(f"The file '{'{}_segment_{}.mp3'.format(OUTPUT_PATH, i)}' does not exist.")
                with open('output_scripts/{}_segment_{}.txt'.format(date_str, i), 'r', encoding='utf-8') as segmentfile:
                    preline_list = segmentfile.read().split('\n')
                    line_list = []
                    for line in preline_list:
                        if len(line) > 3:
                            line_list.append(line)
                    print_and_write(line_list)
                    segment_audio = 'coolio'
                    for j in range(len(line_list)):
                        thing = line_list[j]
                        print_and_write(thing)

                        labeled = split_speaker(thing, voices_list, speaker, strict=True)
                        if labeled is None:
                            print_and_write(f"TTS: not voicing a line with no speaker label: {thing[:160]!r}")
                            continue
                        speaker, text = labeled

                        if text.strip().startswith('[') and text.strip().endswith(']'):
                            continue
                        else:
                            text = text.replace('[', '').replace(']', '')

                        if len(text) > 2:
                            outfile_name = '{}_segment_{}_line_{}.wav'.format(OUTPUT_PATH, i, j)
                            print_and_write()
                            print_and_write(speaker, outfile_name, text)

                            google_speak(speaker, text, outfile_name)

                            if segment_audio == 'coolio':
                                print_and_write(outfile_name)
                                segment_audio = AudioSegment.from_mp3(outfile_name)
                                segment_audio += AudioSegment.silent(duration=0.7 * _SECOND)
                            else:
                                if speaker == 'Elli':
                                    segment_audio += AudioSegment.from_mp3(outfile_name)
                                    segment_audio += AudioSegment.silent(duration=0.7 * _SECOND)
                                else:
                                    segment_audio += AudioSegment.from_mp3(outfile_name)
                                    segment_audio += AudioSegment.silent(duration=0.7 * _SECOND)

                    segment_audio.export('{}_segment_{}.mp3'.format(OUTPUT_PATH, i), format="mp3")
        else:
            pass

    with open('output_scripts/{}_intro1.txt'.format(date_str), 'r', encoding='utf-8') as introfile:
        intro = introfile.read()
        google_speak('Grace', intro, '{}_intro1.wav'.format(OUTPUT_PATH))

    with open('output_scripts/{}_intro2.txt'.format(date_str), 'r', encoding='utf-8') as introfile:
        intro = introfile.read()
        google_speak('Grace', intro, '{}_intro2.wav'.format(OUTPUT_PATH))

    with open('output_scripts/{}_outro.txt'.format(date_str), 'r', encoding='utf-8') as outrofile:
        outro = outrofile.read()
        google_speak('Grace', outro, '{}_outro.wav'.format(OUTPUT_PATH))

    with open('output_scripts/{}_overview.txt'.format(date_str), 'r', encoding='utf-8') as outrofile:
        from newscaster.audio.overview import overview_audio_maker
        overview_audio_maker(date_str, OUTPUT_PATH)
