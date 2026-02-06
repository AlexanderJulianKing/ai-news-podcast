import os

from pydub import AudioSegment

from newscaster.audio.tts import google_speak


def split_text_overview(text, limit=5000):
    """Splits the text into chunks that are under the Google Text-to-Speech API limit."""
    if len(text.encode('utf-8')) <= limit:
        return [text]
    parts = []
    current_part = ""
    for word in text.split():
        if len((current_part + word).encode('utf-8')) < limit:
            current_part += word + " "
        else:
            parts.append(current_part)
            current_part = word + " "
    parts.append(current_part)
    return parts


def merge_audio_files_overview(file_list, output_file):
    """Merges multiple audio files into a single file."""
    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_wav(file)
        combined += audio
    combined.export(output_file, format="wav")


def overview_audio_maker(date_str, OUTPUT_PATH):
    with open(f'output_scripts/{date_str}_overview.txt', 'r', encoding='utf-8') as outrofile:
        overview = outrofile.read()

    parts = split_text_overview(overview)

    audio_files = []
    for i, part in enumerate(parts):
        part_filename = f"{OUTPUT_PATH}_part_{i}.wav"
        google_speak('Grace2', part, part_filename)
        audio_files.append(part_filename)

    final_output = f"{OUTPUT_PATH}_overview.wav"
    merge_audio_files_overview(audio_files, final_output)

    for file in audio_files:
        os.remove(file)
