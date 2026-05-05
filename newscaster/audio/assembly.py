import os

from pydub import AudioSegment

from newscaster.config import _SECOND
from newscaster.logging import print_and_write


def assemble_podcast(formatted_date2):
    """Stitch the final podcast from whichever segment audios exist on disk.

    Order: intro → all available segments in slot order → overview → outro.
    Slots whose scripts failed during gather/segments stages are simply absent
    from segment_audio/ and silently skipped — no podcast-wide failure for
    one missing story.
    """
    OUTPUT_PATH = 'segment_audio/' + formatted_date2

    podcast = AudioSegment.from_mp3('{}_intro.mp3'.format(OUTPUT_PATH))
    podcast += AudioSegment.silent(duration=2 * _SECOND)

    available_slots = []
    for i in range(4):
        path = '{}_segment_{}.mp3'.format(OUTPUT_PATH, i)
        if os.path.exists(path):
            available_slots.append(i)
        else:
            print_and_write(f'AUDIO ASSEMBLY: no audio for slot {i}; skipping')

    if not available_slots:
        print_and_write('AUDIO ASSEMBLY CRITICAL: no segment audio available for any slot; aborting assembly')
        return

    for slot in available_slots:
        try:
            podcast += AudioSegment.from_mp3('{}_segment_{}.mp3'.format(OUTPUT_PATH, slot))
            podcast += AudioSegment.silent(duration=2 * _SECOND)
        except Exception as e:
            print_and_write(f'AUDIO ASSEMBLY: failed to load slot {slot}: {e}; continuing without it')

    try:
        podcast += AudioSegment.from_mp3('{}_overview.wav'.format(OUTPUT_PATH))
        podcast += AudioSegment.silent(duration=2 * _SECOND)
    except Exception as e:
        print_and_write(f'AUDIO ASSEMBLY: missing overview audio: {e}')

    NEW_OUTPUT_PATH = 'output_audio/' + formatted_date2
    podcast += AudioSegment.from_mp3('{}_outro.wav'.format(OUTPUT_PATH))

    output_file = "{}.mp3".format(NEW_OUTPUT_PATH)
    output_file_HQ = "{}_HQ.mp3".format(NEW_OUTPUT_PATH)

    podcast.export(output_file, format="mp3", bitrate="124k")
    podcast.export(output_file_HQ, format="mp3", bitrate='248k')
