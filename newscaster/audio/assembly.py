from pydub import AudioSegment

from newscaster.config import _SECOND
from newscaster.logging import print_and_write


def assemble_podcast(formatted_date2):
    OUTPUT_PATH = 'segment_audio/' + formatted_date2

    podcast = AudioSegment.from_mp3('{}_intro.mp3'.format(OUTPUT_PATH))
    podcast += AudioSegment.silent(duration=2 * _SECOND)

    for i in range(2):
        try:
            podcast += AudioSegment.from_mp3('{}_segment_{}.mp3'.format(OUTPUT_PATH, i))
            podcast += AudioSegment.silent(duration=2 * _SECOND)
        except Exception as e:
            print_and_write(f'Missing audio segment: {e}')
        if i == 1:
            try:
                podcast += AudioSegment.from_mp3('{}_overview.wav'.format(OUTPUT_PATH))
                podcast += AudioSegment.silent(duration=2 * _SECOND)
            except Exception as e:
                print_and_write(f'Missing audio segment: {e}')
    NEW_OUTPUT_PATH = 'output_audio/' + formatted_date2
    podcast += AudioSegment.from_mp3('{}_outro.wav'.format(OUTPUT_PATH))

    output_file = "{}.mp3".format(NEW_OUTPUT_PATH)
    output_file_HQ = "{}_HQ.mp3".format(NEW_OUTPUT_PATH)

    podcast.export(output_file, format="mp3", bitrate="124k")
    podcast.export(output_file_HQ, format="mp3", bitrate='248k')
