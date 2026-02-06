from pydub import AudioSegment

from newscaster.config import _SECOND


def assemble_podcast(formatted_date2):
    OUTPUT_PATH = 'segment_audio/' + formatted_date2

    AudioSegment.from_mp3('{}_intro.mp3'.format(OUTPUT_PATH))
    podcast = AudioSegment.from_mp3('{}_intro.mp3'.format(OUTPUT_PATH))
    podcast += AudioSegment.silent(duration=2 * _SECOND)

    for i in range(2):
        try:
            podcast += AudioSegment.from_mp3('{}_segment_{}.mp3'.format(OUTPUT_PATH, i))
            podcast += AudioSegment.silent(duration=2 * _SECOND)
        except:
            pass
        if i == 1:
            try:
                podcast += AudioSegment.from_mp3('{}_overview.wav'.format(OUTPUT_PATH, i))
                podcast += AudioSegment.silent(duration=2 * _SECOND)
            except:
                pass
    NEW_OUTPUT_PATH = 'output_audio/' + formatted_date2
    podcast += AudioSegment.from_mp3('{}_outro.wav'.format(OUTPUT_PATH))

    if formatted_date2 == '2023_10_02':
        podcast = AudioSegment.from_mp3('goodbye.mp3') + podcast
    output_file = "{}.mp3".format(NEW_OUTPUT_PATH)
    output_file_HQ = "{}_HQ.mp3".format(NEW_OUTPUT_PATH)

    podcast.export(output_file, format="mp3", bitrate="124k")
    podcast.export(output_file_HQ, format="mp3", bitrate='248k')
