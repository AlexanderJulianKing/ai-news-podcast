from pydub import AudioSegment

from newscaster.logging import print_and_write


def fun_intromaker(formatted_date2):
    INFILE_PATH = 'segment_audio/' + formatted_date2
    intro_audio1 = AudioSegment.from_mp3('{}_intro1.wav'.format(INFILE_PATH))
    intro_audio2 = AudioSegment.from_mp3('{}_intro2.wav'.format(INFILE_PATH))

    intro_audio1_duration = intro_audio1.duration_seconds
    intro_audio2_duration = intro_audio2.duration_seconds

    silent_part_duration = AudioSegment.from_mp3('theme_song/newscaster_theme_silent_part.mp3').duration_seconds

    leadin = AudioSegment.from_mp3('theme_song/newscaster_theme_leadin2.mp3')
    leadin_extender = AudioSegment.from_mp3('theme_song/newscaster_theme_leadin_extender.mp3')

    leadin = leadin_extender + leadin

    intro_audio1 = AudioSegment.silent(duration=(leadin.duration_seconds - intro_audio1_duration) * 1000) + intro_audio1

    silent_part = AudioSegment.silent(duration=(silent_part_duration) * 1000)

    intro_audio_voice = intro_audio1 + silent_part + intro_audio2
    BGM_minus2 = AudioSegment.from_mp3('theme_song/newscaster_theme_1_-2.mp3')
    BGM_minus1 = AudioSegment.from_mp3('theme_song/newscaster_theme_1_-1.mp3')
    BGM_0 = AudioSegment.from_mp3('theme_song/newscaster_theme_1_0.mp3')
    BGM_1 = AudioSegment.from_mp3('theme_song/newscaster_theme_1_1.mp3')
    BGM_2 = AudioSegment.from_mp3('theme_song/newscaster_theme_1_2.mp3')
    BGM_3 = AudioSegment.from_mp3('theme_song/newscaster_theme_1_3.mp3')
    BGM_4 = AudioSegment.from_mp3('theme_song/newscaster_theme_1_4.mp3')

    BGM_list = [BGM_minus2, BGM_minus1, BGM_0, BGM_1, BGM_2, BGM_3, BGM_4]

    best_i = 0
    for i in range(len(BGM_list)):
        print_and_write(' \n')
        print_and_write('BGM_list', i)
        print_and_write(intro_audio_voice.duration_seconds, BGM_list[i].duration_seconds)
        if intro_audio_voice.duration_seconds < BGM_list[i].duration_seconds:
            best_i = i
            break
        else:
            pass

    BGM = BGM_list[best_i]
    intro_audio_voice = intro_audio_voice + AudioSegment.silent(duration=(BGM.duration_seconds - intro_audio_voice.duration_seconds) * 1000)

    intro_audio_overlay = intro_audio_voice.overlay(BGM)

    intro_audio_overlay.export("segment_audio/{}_intro.mp3".format(formatted_date2), format="mp3")
