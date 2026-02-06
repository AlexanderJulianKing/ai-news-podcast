import os
import time

from google.cloud import texttospeech
from pydub import AudioSegment

from newscaster.config import _SECOND
from newscaster.logging import print_and_write
from newscaster.text_utils import text_cleaner
from newscaster.llm import gemini


def google_speak(name, text, filename):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "newscaster1-03dc16232821.json"

    text = text_cleaner(text)

    client = texttospeech.TextToSpeechClient()
    if name == 'Elias':
        robo_name = "en-US-Studio-M"
    elif name == 'Chloe':
        robo_name = 'en-US-Chirp3-HD-Leda'
    elif name == 'Grace' or name == 'Grace2':
        robo_name = 'en-US-Chirp3-HD-Aoede'
    elif name == 'Ethan':
        robo_name = 'en-US-Chirp3-HD-Fenrir'

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
                    text = gemini(text, system_prompt='Please modify the text to fit these requirements: \n1: The Entirety of the text is no greater than 5000 characters. \n2: Each individual sentence is no greater than 700 characters long. \n3: Do not refer to Donald Trump as the former president; he is the current president. \n Give no other text.')
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

    return


def text2speech(date_str, voices_list):
    voices_list = list(voices_list) + ['Connie']
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

                        if ':' in thing:
                            thing = thing.split(':')
                            print_and_write(thing)
                            speaker1 = thing[0].capitalize()
                            if speaker1 in voices_list:
                                speaker = speaker1
                                text = ''
                            else:
                                text = speaker1
                            if len(thing) > 2:
                                text = text + thing[1] + thing[2]
                            elif len(thing) == 2:
                                text = text + thing[1]
                            else:
                                text = thing[0]
                        else:
                            text = thing

                        if '[' in text:
                            pass
                        elif len(text) > 2:
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
                        else:
                            pass

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
