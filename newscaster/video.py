from datetime import date

from moviepy.editor import AudioFileClip, ImageClip


def create_video_from_audio_and_image(audio_path, image_path, output_path):
    audio = AudioFileClip(audio_path)
    image = ImageClip(image_path)
    image = image.set_duration(audio.duration)
    video = image.set_audio(audio)
    video.write_videofile(output_path, fps=24)


def create_and_export_video():
    image_file = "image copy.png"
    output_file = "output_video.mp4"

    today = date.today()
    formatted_date2 = today.strftime("%Y_%m_%d")
    NEW_OUTPUT_PATH = 'output_audio/' + formatted_date2
    audio_file = "{}_HQ.mp3".format(NEW_OUTPUT_PATH)
    print(audio_file)
    print('creating movie')
    create_video_from_audio_and_image(audio_file, image_file, output_file)
