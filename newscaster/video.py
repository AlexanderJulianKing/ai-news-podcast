import subprocess
from datetime import date


def _audio_seconds(audio_path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def create_video_from_audio_and_image(audio_path, image_path, output_path):
    """Still image plus audio, encoded by ffmpeg at 1 frame per second.

    MoviePy re-encoded the same 2000x2000 image 24 times a second, which took about
    49 minutes on the Pi for a 26-minute episode; this takes about 4 (2026-10-09).
    `-t` pins the length to the audio, because `-shortest` overshoots with a looped image.
    """
    seconds = _audio_seconds(audio_path)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-loop", "1", "-framerate", "1", "-i", image_path,
         "-i", audio_path,
         "-t", f"{seconds:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
         "-vf", "scale=1920:1920", "-r", "1",
         "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart",
         output_path],
        check=True,
    )


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
