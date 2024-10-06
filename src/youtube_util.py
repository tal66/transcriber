import datetime
import logging
import os
import sys
from pathlib import Path

import yt_dlp

from src.audio_util import AudioEditor
from src.settings import TEMP_FILES_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_audio(youtube_url: str, start_time=None, end_time=None, post_ext='wav', yt_preferredquality=None,
                   yt_format=None) -> str:
    """
    download audio from youtube link.

    downloads the whole audio (trimming is done only after).
    start_time, end_time: e.g. '00:00:30', '00:01:30'.

    returns: filename
    """
    title = yt_dlp.YoutubeDL().extract_info(youtube_url, download=False)['title']

    # file already exists
    prev_res_file_out = f"{TEMP_FILES_DIR}/{title}.{post_ext}"
    if Path(prev_res_file_out).is_file():
        logger.info(f"file '{prev_res_file_out}' already exists")
        return prev_res_file_out

    post_proc = [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': post_ext,
    }]

    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'postprocessors': post_proc,
        'outtmpl': f"{TEMP_FILES_DIR}/%(title)s.%(ext)s",
    }

    if yt_format:
        ydl_opts['format'] = yt_format
    if yt_preferredquality:  # e.g. '192'
        post_proc[0]['preferredquality'] = yt_preferredquality

    logger.info(f"downloading audio from {youtube_url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # ydl.download([youtube_url])
        result = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(result)

    # print(result)
    filename = f"{TEMP_FILES_DIR}/{Path(filename).stem}.{post_ext}"

    set_date_to_now(filename)

    logger.info(f"{get_info_from_result(result)[0]}")
    logger.info(f"file: '{Path(filename).resolve()}', size: {Path(filename).stat().st_size / 1024 ** 2:.2f} MB")  # actually MiB

    # trim
    if start_time or end_time:
        logger.info(f"trimming '{filename}' {start_time}-{end_time}...")
        trimmed_filename = AudioEditor.audio_segment(filename, start_time, end_time, output_ext=post_ext)
        return trimmed_filename

    return filename


def get_info_from_result(result: dict) -> tuple[str, dict]:
    meta = {}
    meta['title'] = result.get('title')
    meta['channel'] = result.get('uploader')
    meta['duration'] = format_duration(result['duration'])
    meta['pub_date'] = result.get('upload_date')

    # chapters
    if result.get('chapters'):
        chapters = []

        for chapter in result['chapters']:
            c_title = chapter['title']
            c_start_time = chapter['start_time']
            c_start_time = format_duration(c_start_time)
            chapters.append(f"{c_start_time} - {c_title}")

        meta['chapters'] = chapters

    # str
    result_str = f"{meta['title']}"
    result_str += f", {meta['duration']}"
    result_str += f"\nChannel: {meta['channel']}"
    if 'chapters' in meta:
        result_str += f"\nChapters:\n{'\n'.join(meta['chapters'])}"

    return result_str, meta


def get_info(youtube_url: str):
    """ get info from youtube link """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(youtube_url, download=False)

    result_str, meta = get_info_from_result(result)
    return result_str, meta


def get_captions(youtube_url: str):
    pass


def set_date_to_now(filename):
    # set file modification time
    now = datetime.datetime.now().timestamp()
    os.utime(filename, (now, now))


def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


if __name__ == '__main__':
    if len(sys.argv) == 2:
        youtube_url = sys.argv[1]
        download_audio(youtube_url)
