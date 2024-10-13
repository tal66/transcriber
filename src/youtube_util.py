import datetime
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp

from src.audio_util import AudioEditor
from src.settings import TEMP_FILES_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_audio(youtube_url: str, start_time=None, end_time=None, post_ext='mp3', yt_preferredquality=None,
                   yt_format=None) -> tuple[str, dict]:
    """
    download audio from youtube link.

    downloads the whole audio (trimming is done only after).
    start_time, end_time: e.g. '00:00:30', '00:01:30'.

    returns: filename, metadata
    """
    if not youtube_url.startswith('https://www.youtube.com/shorts'):
        # remove all query params other than 'v'
        # (don't download playlist, or require editing the url. not aware of other query params needed)
        query = parse_qs(urlparse(youtube_url).query)
        if len(query) > 1:
            video_id = query.get('v', [''])[0]
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"removed redundant query params")

    yt_info_dict = yt_dlp.YoutubeDL().extract_info(youtube_url, download=False)
    title = yt_info_dict.get('title', 'unknown_title')
    sanitized = re.sub(r'[^\w\s\-\(\)\[\]]', '_', title)
    res_filename_no_ext = f"{TEMP_FILES_DIR}/{sanitized}"
    res_filename = f"{res_filename_no_ext}.{post_ext}"

    # file already exists
    if Path(res_filename).is_file():
        logger.info(f"file '{res_filename}' already exists")
        return res_filename, get_info_from_result(yt_info_dict)[1]

    post_proc = [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': post_ext,
    }]

    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,  # yt ignores playlist only for info, not for download
        'postprocessors': post_proc,
        'outtmpl': res_filename_no_ext,
    }

    if yt_format:
        ydl_opts['format'] = yt_format
    if yt_preferredquality:  # e.g. '192'
        post_proc[0]['preferredquality'] = yt_preferredquality

    logger.info(f"downloading audio from {youtube_url} ({title})...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(youtube_url, download=True)

    set_date_to_now(res_filename)

    info_str, meta = get_info_from_result(result)
    file_path = Path(res_filename).resolve()
    logger.info(info_str)
    logger.info(f"file: '{file_path}', size: {file_path.stat().st_size / 1024 ** 2:.2f} MB")  # actually MiB

    # trim
    if start_time or end_time:
        logger.info(f"trimming '{res_filename}' {start_time}-{end_time}...")
        trimmed_filename = AudioEditor.audio_segment(res_filename, start_time, end_time, output_ext=post_ext)
        return trimmed_filename, meta

    return res_filename, meta


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
    result_str = f"Title: {meta['title']}"
    result_str += f", duration: {meta['duration']}"
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
