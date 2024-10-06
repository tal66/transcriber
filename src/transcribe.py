import logging
import os
import queue
import re
import threading
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import torch
import whisper

from src.audio_util import DeviceUtil, AudioEditor, to_str_hhmmss
from src.settings import TEMP_FILES_DIR, LOOPBACK_DEVICE_ID

# config
SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE = 16000  # 16kHz
WHISPER_CHANNELS = 1
CHUNK_DURATION_SEC = 5  # x seconds at a time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')

LANG = 'en'
model_name = "small.en"  # sometimes small is better than small.en
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model(model_name).to(device)
logger.info(f"model: {model_name}, device: {device}")

audio_queue = queue.Queue()
transcription_queue = queue.Queue()
devices = DeviceUtil.list_audio_devices()


def transcribe_audio():
    """transcribe audio chunks from queue"""
    print(f"Transcription: ")

    while True:
        audio_data = []
        while len(audio_data) < SAMPLE_RATE * CHUNK_DURATION_SEC:
            if not audio_queue.empty():
                audio_data.extend(audio_queue.get())

        if len(audio_data) > 0:
            audio_np = np.array(audio_data)

            # convert to mono channel, resample
            audio_mono = np.mean(audio_np, axis=1)
            audio_resampled = librosa.resample(audio_mono, orig_sr=SAMPLE_RATE, target_sr=WHISPER_SAMPLE_RATE)

            # transcribe
            result = model.transcribe(audio_resampled, language=LANG)

            if isinstance(result, tuple):
                segments, info = result
                for segment in segments:
                    print(f"{segment.text}")
            else:
                print(result['text'])


def record_and_transcribe_real_time(duration, device_id):
    """real time (less accurate)"""

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    logger.info(f"using device [{device_id}]: {devices[device_id]['name']}, "
                f"{devices[device_id]['max_input_channels']} channels, "
                f"{devices[device_id]['default_samplerate']} Hz")

    try:
        # transcription thread
        transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)
        transcribe_thread.start()
        logger.debug("transcription thread started")

        # record
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=WHISPER_CHANNELS,
                            device=device_id, callback=audio_callback):
            logger.info(f"recording and transcribing for {duration} seconds...")
            sd.sleep(int(duration * 1000))
    except Exception as e:
        logger.info(f"audio devices list:\n{DeviceUtil.list_audio_devices()}")
        logger.error(e)


###########

def transcribe_file(audio_file, show_timestamps=False):
    logger.info(f"transcribing '{audio_file}'...")
    result = model.transcribe(audio_file, language=LANG, verbose=True if logger.level == logging.DEBUG else False)

    result_str = result["text"]
    if show_timestamps:

        res_list = []
        for segment in result["segments"]:
            line = f"{to_str_hhmmss(segment['start'])} - {to_str_hhmmss(segment['end'])}: {segment['text']}"
            res_list.append(line)
        result_str = "\n".join(res_list)

    out_file = f"{TEMP_FILES_DIR}/t_{Path(audio_file).stem}.txt"
    out_file = re.sub(r'[<>!^&*@#$+`]', '', out_file)
    out_file = re.sub(r'[:：]', '_', out_file)

    Path(out_file).write_text(result_str)
    logger.info(f"saved as: '{out_file}'")
    return out_file


def transcribe_file_segment(audio_file, start_time_str=None, end_time_str=None, show_timestamps=False):
    if (not start_time_str) and (not end_time_str):
        out_file = transcribe_file(audio_file, show_timestamps)
        return out_file

    logger.debug(f"creating temp segment file...")
    segment_file = AudioEditor.audio_segment(audio_file, start_time_str, end_time_str)
    out_file = transcribe_file(segment_file, show_timestamps)

    logger.debug(f"rm segment file '{segment_file}'")
    os.remove(segment_file)
    return out_file


if __name__ == "__main__":
    device_id = LOOPBACK_DEVICE_ID

    # Record and transcribe
    record_and_transcribe_real_time(60 * 2, device_id)

    # transcribe_file_segment(file, "8:00", "15:00")
