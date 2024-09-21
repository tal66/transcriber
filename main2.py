import logging
import queue
import threading
import wave

import numpy as np
import sounddevice as sd
import whisper
from scipy.signal import resample

LANG = 'en'

# Audio params
SAMPLE_RATE = 44100
REC_CHANNELS = 2
# DURATION_SEC = 10

# Whisper params
WHISPER_SAMPLE_RATE = 16000  # 16kHz
WHISPER_CHANNELS = 1
CHUNK_DURATION = 3  # process x seconds at a time
OVERLAP_DURATION = 1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

model = whisper.load_model("medium").to('cuda')
audio_queue = queue.Queue()
transcription_queue = queue.Queue()

def list_audio_devices():
    devices = sd.query_devices()
    # for i, device in enumerate(devices):
    #     print(f"Device ID: {i}, Name: {device['name']}, Max Output Channels: {device['max_output_channels']}")

    return devices


def find_loopback_device(devices):
    """audio currently being played by your computer"""
    for i, device in enumerate(devices):
        if ('loopback' in device['name'].lower()) or ('stereo mix' in device['name'].lower()):
            return i
    return None


def record(duration):
    print(f"recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=REC_CHANNELS, dtype='int16',
                       device=device_id)

    sd.wait()
    logger.info("finished recording")

    # normalize
    recording = recording / np.max(np.abs(recording))
    # convert to int16 for wav file
    audio_data = (recording * 32767).astype(np.int16)

    logger.info(f"recorded {len(recording)} samples")
    return audio_data


def save_audio_file(audio_data, filename="system_audio.wav"):
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(REC_CHANNELS)
        wav_file.setsampwidth(2)  # 2 byte int16
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_data.tobytes())
    print(f"audio saved as {filename}")


###########


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def transcribe_audio():
    """transcribe audio chunks from the queue"""
    print(f"Transcription: ")
    while True:
        audio_data = []
        while len(audio_data) < SAMPLE_RATE * CHUNK_DURATION:
            if not audio_queue.empty():
                audio_data.extend(audio_queue.get())

        if len(audio_data) > 0:
            audio_np = np.array(audio_data)

            # convert to mono, downsample, normalize for Whisper
            audio_mono = np.mean(audio_np, axis=1)
            audio_resampled = resample(audio_mono, int(len(audio_mono) * WHISPER_SAMPLE_RATE / SAMPLE_RATE))
            audio_resampled = audio_resampled / np.max(np.abs(audio_resampled))

            # transcribe
            result = model.transcribe(audio_resampled, language=LANG)
            print(f"{result['text']}")


def record_and_transcribe(duration, device_id):
    try:
        # transcription thread
        transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)
        transcribe_thread.start()

        # record
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=WHISPER_CHANNELS, device=device_id,
                            callback=audio_callback):
            logger.info(f"recording and transcribing for {duration} seconds...")
            sd.sleep(int(duration * 1000))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    wav_file = "system_audio.wav"
    devices = list_audio_devices()

    device_id = find_loopback_device(devices)
    logger.info(f"using device [{device_id}]: {devices[device_id]['name']}")

    # audio_data = record(duration=DURATION_SEC)
    # save_audio_file(audio_data, wav_file)

    # Record and transcribe for 60 seconds
    record_and_transcribe(60 * 3, device_id)
