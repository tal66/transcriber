import datetime
import logging
import queue
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

# Audio params
SAMPLE_RATE = 44100
REC_CHANNELS = 1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')

def get_now_str():
    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d_%H%M%S')
    return now_str


class DeviceUtil:
    devices = sd.query_devices()

    @classmethod
    def find_loopback_device(cls):
        """audio played by computer"""
        for i, device in enumerate(cls.devices):
            if ('loopback' in device['name'].lower()) or ('stereo mix' in device['name'].lower()):
                return i

        return None

    @staticmethod
    def find_output_device():
        default_devices = sd.default.device
        for i, device in enumerate(DeviceUtil.devices):
            if device['max_output_channels'] > 0 and i in default_devices:
                return i

        return None

    @staticmethod
    def find_microphone_device(devices):
        for i, device in enumerate(devices):
            if 'microphone' in device['name'].lower() and device['max_input_channels'] > 0:
                return i

        return None

    @staticmethod
    def log_device_info(device_id):
        device = DeviceUtil.devices[device_id]
        logger.info(f"Device ID: {device_id}, Name: {device['name']}, "
                    f"Max Output Channels: {device['max_output_channels']}, Max Input Channels: {device['max_input_channels']}, "
                    f"Sample Rate: {device['default_samplerate']}")

    @staticmethod
    def list_audio_devices():
        devices = sd.query_devices()

        return devices


class AudioRecorder:
    device_id = None  # initialized outside

    @classmethod
    def record(cls, duration_sec, device_id=None) -> np.ndarray:

        if device_id is None:
            device_id = cls.device_id

        DeviceUtil.log_device_info(device_id)

        logger.info(f"recording audio for {duration_sec} seconds...")
        recording = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=REC_CHANNELS, dtype='int16', device=device_id)

        sd.wait()
        logger.info("finished recording")

        # normalize
        recording = recording / np.max(np.abs(recording))
        # convert to int16 for wav file
        audio_data = (recording * 32767).astype(np.int16)

        logger.info(f"recorded {len(recording)} samples")
        return audio_data

    @staticmethod
    def save_audio_data_to_file(audio_data, filename="audio.wav"):
        with wave.open(filename, "w") as wav_file:
            wav_file.setnchannels(REC_CHANNELS)
            wav_file.setsampwidth(2)  # 2 byte int16
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())

        logger.info(f"audio saved as {filename}")

    @staticmethod
    def record_unlimited(device_id, filename=None):
        """
        record audio to file until KeyboardInterrupt
        https://python-sounddevice.readthedocs.io/en/0.5.0/examples.html#recording-with-arbitrary-duration
        """
        if filename is None:
            filename = f"audio_{get_now_str()}.wav"

        q = queue.Queue()

        try:
            def callback(indata, frames, time, status):
                """called from a separate thread for each audio block."""
                if status:
                    print(status)
                q.put(indata.copy())

            # open file before recording
            with sf.SoundFile(filename, mode='w', samplerate=SAMPLE_RATE, channels=REC_CHANNELS) as file:

                with sd.InputStream(samplerate=SAMPLE_RATE, device=device_id, channels=REC_CHANNELS, callback=callback):
                    logger.info(f"recording to '{filename}'.")
                    print("press Ctrl+C to stop recording")
                    DeviceUtil.log_device_info(device_id)

                    # record
                    while True:
                        file.write(q.get())

        except KeyboardInterrupt:
            print(f"recording finished: '{filename}', {file.frames / SAMPLE_RATE:.1f} seconds")

        return filename


class AudioPlayer:
    device_id = None

    @staticmethod
    def play_file(wav_file):
        logger.info(f"playing {wav_file}...")
        data, fs = sf.read(wav_file)
        sd.play(data, fs)
        sd.wait()

    @classmethod
    def play_file_low_level(cls, filename):
        """
        low level play file
        https://python-sounddevice.readthedocs.io/en/0.5.0/examples.html#play-a-sound-file
        """
        event = threading.Event()

        try:
            logger.info(f"playing {filename}...")
            data, fs = sf.read(filename, always_2d=True)  #
            current_frame = 0

            def callback(outdata, frames, time, status):
                nonlocal current_frame
                if status:
                    print(status)

                chunksize = min(len(data) - current_frame, frames)
                outdata[:chunksize] = data[current_frame:current_frame + chunksize]

                if chunksize < frames:
                    outdata[chunksize:] = 0
                    raise sd.CallbackStop()

                current_frame += chunksize

            logger.debug(f"data.shape: {data.shape}")
            num_channels = data.shape[1]
            stream = sd.OutputStream(samplerate=fs, device=cls.device_id, channels=num_channels,
                                     callback=callback, finished_callback=event.set)
            with stream:
                event.wait()

        except KeyboardInterrupt:
            print('stopping...')

    @staticmethod
    def play_audio_data(audio_data: np.ndarray):
        sd.play(audio_data, SAMPLE_RATE)
        sd.wait()


class AudioEditor:
    @staticmethod
    def audio_segment(audio_file, start_time_str="00:00", end_time_str=None, output_ext="wav") -> str:

        logger.info(f"segmenting '{audio_file}' {start_time_str}-{end_time_str}...")

        def to_ms(time_str: str) -> int:
            """ (HH:MM:SS or MM:SS) to milliseconds"""
            parts = time_str.split(':')

            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return (hours * 3600 + minutes * 60 + seconds) * 1000
            elif len(parts) == 2:
                minutes, seconds = map(int, parts)
                return (minutes * 60 + seconds) * 1000
            else:
                raise ValueError("invalid time format. expecting 'MM:SS' or 'HH:MM:SS'.")

        audio = AudioSegment.from_file(audio_file)

        # times
        start_time = to_ms(start_time_str)
        if start_time > len(audio):
            raise ValueError(f"start_time > audio duration. start: {start_time} ms, audio duration: {len(audio)} ms")

        if not end_time_str:
            end_time = len(audio)  # default
        else:
            end_time = to_ms(end_time_str)

        if not start_time:
            start_time = 0

        if end_time <= start_time:
            raise ValueError(f"end_time <= start_time. start: {start_time}, end: {end_time}")
        # if end_time > len(audio) : goes until end

        # segment
        logger.info(f"audio duration: {len(audio) / 1000} s -> {(end_time - start_time) / 1000} s")
        audio_segment = audio[start_time:end_time]

        # save
        out_filename = f"segment_{start_time}_{end_time}_{Path(audio_file).stem}.{output_ext}"
        audio_segment.export(out_filename, format=output_ext)

        logger.info(f"segment saved as '{out_filename}'")
        return out_filename


if __name__ == '__main__':
    mic_device_id = DeviceUtil.find_microphone_device(DeviceUtil.devices)
    speakers_device_id = DeviceUtil.find_output_device()
    loopback_device_id = DeviceUtil.find_loopback_device()

    AudioPlayer.device_id = speakers_device_id
    AudioRecorder.device_id = mic_device_id

    # record and play
    # rec_filename = AudioRecorder.record_unlimited(loopback_device_id)
    # AudioPlayer.play_file(rec_filename)
    AudioEditor.audio_segment('rec.wav')
