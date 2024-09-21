import threading

import sounddevice as sd
import soundfile as sf


class AudioPlayer:
    device_id = None
    devices = sd.query_devices()

    @classmethod
    def find_output_device(cls):
        default_devices = sd.default.device
        for i, device in enumerate(cls.devices):
            if device['max_output_channels'] > 0 and i in default_devices:
                return i
        return None

    @classmethod
    def play_file(cls, filename):
        event = threading.Event()

        try:
            data, fs = sf.read(filename)  # , always_2d=True

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

            num_channels = data.shape[1]
            stream = sd.OutputStream(samplerate=fs, device=cls.device_id, channels=num_channels,
                                     callback=callback, finished_callback=event.set)
            with stream:
                event.wait()
        except KeyboardInterrupt:
            exit('\nInterrupted by user')

    @staticmethod
    def play_file_simple(wav_file):
        data, fs = sf.read(wav_file)
        sd.play(data, fs)
        sd.wait()


AudioPlayer.device_id = AudioPlayer.find_output_device()
