import json
import logging
import os
from pathlib import Path

import torch
import whisper
from pyannote.audio import Pipeline

from src.audio_util import AudioEditor, to_str_hhmmss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# token
HUGGINGFACE_TOKEN = None
try:
    from secrets_ import HUGGINGFACE_TOKEN
except ImportError:
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

if not HUGGINGFACE_TOKEN:
    raise ValueError("Hugging Face token not found")

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model("small").to(device)
logger.info(f"device: {device}")

# conf
TIMESTAMP_FREQ_SEC = 60 * 5


def transcribe_and_diarize(filename: str, end_time_str=None, num_speakers=None):
    """
    end_time_str format example: '10:30'.
    """
    logger.info(f"file: '{filename}'")

    # file exists
    if not Path(filename).is_file():
        raise FileNotFoundError(f"file '{filename}' not found")

    # trim audio
    if end_time_str:
        filename = AudioEditor.audio_segment(filename, end_time_str=end_time_str)

    # transcription
    result_file = f"./temp/t_result_{Path(filename).stem}.txt"
    if not Path(result_file).exists():
        logger.info(f"transcribing '{filename}'...")
        result = model.transcribe(filename)
        Path(result_file).write_text(json.dumps(result))
    else:
        logger.info(f"loaded transcribe result from file '{result_file}'...")

    result = Path(result_file).read_text()
    result = json.loads(result)

    # diarization
    logger.info(f"speaker-diarization for '{filename}'...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN)
    pipeline.to(torch.device(device))
    diarization = pipeline(filename, num_speakers=num_speakers)

    transcription = []
    timestamp_iter_num = 1

    itertracks = diarization.itertracks(yield_label=True)
    turn, _, speaker = next(itertracks)
    current_speaker = None

    for segment in result['segments']:
        seg_start_time = segment['start']
        # seg_end_time = segment['end']
        text = segment['text']

        while turn and (turn.end < seg_start_time + 0.05):
            turn, _, speaker = next(itertracks, (None, None, None))

        if speaker != current_speaker:
            transcription.append("\n")
            current_speaker = speaker

            if seg_start_time > (timestamp_iter_num * TIMESTAMP_FREQ_SEC):
                timestamp_iter_num += 1
                transcription.append(f"\n{to_str_hhmmss(seg_start_time)}\n\n")

            transcription.append(f"{speaker}: ")

        transcription.append(text)

    final_transcription = ''.join(transcription[1:])  # first line is empty

    prefix = "sd_"
    out_filename = f"{prefix}{Path(filename).stem}.txt"
    Path(out_filename).write_text(final_transcription)

    logger.info(f"file saved as '{out_filename}'")

    # rm segmented file
    if end_time_str:
        os.remove(filename)

    return out_filename


if __name__ == '__main__':
    filename = 'audio.wav'
    transcribe_and_diarize(filename)
