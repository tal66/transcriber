import logging
from pathlib import Path

import torch
import whisper
from pyannote.audio import Pipeline

from secrets_ import huggingface_token

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model("small").to(device)

filename = 'Warren Buffett shares advice on becoming successful.wav'
filename = 'The Emptiness Machine (Official Music Video) - Linkin Park.wav'
filename = 'Sergey Brin ｜ All-In Summit 2024.wav'

# transcription
result = model.transcribe(filename)

# diarization
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=huggingface_token)
pipeline.to(torch.device("cuda"))
diarization = pipeline(filename)

transcription = []

itertracks = diarization.itertracks(yield_label=True)
turn, _, speaker = next(itertracks)
current_speaker = None

for segment in result['segments']:
    seg_start_time = segment['start']
    seg_end_time = segment['end']
    text = segment['text']

    while turn and (turn.end < seg_start_time):
        turn, _, speaker = next(itertracks, (None, None, None))

    if speaker != current_speaker:
        transcription.append("\n")
        current_speaker = speaker
        transcription.append(f"{speaker}: ")

    transcription.append(text)
    logger.info(f"{speaker}: {text}")

final_transcription = ''.join(transcription[1:])  # first line is empty
# print(final_transcription)

out_filename = "sd_" + Path(filename).stem + ".txt"
Path(out_filename).write_text(final_transcription)
logger.info(f"file saved as '{out_filename}'")
