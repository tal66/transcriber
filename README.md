
## Transcribe

Transcribe audio to text (using whisper). Runs locally.

### Real Time

File: `transcribe.py`. Output example: [20s video](img/rt_sample.mp4)

<img src="./img/rt_img.png" width="500" height="350" alt="ui">


### YouTube / File

Files: `transcribe.py`, `youtube_util.py`. Possibly select start and end time. 

Optional UI (served by flask) to get the transcript from a file or YouTube link, using `python -m src.app`. Takes few seconds for a few minutes of audio, for example [this song](https://www.youtube.com/watch?v=tI-5uv4wryI) took 5 seconds:

<img src="./img/ui.png" width="600" height="435" alt="ui">


### Speaker diarization

File: `speaker_diarization.py`, separating the speakers is done using `pyannote` (hugging face token required). Output example:

```
SPEAKER_01:  No introduction needed.
SPEAKER_03:  Welcome. I just agreed to this last minute, as you know.
...
```

## Settings and dependencies

Install and read the instructions for [torch](https://pytorch.org/), [pyannote](https://github.com/pyannote/pyannote-audio) (for separating speakers - hugging face token required), and install the other requirements.

Set device ID's in `settings.py` (helper functions are in `audio_util.py`).

The model may hallucinate a bit, and be non-deterministic. 

