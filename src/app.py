import logging
import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from src.settings import UPLOAD_DIR
from src.youtube_util import download_audio
from src.transcribe import transcribe_file, transcribe_file_segment

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    source_type = request.form['source_type']
    show_timestamps = request.form.get('show_timestamps') == 'on'
    start_time = request.form.get('start_time')
    end_time = request.form.get('end_time')
    logger.info(f"{source_type}, show_timestamps: {show_timestamps}, start_time: {start_time}, end_time: {end_time}")

    yt_meta = None
    upload_filepath = None
    try:
        if source_type == 'file':
            if 'audio_file' not in request.files:
                return jsonify({'error': 'No audio_file'}), 400

            file = request.files['audio_file']

            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if file:
                filename = secure_filename(file.filename)
                upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_filepath)

        elif source_type == 'youtube':
            youtube_url = request.form.get('youtube_url')
            if not youtube_url:
                return jsonify({'error': 'No YouTube URL provided'}), 400
            if not youtube_url.startswith('https://www.youtube.com/'):
                return jsonify({'error': 'Not YouTube URL'}), 400

            upload_filepath, yt_meta = download_audio(youtube_url)
            logger.info(f"downloaded audio: '{upload_filepath}'")
        else:
            logger.error(f"Invalid source_type: {source_type}")
            return jsonify({'error': 'Invalid input'}), 400
    except Exception as e:
        logger.error(e, exc_info=True)
        return jsonify({'error': str(e)}), 500

    if start_time or end_time:
        result_file = transcribe_file_segment(upload_filepath, start_time, end_time, show_timestamps=show_timestamps)
    else:
        result_file = transcribe_file(upload_filepath, show_timestamps=show_timestamps)

    # rm uploaded file
    os.remove(upload_filepath)

    return jsonify(
        {'message': 'Transcription completed', 'output_file': result_file, 'transcript': Path(result_file).read_text(),
         'yt_meta': yt_meta})


if __name__ == '__main__':
    app.run(debug=True)
