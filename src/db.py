import logging

from pymongo import MongoClient
from src.settings import CONN_STRING

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = MongoClient(CONN_STRING)
db = client['transcribe_db']
transcripts_collection = db['transcripts']
transcripts_collection.create_index([('title', 'text'), ('content', 'text')])
logger.info("connected to MongoDB")


def save_transcript(doc: dict):
    result = transcripts_collection.insert_one(doc)
    logger.info(f"inserted transcript: {result.inserted_id}")
    return str(result.inserted_id)


def get_transcript(transcript_id: str):
    res = transcripts_collection.find_one({'_id': transcript_id})
    return res


def search_transcripts(query):
    results = transcripts_collection.find(
        {'$text': {'$search': query}}, {'score': {'$meta': 'textScore'}}
    ).sort([('score', {'$meta': 'textScore'})]).limit(10)

    docs = [{'id': str(doc['_id']), 'title': doc['title'], 'content': doc['content'],
             'src_type': doc.get('src_type', 'unknown')} for doc in results]
    return docs
