import os
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

STOPWORDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vietnamese-stopwords-dash.txt'))
with open(STOPWORDS_PATH, encoding='utf-8') as f:
    STOPWORDS = set([line.strip() for line in f if line.strip()])

EMBED_MODEL = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

def word_segment(text):
    return ViTokenizer.tokenize(text)

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w.lower() not in STOPWORDS]
    return ' '.join(filtered)

def preprocess(text):
    seg = word_segment(text)
    clean = remove_stopwords(seg)
    return clean

def get_embedding(text):
    return EMBED_MODEL.encode([text])[0]

if __name__ == "__main__":
    raw_text = "Tôi yêu tiếng Việt và AI Challenge 2025."
    print("Văn bản gốc:", raw_text)
    seg = word_segment(raw_text)
    print("Tách từ:", seg)
    clean = remove_stopwords(seg)
    print("Loại stopwords:", clean)
    emb = get_embedding(clean)
    print("Embedding shape:", emb.shape) 