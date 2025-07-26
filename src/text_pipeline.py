import os
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

# Sửa đường dẫn để chạy từ thư mục src
STOPWORDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vietnamese-stopwords-dash.txt'))

try:
    with open(STOPWORDS_PATH, encoding='utf-8') as f:
        STOPWORDS = set([line.strip() for line in f if line.strip()])
    print(f"✅ Loaded {len(STOPWORDS)} stopwords from {STOPWORDS_PATH}")
except Exception as e:
    print(f"❌ Error loading stopwords: {e}")
    STOPWORDS = set()

try:
    EMBED_MODEL = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    print("✅ Text embedding model loaded successfully")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    EMBED_MODEL = None

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
    if EMBED_MODEL is None:
        raise RuntimeError("Embedding model not loaded")
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