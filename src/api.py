from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from faiss_pipeline import FaissMultiModalSearch
from text_pipeline import preprocess, get_embedding as get_text_emb
from image_pipeline import get_image_embedding
from audio_pipeline import get_audio_embedding
import os

app = FastAPI()

# Khởi tạo searcher cho từng modal
text_searcher = FaissMultiModalSearch(dim=768, index_path="data/faiss_text.bin", meta_path="data/faiss_text.pkl")
text_searcher.load()
image_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image.bin", meta_path="data/faiss_image.pkl")
image_searcher.load()
audio_searcher = FaissMultiModalSearch(dim=768, index_path="data/faiss_audio.bin", meta_path="data/faiss_audio.pkl")
audio_searcher.load()

class TextQuery(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search_text")
def search_text(req: TextQuery):
    q_clean = preprocess(req.query)
    q_emb = get_text_emb(q_clean)
    results = text_searcher.search(q_emb, top_k=req.top_k)
    return {"results": results}

@app.post("/search_image")
def search_image(file: UploadFile = File(...), top_k: int = 5):
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    emb = get_image_embedding(temp_path)
    results = image_searcher.search(emb, top_k=top_k)
    os.remove(temp_path)
    return {"results": results}

@app.post("/search_audio")
def search_audio(file: UploadFile = File(...), top_k: int = 5):
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    emb = get_audio_embedding(temp_path)
    results = audio_searcher.search(emb, top_k=top_k)
    os.remove(temp_path)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 