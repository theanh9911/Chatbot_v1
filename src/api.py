from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from .faiss_pipeline import FaissMultiModalSearch
from .text_pipeline import preprocess, get_embedding as get_text_emb
from .image_pipeline import get_image_embedding
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
import base64

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Challenge HCM API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo searcher cho từng modal
try:
    text_searcher = FaissMultiModalSearch(dim=768, index_path="data/faiss_text.bin", meta_path="data/faiss_text.pkl")
    text_searcher.load()
    logger.info("✅ Text searcher loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading text searcher: {e}")
    text_searcher = None

try:
    image_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image.bin", meta_path="data/faiss_image.pkl")
    image_searcher.load()
    logger.info("✅ Image searcher loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading image searcher: {e}")
    image_searcher = None

class TextQuery(BaseModel):
    query: str
    top_k: int = 5

    class Config:
        schema_extra = {
            "example": {
                "query": "xin chào",
                "top_k": 5
            }
        }

@app.post("/search_text")
def search_text(req: TextQuery):
    if text_searcher is None:
        raise HTTPException(status_code=503, detail="Text searcher not available")
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    
    try:
        logger.info(f"Processing text search: '{req.query}' with top_k={req.top_k}")
        q_clean = preprocess(req.query)
        q_emb = get_text_emb(q_clean)
        results = text_searcher.search(q_emb, top_k=req.top_k)
        # Giả sử mỗi kết quả có trường 'score' (nếu chưa có, cần cập nhật pipeline để trả về score)
        file_score = {}
        for idx, r in enumerate(results):
            file = r.get('file')
            # Ưu tiên lấy score nếu có, nếu không thì dùng vị trí (top đầu score cao hơn)
            score = r.get('score', None)
            if score is None:
                # Nếu không có score, gán score giảm dần theo thứ tự
                score = req.top_k - idx
            if file:
                if file not in file_score or score > file_score[file]:
                    file_score[file] = score
        # Sắp xếp file theo score giảm dần
        matched_files = [ {"file": f, "score": file_score[f]} for f in sorted(file_score, key=lambda x: file_score[x], reverse=True) ]
        logger.info(f"Text search completed, found {len(matched_files)} unique files (sorted by score)")
        return {"matched_files": matched_files}
    except Exception as e:
        logger.error(f"Text search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search_image")
def search_image(file: UploadFile = File(...), top_k: int = 5):
    if image_searcher is None:
        raise HTTPException(status_code=503, detail="Image searcher not available")
    
    # Validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    
    # Kiểm tra kích thước file (max 10MB)
    file_size = 0
    content = file.file.read()
    file_size = len(content)
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
    
    try:
        logger.info(f"Processing image search: {file.filename} ({file_size} bytes) with top_k={top_k}")
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        emb = get_image_embedding(temp_path)
        results = image_searcher.search(emb, top_k=top_k)
        
        # Bổ sung trường image_base64 cho mỗi kết quả
        new_results = []
        for r in results:
            r = dict(r)  # copy để không ảnh hưởng meta gốc
            file_name = r.get('file')
            image_base64 = None
            # Xác định đường dẫn file ảnh hoặc frame đầu video
            img_path = None
            if file_name:
                # Ưu tiên tìm trong thư mục video
                vid_path = os.path.join("../data/vid", file_name)
                img_path = None
                if os.path.exists(vid_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # Nếu là video, lấy frame đầu tiên
                    import cv2
                    cap = cv2.VideoCapture(vid_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
                            cv2.imwrite(tmp_img.name, frame)
                            tmp_img.flush()
                            with open(tmp_img.name, 'rb') as imgf:
                                image_base64 = base64.b64encode(imgf.read()).decode('utf-8')
                        os.remove(tmp_img.name)
                elif os.path.exists(vid_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = vid_path
                else:
                    # Thử tìm trong thư mục images
                    img_path2 = os.path.join("../data/images", file_name)
                    if os.path.exists(img_path2) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = img_path2
                if img_path:
                    with open(img_path, 'rb') as imgf:
                        image_base64 = base64.b64encode(imgf.read()).decode('utf-8')
            r['image_base64'] = image_base64
            new_results.append(r)
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        logger.info(f"Image search completed, found {len(new_results)} results")
        return {"matched_files": new_results}
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Image search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "AI Challenge HCM API - Text and Image Search",
        "version": "1.0.0",
        "endpoints": {
            "text_search": "/search_text",
            "image_search": "/search_image",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "text_searcher": text_searcher is not None,
        "image_searcher": image_searcher is not None,
        "text_index_size": text_searcher.index.ntotal if text_searcher else 0,
        "image_index_size": image_searcher.index.ntotal if image_searcher else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True) 