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
    logger.info("✅ Image searcher (video frames) loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading image searcher (video frames): {e}")
    image_searcher = None

try:
    static_image_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image_img.bin", meta_path="data/faiss_image_img.pkl")
    static_image_searcher.load()
    logger.info("✅ Static image searcher loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading static image searcher: {e}")
    static_image_searcher = None

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
        
        # Tạo kết quả chi tiết với score cho từng result
        detailed_results = []
        for idx, r in enumerate(results):
            # Tính score dựa trên distance từ FAISS (nếu có) hoặc vị trí
            distance = r.get('distance', None)
            if distance is not None:
                # Chuyển đổi distance thành score (distance càng nhỏ, score càng cao)
                import math
                try:
                    # Tránh overflow bằng cách giới hạn distance
                    safe_distance = min(max(distance, -10), 10)  # Giới hạn trong [-10, 10]
                    score = 1.0 / (1.0 + math.exp(safe_distance - 2.0))  # Normalize distance
                    score = round(score * 100, 2)  # Chuyển thành phần trăm
                except (OverflowError, ValueError) as e:
                    # Fallback nếu có lỗi math
                    logger.warning(f"Math error calculating score for distance {distance}: {e}")
                    score = req.top_k - idx
            else:
                # Fallback: score dựa trên vị trí
                score = req.top_k - idx
            
            detailed_result = {
                "file": r.get('file', 'N/A'),
                "line": r.get('line', 'N/A'),
                "text": r.get('text', 'N/A'),
                "description": r.get('text', 'N/A'),  # Sử dụng text làm description
                "score": score,
                "type": "text"
            }
            detailed_results.append(detailed_result)
        
        logger.info(f"Text search completed, found {len(detailed_results)} results with scores")
        return {"matched_files": detailed_results}
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
        all_results = []
        
        # Search trong static images trước (ưu tiên khi search static image)
        static_results = []
        if static_image_searcher:
            try:
                static_results = static_image_searcher.search(emb, top_k=top_k)
                logger.info(f"Found {len(static_results)} static image results")
                
                # Thêm file upload vào kết quả với distance = 0 (perfect match)
                upload_filename = file.filename
                if upload_filename:
                    # Tạo kết quả cho file upload
                    upload_result = {
                        'file': upload_filename,
                        'description': f'Ảnh {upload_filename} (uploaded)',
                        'distance': 0.0,  # Perfect similarity
                        'type': 'uploaded_image',
                        'is_upload': True
                    }
                    static_results.insert(0, upload_result)  # Thêm vào đầu
                    logger.info(f"Added upload file {upload_filename} with distance 0.0")
                    
                    # Loại bỏ file trùng trong database (nếu có)
                    static_results = [r for r in static_results if not (r.get('file') == upload_filename and not r.get('is_upload'))]
                    logger.info(f"Filtered duplicates, total: {len(static_results)} results")
                    
            except Exception as e:
                logger.error(f"Static image search error: {e}")
        
        # Search trong video frames
        video_results = []
        if image_searcher:
            try:
                video_results = image_searcher.search(emb, top_k=top_k)
                logger.info(f"Found {len(video_results)} video frame results")
            except Exception as e:
                logger.error(f"Video frame search error: {e}")
        
        # Kết hợp kết quả với ưu tiên static images và loại bỏ trùng lặp
        all_results = []
        seen_files = set()  # Để track files đã thêm
        
        if static_results:
            # Thêm static images trước với score cao hơn
            for i, result in enumerate(static_results):
                result = dict(result)
                file_name = result.get('file', '')
                
                # Kiểm tra trùng lặp
                if file_name not in seen_files:
                    # Sử dụng distance thực tế từ FAISS
                    distance = result.get('distance', None)
                    if result.get('is_upload'):
                        result['type'] = 'uploaded_image'
                        logger.info(f"Added uploaded image: {file_name} with distance {distance}")
                    else:
                        result['type'] = 'static_image'
                        logger.info(f"Added static image: {file_name} with distance {distance}")
                    
                    all_results.append(result)
                    seen_files.add(file_name)
                else:
                    logger.info(f"Skipped duplicate static image: {file_name}")
        
        if video_results:
            # Thêm video frames sau với score thấp hơn
            for i, result in enumerate(video_results):
                result = dict(result)
                file_name = result.get('file', '')
                frame_info = result.get('description', '')
                
                # Tạo unique key cho video frames
                unique_key = f"{file_name}_{frame_info}"
                
                # Kiểm tra trùng lặp
                if unique_key not in seen_files:
                    # Sử dụng distance thực tế từ FAISS
                    distance = result.get('distance', None)
                    result['type'] = 'video_frame'
                    all_results.append(result)
                    seen_files.add(unique_key)
                    logger.info(f"Added video frame: {file_name} with distance {distance}")
                else:
                    logger.info(f"Skipped duplicate video frame: {unique_key}")
        
        if not all_results:
            logger.warning("No results found from either searcher")
            return {"matched_files": []}
        
        # Sắp xếp theo score (static images sẽ có score cao hơn)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Bổ sung trường image_base64 cho mỗi kết quả
        new_results = []
        for idx, r in enumerate(all_results):
            r = dict(r)  # copy để không ảnh hưởng meta gốc
            file_name = r.get('file')
            image_base64 = None
            
            # Tính score dựa trên distance từ FAISS
            distance = r.get('distance', None)
            if distance is not None:
                # Chuyển đổi distance thành score (distance càng nhỏ, score càng cao)
                # Sử dụng sigmoid để normalize score về 0-100
                import math
                try:
                    # Tránh overflow bằng cách giới hạn distance
                    safe_distance = min(max(distance, -10), 10)  # Giới hạn trong [-10, 10]
                    score = 1.0 / (1.0 + math.exp(safe_distance - 2.0))  # Normalize distance
                    r['score'] = round(score * 100, 2)  # Chuyển thành phần trăm
                    logger.info(f"Converted distance {distance} to score {r['score']}")
                except (OverflowError, ValueError) as e:
                    # Fallback nếu có lỗi math
                    logger.warning(f"Math error calculating score for distance {distance}: {e}")
                    r['score'] = round(max(0, 100 - distance * 10), 2)  # Linear conversion
            else:
                # Fallback: score dựa trên vị trí
                r['score'] = top_k - idx
            
            # Xác định đường dẫn file ảnh hoặc frame đầu video
            img_path = None
            if file_name:
                # Ưu tiên tìm trong thư mục video
                vid_path = os.path.join("data/vid", file_name)
                img_path = None
                if os.path.exists(vid_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # Nếu là video, lấy frame cụ thể dựa trên metadata
                    import cv2
                    cap = cv2.VideoCapture(vid_path)
                    
                    # Lấy thông tin frame từ metadata
                    frame_info = r.get('description', '')
                    frame_number = r.get('frame', 0)
                    
                    # Tìm frame number từ description hoặc sử dụng frame từ metadata
                    if 'Frame' in frame_info:
                        try:
                            # Extract frame number từ description "Frame X tại Ys"
                            frame_match = frame_info.split('Frame ')[1].split(' ')[0]
                            frame_number = int(frame_match)
                            logger.info(f"Extracted frame number: {frame_number} from description: {frame_info}")
                        except Exception as e:
                            logger.warning(f"Failed to extract frame number from description: {frame_info}, error: {e}")
                            frame_number = r.get('frame', 0)
                    
                    logger.info(f"Attempting to extract frame {frame_number} from video: {vid_path}")
                    
                    # Thử nhiều frame khác nhau nếu frame cụ thể không đọc được
                    frame_candidates = [frame_number]
                    if frame_number > 0:
                        frame_candidates.append(frame_number - 1)  # Frame trước
                    frame_candidates.append(0)  # Frame đầu tiên
                    frame_candidates.append(min(10, frame_number + 2))  # Frame sau
                    
                    ret = False
                    frame = None
                    
                    for candidate_frame in frame_candidates:
                        if candidate_frame < 0:
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_frame)
                        ret, frame = cap.read()
                        if ret:
                            logger.info(f"Successfully read frame {candidate_frame} from {file_name}")
                            break
                        else:
                            logger.warning(f"Failed to read frame {candidate_frame}")
                    
                    cap.release()
                    
                    if ret and frame is not None:
                        logger.info(f"Successfully read frame from {file_name}")
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
                            cv2.imwrite(tmp_img.name, frame)
                            tmp_img.flush()
                            with open(tmp_img.name, 'rb') as imgf:
                                image_base64 = base64.b64encode(imgf.read()).decode('utf-8')
                            logger.info(f"Generated base64 image, length: {len(image_base64)}")
                        os.remove(tmp_img.name)
                    else:
                        logger.error(f"Failed to read any frame from {file_name}")
                elif os.path.exists(vid_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = vid_path
                    logger.info(f"Found static image: {img_path}")
                else:
                    # Thử tìm trong thư mục images
                    img_path2 = os.path.join("data/images", file_name)
                    if os.path.exists(img_path2) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = img_path2
                        logger.info(f"Found static image in images folder: {img_path}")
                    else:
                        # Kiểm tra nếu là file upload
                        if r.get('is_upload'):
                            # Sử dụng file upload đã được lưu tạm
                            img_path = temp_path
                            logger.info(f"Using uploaded file for display: {img_path}")
                if img_path:
                    try:
                        with open(img_path, 'rb') as imgf:
                            image_base64 = base64.b64encode(imgf.read()).decode('utf-8')
                            logger.info(f"Generated base64 image from static file, length: {len(image_base64)}")
                    except Exception as e:
                        logger.error(f"Failed to read static image {img_path}: {e}")
                        image_base64 = None
            
            # Thêm thông tin chi tiết
            if 'score' not in r:
                r['score'] = top_k - idx  # Score dựa trên vị trí
            
            # Thêm type nếu chưa có
            if 'type' not in r:
                r['type'] = 'image'
            
            r['image_base64'] = image_base64
            
            # Log thông tin chi tiết về score
            file_type = r.get('type', 'unknown')
            score = r.get('score', 0)
            logger.info(f"Result {idx+1}: {file_name} | Type: {file_type} | Score: {score}")
            new_results.append(r)
            
            logger.info(f"Result {idx+1}: file={file_name}, type=image, has_image={'yes' if image_base64 else 'no'}")
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

@app.get("/debug/videos")
def debug_videos():
    """Debug endpoint để kiểm tra video files"""
    try:
        vid_dir = "data/vid"
        if not os.path.exists(vid_dir):
            return {"error": f"Video directory {vid_dir} not found"}
        
        video_files = [f for f in os.listdir(vid_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        results = []
        for fname in video_files:
            vid_path = os.path.join(vid_dir, fname)
            try:
                import cv2
                cap = cv2.VideoCapture(vid_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    results.append({
                        "file": fname,
                        "total_frames": total_frames,
                        "fps": fps,
                        "duration": duration,
                        "status": "readable"
                    })
                else:
                    results.append({
                        "file": fname,
                        "status": "unreadable"
                    })
            except Exception as e:
                results.append({
                    "file": fname,
                    "status": f"error: {str(e)}"
                })
        
        return {
            "video_directory": vid_dir,
            "total_videos": len(video_files),
            "videos": results
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/static-images")
def debug_static_images():
    """Debug endpoint để kiểm tra static image files"""
    try:
        img_dir = "data/images"
        if not os.path.exists(img_dir):
            return {"error": f"Image directory {img_dir} not found"}
        
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        results = []
        for fname in image_files:
            img_path = os.path.join(img_dir, fname)
            try:
                file_size = os.path.getsize(img_path)
                results.append({
                    "file": fname,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 3),
                    "status": "readable"
                })
            except Exception as e:
                results.append({
                    "file": fname,
                    "status": f"error: {str(e)}"
                })
        
        return {
            "image_directory": img_dir,
            "total_images": len(image_files),
            "images": results
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True) 