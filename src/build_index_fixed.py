import os
from image_pipeline import clip_processor, clip_model, get_image_embedding
from faiss_pipeline import FaissMultiModalSearch
import torch

print("🚀 Building indexes for AI Challenge HCM...")

# Build index cho text (sử dụng CLIP thay vì SimCSE để đảm bảo tính nhất quán)
print("📝 Building text index with CLIP...")
texts = []
metas = []

# Tự động scan tất cả text files trong data/text/
text_dir = "data/text"
if os.path.exists(text_dir):
    text_files = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
    
    if not text_files:
        print("⚠️ No text files found in data/text/")
    else:
        print(f"📁 Found {len(text_files)} text files: {text_files}")
        
        for fname in text_files:
            fpath = os.path.join(text_dir, fname)
            print(f"📖 Processing file: {fname}")
            
            with open(fpath, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    text = line.strip()
                    if text:
                        texts.append(text)
                        metas.append({"file": fname, "line": i+1, "text": text})
else:
    print("⚠️ Text directory data/text/ not found")

# Xử lý text với CLIP (thay vì SimCSE)
if texts:
    print(f"📊 Processing {len(texts)} text entries with CLIP...")
    embs = []
    for text in texts:
        # Tạo CLIP text embedding với truncation
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            emb = clip_model.get_text_features(**inputs)
        embs.append(emb[0].cpu().numpy())
    
    # Sử dụng FlatL2 cho dữ liệu nhỏ, IVF+PQ cho dữ liệu lớn
    use_ivfpq = len(embs) >= 256  # Chỉ dùng IVF+PQ khi có >= 256 samples
    nlist = min(16, len(embs) // 2) if len(embs) > 1 else 1
    
    # CLIP có dimension 512
    text_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_text.bin", meta_path="data/faiss_text.pkl", nlist=nlist, use_ivfpq=use_ivfpq, use_cosine=True)
    if use_ivfpq:
        text_searcher.train(embs)
    text_searcher.add_batch(embs, metas)
    text_searcher.save()
    print(f"✅ Text index built successfully with CLIP + Cosine. Samples: {len(embs)}, nlist: {nlist}, use_ivfpq: {use_ivfpq}")
else:
    print("⚠️ No text files found in data/text/")

# Build index cho video (nhiều frames)
print("🖼️ Building image index from video frames...")
try:
    import cv2
    vid_dir = "data/vid"
    vid_embs = []
    vid_metas = []
    
    if os.path.exists(vid_dir):
        video_files = [f for f in os.listdir(vid_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print("⚠️ No video files found in data/vid/")
        else:
            for fname in video_files:
                vid_path = os.path.join(vid_dir, fname)
                vidcap = cv2.VideoCapture(vid_path)
                
                if not vidcap.isOpened():
                    print(f"❌ Cannot open video: {fname}")
                    continue
                
                # Lấy thông tin video
                total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                
                print(f"📹 Processing video: {fname} ({total_frames} frames, {duration:.1f}s)")
                
                # Extract nhiều frames (mỗi 2 giây 1 frame)
                frame_interval = max(1, int(fps * 2))  # 1 frame mỗi 2 giây
                frame_count = 0
                
                for frame_idx in range(0, total_frames, frame_interval):
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, image = vidcap.read()
                    
                    if success:
                        # Lưu frame tạm thời
                        frame_path = os.path.join(vid_dir, f"{fname}_frame_{frame_count}.jpg")
                        cv2.imwrite(frame_path, image)
                        
                        # Tạo embedding cho frame
                        emb = get_image_embedding(frame_path)
                        vid_embs.append(emb)
                        
                        # Metadata cho frame
                        frame_time = frame_idx / fps if fps > 0 else 0
                        vid_metas.append({
                            "file": fname,
                            "description": f"Frame {frame_count} tại {frame_time:.1f}s của video {fname}",
                            "frame_number": frame_count,
                            "frame_time": frame_time
                        })
                        
                        frame_count += 1
                        
                        # Xóa file tạm
                        os.remove(frame_path)
                
                vidcap.release()
                print(f"✅ Extracted {frame_count} frames from {fname}")
            
            if vid_embs:
                # Sử dụng FlatIP cho video frames với cosine similarity
                video_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image.bin", meta_path="data/faiss_image.pkl", use_ivfpq=False, use_cosine=True)
                video_searcher.add_batch(vid_embs, vid_metas)
                video_searcher.save()
                print(f"✅ Video frames index built successfully with Cosine. Samples: {len(vid_embs)}")
            else:
                print("⚠️ No video frames extracted")
    else:
        print("⚠️ Video directory data/vid/ not found")
        
except ImportError:
    print("⚠️ OpenCV not available, skipping video processing")
except Exception as e:
    print(f"❌ Error processing videos: {e}")

# Build index cho static images
print("🖼️ Building static image index...")
try:
    img_dir = "data/images"
    img_embs = []
    img_metas = []
    
    if os.path.exists(img_dir):
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print("⚠️ No image files found in data/images/")
        else:
            print(f"📁 Found {len(image_files)} image files: {image_files}")
            
            for fname in image_files:
                img_path = os.path.join(img_dir, fname)
                print(f"🖼️ Processing image: {fname}")
                
                # Tạo embedding cho image
                emb = get_image_embedding(img_path)
                img_embs.append(emb)
                
                # Metadata cho image
                img_metas.append({
                    "file": fname,
                    "description": f"Ảnh {fname}",
                    "type": "static_image"
                })
            
            if img_embs:
                # Sử dụng FlatIP cho static images với cosine similarity
                static_image_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image_img.bin", meta_path="data/faiss_image_img.pkl", use_ivfpq=False, use_cosine=True)
                static_image_searcher.add_batch(img_embs, img_metas)
                static_image_searcher.save()
                print(f"✅ Static images index built successfully with Cosine. Samples: {len(img_embs)}")
            else:
                print("⚠️ No static images processed")
    else:
        print("⚠️ Images directory data/images/ not found")
        
except Exception as e:
    print(f"❌ Error processing static images: {e}")

print("🎉 All indexes built successfully!") 