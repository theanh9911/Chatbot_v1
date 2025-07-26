import os
from text_pipeline import preprocess, get_embedding as get_text_emb
from image_pipeline import get_image_embedding
from faiss_pipeline import FaissMultiModalSearch

print("🚀 Building indexes for AI Challenge HCM...")

# Build index cho text
print("📝 Building text index...")
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

# Xử lý text sau khi đã load tất cả files
if texts:
    print(f"📊 Processing {len(texts)} text entries...")
    embs = [get_text_emb(preprocess(t)) for t in texts]
    # Sử dụng FlatL2 cho dữ liệu nhỏ, IVF+PQ cho dữ liệu lớn
    use_ivfpq = len(embs) >= 256  # Chỉ dùng IVF+PQ khi có >= 256 samples
    nlist = min(16, len(embs) // 2) if len(embs) > 1 else 1
    
    text_searcher = FaissMultiModalSearch(dim=768, index_path="data/faiss_text.bin", meta_path="data/faiss_text.pkl", nlist=nlist, use_ivfpq=use_ivfpq)
    if use_ivfpq:
        text_searcher.train(embs)
    text_searcher.add_batch(embs, metas)
    text_searcher.save()
    print(f"✅ Text index built successfully. Samples: {len(embs)}, nlist: {nlist}, use_ivfpq: {use_ivfpq}")
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
                        
                        # Tạo embedding
                        emb = get_image_embedding(frame_path)
                        vid_embs.append(emb)
                        
                        # Metadata với thông tin frame
                        time_sec = frame_idx / fps if fps > 0 else 0
                        vid_metas.append({
                            "file": fname, 
                            "frame": frame_count,
                            "time": f"{time_sec:.1f}s",
                            "description": f"Frame {frame_count} tại {time_sec:.1f}s của video {fname}"
                        })
                        
                        # Cleanup
                        os.remove(frame_path)
                        frame_count += 1
                        
                        # Giới hạn tối đa 10 frames per video
                        if frame_count >= 10:
                            break
                
                vidcap.release()
                print(f"✅ Extracted {frame_count} frames from {fname}")
        
        if vid_embs:
            # Sử dụng FlatL2 cho dữ liệu nhỏ thay vì IVF+PQ
            use_ivfpq = len(vid_embs) >= 256  # Chỉ dùng IVF+PQ khi có >= 256 samples
            nlist = min(4, len(vid_embs) // 2) if len(vid_embs) > 1 else 1
            
            image_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image.bin", meta_path="data/faiss_image.pkl", nlist=nlist, use_ivfpq=use_ivfpq)
            if use_ivfpq:
                image_searcher.train(vid_embs)
            image_searcher.add_batch(vid_embs, vid_metas)
            image_searcher.save()
            print(f"✅ Image index built successfully. Samples: {len(vid_embs)}, nlist: {nlist}, use_ivfpq: {use_ivfpq}")
        else:
            print("⚠️ No video frames extracted")
    else:
        print("⚠️ Video directory data/vid/ not found")
        
except ImportError:
    print("❌ Missing opencv-python. Install with: pip install opencv-python")
except Exception as e:
    print(f"❌ Error processing videos: {e}")

# Build index cho ảnh (nếu có data/images/)
print("🖼️ Building image index from static images...")
img_dir = "data/images"
if os.path.exists(img_dir):
    img_embs = []
    img_metas = []
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("⚠️ No image files found in data/images/")
    else:
        for fname in image_files:
            fpath = os.path.join(img_dir, fname)
            emb = get_image_embedding(fpath)
            img_embs.append(emb)
            img_metas.append({"file": fname, "description": f"Ảnh {fname}"})
            print(f"🖼️ Processed image: {fname}")
        
        if img_embs:
            nlist = min(8, len(img_embs) // 2) if len(img_embs) > 1 else 1
            use_ivfpq = len(img_embs) >= 256  # Chỉ dùng IVF+PQ khi có >= 256 samples
            image_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image_img.bin", meta_path="data/faiss_image_img.pkl", nlist=nlist, use_ivfpq=use_ivfpq)
            if use_ivfpq:
                image_searcher.train(img_embs)
            image_searcher.add_batch(img_embs, img_metas)
            image_searcher.save()
            print(f"✅ Static image index built successfully. Samples: {len(img_embs)}, nlist: {nlist}, use_ivfpq: {use_ivfpq}")
        else:
            print("⚠️ No valid images processed")
else:
    print("⚠️ Image directory data/images/ not found")

print("🎉 Index building completed!")
print("📁 Generated files:")
print("  - data/faiss_text.bin (text index)")
print("  - data/faiss_text.pkl (text metadata)")
print("  - data/faiss_image.bin (image index)")
print("  - data/faiss_image.pkl (image metadata)") 