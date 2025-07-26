import os
from text_pipeline import preprocess, get_embedding as get_text_emb
from image_pipeline import get_image_embedding
from audio_pipeline import get_audio_embedding
from faiss_pipeline import FaissMultiModalSearch

# Build index cho text
texts = []
metas = []
for fname in ["data/text/t1.txt", "data/text/t2.txt"]:
    if os.path.exists(fname):
        with open(fname, encoding="utf-8") as f:
            for i, line in enumerate(f):
                text = line.strip()
                if text:
                    texts.append(text)
                    metas.append({"file": os.path.basename(fname), "line": i+1, "text": text})
if texts:
    embs = [get_text_emb(preprocess(t)) for t in texts]
    nlist = min(16, len(embs) // 2) if len(embs) > 1 else 1
    use_ivfpq = len(embs) > 1
    text_searcher = FaissMultiModalSearch(dim=768, index_path="data/faiss_text.bin", meta_path="data/faiss_text.pkl", nlist=nlist, use_ivfpq=use_ivfpq)
    if use_ivfpq:
        text_searcher.train(embs)
    text_searcher.add_batch(embs, metas)
    text_searcher.save()
    print(f"Đã build index cho text. Số mẫu: {len(embs)}, nlist: {nlist}, use_ivfpq: {use_ivfpq}")

# Build index cho audio (nhiều file)
try:
    from pydub import AudioSegment
    audio_dir = "data/audio"
    audio_embs = []
    audio_metas = []
    for fname in os.listdir(audio_dir):
        if fname.lower().endswith(('.mp3', '.wav')):
            fpath = os.path.join(audio_dir, fname)
            if fname.lower().endswith('.mp3'):
                tmp_wav = os.path.join(audio_dir, fname + "_tmp.wav")
                AudioSegment.from_mp3(fpath).export(tmp_wav, format="wav")
                emb = get_audio_embedding(tmp_wav)
                os.remove(tmp_wav)
            else:
                emb = get_audio_embedding(fpath)
            audio_embs.append(emb)
            audio_metas.append({"file": fname, "description": f"Nội dung file {fname}"})
    if audio_embs:
        nlist = min(8, len(audio_embs) // 2) if len(audio_embs) > 1 else 1
        use_ivfpq = len(audio_embs) > 1
        audio_searcher = FaissMultiModalSearch(dim=768, index_path="data/faiss_audio.bin", meta_path="data/faiss_audio.pkl", nlist=nlist, use_ivfpq=use_ivfpq)
        if use_ivfpq:
            audio_searcher.train(audio_embs)
        audio_searcher.add_batch(audio_embs, audio_metas)
        audio_searcher.save()
        print(f"Đã build index cho audio. Số mẫu: {len(audio_embs)}, nlist: {nlist}, use_ivfpq: {use_ivfpq}")
except ImportError:
    print("Thiếu thư viện pydub, hãy cài bằng: pip install pydub")

# Build index cho video (nhiều file)
try:
    import cv2
    vid_dir = "data/vid"
    vid_embs = []
    vid_metas = []
    for fname in os.listdir(vid_dir):
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            vid_path = os.path.join(vid_dir, fname)
            frame_path = os.path.join(vid_dir, fname + "_frame.jpg")
            vidcap = cv2.VideoCapture(vid_path)
            success, image = vidcap.read()
            if success:
                cv2.imwrite(frame_path, image)
                emb = get_image_embedding(frame_path)
                vid_embs.append(emb)
                vid_metas.append({"file": fname, "description": f"Frame đầu tiên của video {fname}"})
                os.remove(frame_path)
    if vid_embs:
        nlist = min(8, len(vid_embs) // 2) if len(vid_embs) > 1 else 1
        use_ivfpq = len(vid_embs) > 1
        image_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image.bin", meta_path="data/faiss_image.pkl", nlist=nlist, use_ivfpq=use_ivfpq)
        if use_ivfpq:
            image_searcher.train(vid_embs)
        image_searcher.add_batch(vid_embs, vid_metas)
        image_searcher.save()
        print(f"Đã build index cho video (frame đầu). Số mẫu: {len(vid_embs)}, nlist: {nlist}, use_ivfpq: {use_ivfpq}")
except ImportError:
    print("Thiếu thư viện cv2, hãy cài bằng: pip install opencv-python")

# Build index cho ảnh (nếu có data/images/)
img_dir = "data/images"
if os.path.exists(img_dir):
    img_embs = []
    img_metas = []
    for fname in os.listdir(img_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            fpath = os.path.join(img_dir, fname)
            emb = get_image_embedding(fpath)
            img_embs.append(emb)
            img_metas.append({"file": fname, "description": f"Ảnh {fname}"})
    if img_embs:
        nlist = min(8, len(img_embs) // 2) if len(img_embs) > 1 else 1
        use_ivfpq = len(img_embs) > 1
        image_searcher = FaissMultiModalSearch(dim=512, index_path="data/faiss_image_img.bin", meta_path="data/faiss_image_img.pkl", nlist=nlist, use_ivfpq=use_ivfpq)
        if use_ivfpq:
            image_searcher.train(img_embs)
        image_searcher.add_batch(img_embs, img_metas)
        image_searcher.save()
        print(f"Đã build index cho ảnh. Số mẫu: {len(img_embs)}, nlist: {nlist}, use_ivfpq: {use_ivfpq}") 