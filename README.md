# AI Challenge 2025 - Trợ lý ảo truy xuất thông tin đa phương tiện

## 1. Mô tả dự án

Hệ thống trợ lý ảo thông minh hỗ trợ phân tích và truy xuất thông tin chuyên sâu trong dữ liệu lớn đa phương tiện (văn bản, hình ảnh, âm thanh). Hệ thống gồm các thành phần:
- Xử lý, trích xuất embedding cho text, image, audio
- Lưu trữ và truy vấn nhanh bằng FAISS
- API FastAPI cho truy vấn đa modal
- Giao diện chatbot React đơn giản

## 2. Cài đặt môi trường

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Chuẩn bị dữ liệu
- Đặt dữ liệu vào các thư mục:
  - `data/text/` (văn bản, mỗi dòng 1 entry)
  - `data/audio/` (file .mp3, .wav)
  - `data/vid/` (file .mp4, .avi, ...)
  - `data/images/` (file .jpg, .png, ... nếu có)
- Đảm bảo có file `vietnamese-stopwords-dash.txt` ở thư mục gốc.

## 4. Build index cho toàn bộ dữ liệu

```bash
python src/build_index.py
```
- Script sẽ tự động train và add batch embedding cho từng modal.
- Nếu thiếu thư viện, cài thêm theo hướng dẫn trong log.

## 5. Chạy backend API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```
- Các endpoint: `/search_text`, `/search_image`, `/search_audio`

## 6. Chạy frontend chatbot

```bash
cd frontend
npm install
npm start
```
- Truy cập: [http://localhost:3000](http://localhost:3000)

## 7. Test truy vấn
- Nhập truy vấn văn bản, upload ảnh hoặc audio trên giao diện.
- Kết quả trả về là metadata (tên file, mô tả, nội dung) của kết quả gần nhất.

## 8. Thành phần chính
- `src/text_pipeline.py`: Xử lý văn bản (pyvi + SentenceTransformer)
- `src/image_pipeline.py`: Xử lý hình ảnh (CLIP)
- `src/audio_pipeline.py`: Xử lý âm thanh (Wav2Vec2)
- `src/faiss_pipeline.py`: Lưu trữ/truy vấn embedding (FAISS IVF+PQ)
- `src/build_index.py`: Build index tự động cho tất cả modal
- `src/api.py`: API FastAPI
- `frontend/`: Giao diện chatbot React

## 9. Yêu cầu thư viện
- Xem file `requirements.txt` (Python)
- Frontend: React (cài bằng npm)

## 10. Tài liệu tham khảo
- [PhoBERT/SBERT Vietnamese](https://huggingface.co/VoVanPhuc/sup-SimCSE-VietNamese-phobert-base)
- [CLIP](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
- [Wav2Vec2 Vietnamese](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vi)
- [FAISS](https://github.com/facebookresearch/faiss)
- [pyvi](https://github.com/trungtv/pyvi)

---
**Chúc bạn thành công với AI Challenge!** 🚀 