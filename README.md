# AI Challenge HCM - Multimodal Virtual Assistant

Hệ thống tìm kiếm đa phương thức (text + image) sử dụng FAISS và AI models cho AI Challenge HCM 2025.

## 🎯 Tổng quan

Đây là hệ thống tìm kiếm thông minh có khả năng:
- **Tìm kiếm văn bản** tiếng Việt với semantic search
- **Tìm kiếm ảnh tương tự** dựa trên nội dung
- **Kết hợp kết quả** từ nhiều nguồn dữ liệu
- **Hiển thị điểm số** dựa trên độ tương đồng thực tế
- **Hỗ trợ đa phương thức** (text, image, video frames)

## 🚀 Tính năng chính

### 1. Text Search (Tìm kiếm văn bản)
- **CLIP model** cho text và image embedding
- **Cross-modal search** - tìm ảnh từ text query
- **Tìm kiếm trong file text** (.txt files)
- **Kết quả được sắp xếp** theo độ tương đồng

### 2. Image Search (Tìm kiếm ảnh)
- **CLIP model** cho image embedding
- **FAISS vector search** cho tìm kiếm nhanh
- **Hỗ trợ nhiều định dạng**: JPG, PNG, BMP
- **Trích xuất frames từ video** (MP4, AVI, MOV)
- **Hiển thị ảnh base64** trong kết quả

### 3. Cross-modal Search (Tìm kiếm đa phương thức)
- **Kết hợp text + image search**
- **Tìm ảnh liên quan** khi search text
- **Tìm text mô tả** khi search ảnh
- **Unified scoring system**

### 4. Real-time Scoring System
- **Distance-based scoring** từ FAISS
- **Sigmoid normalization** về thang điểm 0-100
- **Score theo loại file**:
  - Uploaded images: 100
  - Static images: 90-80
  - Video frames: 50-30
  - Text content: 100-80

## 🛠️ Cài đặt chi tiết

### Yêu cầu hệ thống
- **Python**: 3.8 hoặc cao hơn
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Storage**: 2GB trống cho models và indexes
- **OS**: Windows 10+, Linux, macOS

### Bước 1: Clone repository
```bash
git clone https://github.com/your-username/ai-challenge-hcm.git
cd ai-challenge-hcm/Chatbot
```

### Bước 2: Tạo virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

**Lưu ý**: Lần đầu cài đặt có thể mất 10-15 phút để download AI models.

### Bước 4: Chuẩn bị dữ liệu
```bash
# Tạo thư mục dữ liệu
mkdir -p data/text data/images data/vid data/audio

# Thêm file text vào data/text/
# Ví dụ: t1.txt, t2.txt, t3.txt

# Thêm ảnh vào data/images/
# Ví dụ: nấm.jpg, ai_ml.jpg, blockchain.jpg

# Thêm video vào data/vid/
# Ví dụ: ai_demo.mp4, presentation.mp4
```

### Bước 5: Build indexes
```bash
python src/build_index_fixed.py
```

**Output mong đợi**:
```
🚀 Building indexes for AI Challenge HCM...
📝 Building text index...
📁 Found 3 text files: ['t1.txt', 't2.txt', 't3.txt']
✅ Text index built successfully. Samples: 15, nlist: 4, use_ivfpq: False
🖼️ Building image index from video frames...
📹 Processing video: ai_demo.mp4 (120 frames, 4.0s)
✅ Extracted 5 frames from ai_demo.mp4
✅ Image index built successfully. Samples: 5, nlist: 2, use_ivfpq: False
🖼️ Building image index from static images...
🖼️ Processed image: nấm.jpg
🖼️ Processed image: ai_ml.jpg
✅ Static image index built successfully. Samples: 2, nlist: 1, use_ivfpq: False
🎉 Index building completed!
```

### Bước 6: Chạy hệ thống

#### Cách 1: Sử dụng script tự động (Khuyến nghị)
```bash
# Windows
run_system.bat

# Linux/Mac
chmod +x run_system.sh
./run_system.sh
```

#### Cách 2: Chạy thủ công
```bash
# Terminal 1: Backend API
python -m uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Frontend (nếu có)
cd frontend
npm install
npm start
```

## 📁 Cấu trúc project

```
Chatbot/
├── src/                    # Backend source code
│   ├── api.py             # FastAPI endpoints
│   ├── faiss_pipeline.py  # FAISS search engine
│   ├── text_pipeline.py   # Text processing
│   ├── image_pipeline.py  # Image processing
│   ├── build_index_fixed.py # Index building
│   └── __init__.py
├── data/                  # Data directory
│   ├── text/             # Text files (.txt)
│   ├── images/           # Static images (.jpg, .png)
│   ├── vid/              # Videos (.mp4, .avi)
│   ├── audio/            # Audio files
│   ├── faiss_text.bin    # Text search index
│   ├── faiss_text.pkl    # Text metadata
│   ├── faiss_image.bin   # Video frames index
│   ├── faiss_image.pkl   # Video metadata
│   ├── faiss_image_img.bin # Static images index
│   └── faiss_image_img.pkl # Static images metadata
├── frontend/             # React frontend
├── venv/                 # Virtual environment
├── README.md             # Documentation
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── LICENSE              # MIT License
├── .gitignore           # Git ignore rules
├── run_system.bat       # Windows startup
├── run_system.sh        # Linux/Mac startup
└── vietnamese-stopwords-dash.txt # Vietnamese stopwords
```

## 🔧 API Endpoints chi tiết

### 1. Text Search
```bash
POST /search_text
Content-Type: application/json

{
  "query": "nấm",
  "top_k": 10
}
```

**Response**:
```json
{
  "matched_files": [
    {
      "file": "t1.txt",
      "line": 3,
      "text": "Nấm là một loại thực phẩm giàu dinh dưỡng...",
      "description": "Nấm là một loại thực phẩm giàu dinh dưỡng...",
      "score": 85.2,
      "type": "text"
    }
  ]
}
```

### 2. Image Search
```bash
POST /search_image
Content-Type: multipart/form-data

file: <image_file>
top_k: 10
```

**Response**:
```json
{
  "matched_files": [
    {
      "file": "nấm.jpg",
      "description": "Ảnh nấm.jpg",
      "score": 95.8,
      "type": "static_image",
      "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
    }
  ]
}
```

### 3. Health Check
```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "text_searcher": true,
  "image_searcher": true,
  "text_index_size": 15,
  "image_index_size": 7
}
```

### 4. Debug Endpoints
```bash
GET /debug/videos          # Kiểm tra video files
GET /debug/static-images   # Kiểm tra static images
```

## 📊 Hệ thống điểm số (Scoring System)

### Distance → Score Conversion
Hệ thống sử dụng **FAISS distance** và chuyển đổi thành **score 0-100**:

| Distance | Score | Mức độ tương đồng |
|----------|-------|-------------------|
| 0.0      | ~100  | Perfect match     |
| 0.5      | ~82   | Very similar      |
| 1.0      | ~73   | Similar           |
| 2.0      | ~50   | Somewhat similar  |
| 5.0      | ~5    | Less similar      |

### Công thức chuyển đổi
```python
# Sigmoid function
score = 1.0 / (1.0 + exp(distance - 2.0)) * 100

# Linear fallback
score = max(0, 100 - distance * 10)
```

### Loại kết quả và điểm số
- **uploaded_image**: File upload (score: 100)
- **static_image**: Ảnh từ database (score: 90-80)
- **video_frame**: Frames từ video (score: 50-30)
- **text**: Nội dung text (score: 100-80)

## 🔍 Hướng dẫn sử dụng

### 1. Text Search
1. **Nhập từ khóa** vào ô tìm kiếm
2. **Nhấn "Tìm kiếm"**
3. **Xem kết quả**:
   - Text content với score
   - Ảnh liên quan (nếu có)
   - Metadata chi tiết

### 2. Image Search
1. **Upload ảnh** muốn tìm kiếm
2. **Nhấn "Tìm kiếm"**
3. **Xem kết quả**:
   - Ảnh tương tự từ database
   - Video frames liên quan
   - Score dựa trên similarity

### 3. Combined Search
1. **Nhập text** và **upload ảnh**
2. **Nhấn "Tìm kiếm"**
3. **Xem kết quả tổng hợp**:
   - Text results
   - Image results
   - Unified scoring

## 🎯 Ví dụ sử dụng

### Ví dụ 1: Tìm kiếm "nấm"
```
Input: "nấm"
Results:
- Text: "Nấm là thực phẩm giàu dinh dưỡng..." (Score: 85.2)
- Image: nấm.jpg (Score: 95.8)
- Video frame: Frame 3 từ video.mp4 (Score: 45.3)
```

### Ví dụ 2: Upload ảnh nấm
```
Input: Upload nấm.jpg
Results:
- Uploaded image: nấm.jpg (Score: 100.0)
- Similar images: nấm_trắng.jpg (Score: 92.1)
- Video frames: Frame 5 từ cooking.mp4 (Score: 78.4)
```

## 🐛 Troubleshooting

### Lỗi thường gặp và cách khắc phục

#### 1. "Text searcher not available"
**Nguyên nhân**: Index chưa được build
**Giải pháp**:
```bash
python src/build_index_fixed.py
```

#### 2. "Image searcher not available"
**Nguyên nhân**: Image index bị lỗi
**Giải pháp**:
```bash
# Kiểm tra file indexes
ls data/faiss_image*.bin

# Rebuild nếu cần
python src/build_index_fixed.py
```

#### 3. "Math range error"
**Nguyên nhân**: Distance quá lớn gây overflow
**Giải pháp**: Đã được fix trong code mới, restart server

#### 4. "Module not found"
**Nguyên nhân**: Dependencies chưa cài đặt
**Giải pháp**:
```bash
pip install -r requirements.txt
```

#### 5. Frontend không kết nối
**Nguyên nhân**: CORS hoặc port issues
**Giải pháp**:
- Kiểm tra backend chạy trên port 8001
- Kiểm tra CORS settings trong api.py
- Restart cả frontend và backend

#### 6. "No results found"
**Nguyên nhân**: Database trống hoặc query không phù hợp
**Giải pháp**:
- Kiểm tra dữ liệu trong data/
- Thử query khác
- Kiểm tra logs để debug

### Debug Commands
```bash
# Kiểm tra trạng thái hệ thống
curl http://localhost:8001/health

# Kiểm tra video files
curl http://localhost:8001/debug/videos

# Kiểm tra static images
curl http://localhost:8001/debug/static-images

# Test text search
curl -X POST "http://localhost:8001/search_text" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 5}'
```

## 📈 Performance

### Index Sizes
- **Text index**: 512 dimensions (CLIP)
- **Image index**: 512 dimensions (CLIP)
- **Search speed**: < 100ms cho 1000+ documents

### Memory Usage
- **Backend**: ~500MB RAM
- **Frontend**: ~100MB RAM
- **Indexes**: ~50MB RAM
- **Models**: ~200MB RAM

### Supported Formats
- **Text**: .txt files
- **Images**: .jpg, .jpeg, .png, .bmp
- **Videos**: .mp4, .avi, .mov, .mkv
- **Audio**: .mp3, .wav (planned)

## 🔮 Roadmap

### V1.1 (Planned)
- [ ] Audio search support
- [ ] Advanced filtering options
- [ ] Batch upload functionality
- [ ] Export results to PDF/CSV
- [ ] User authentication
- [ ] Search history

### V1.2 (Future)
- [ ] Real-time search
- [ ] Multi-language support
- [ ] Cloud deployment
- [ ] Mobile app
- [ ] Voice search
- [ ] Advanced AI models

## 🤝 Contributing

### Cách đóng góp
1. **Fork** repository
2. **Tạo feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Tạo Pull Request**

### Code Style
- **Python**: PEP 8, Black formatter
- **JavaScript**: ESLint, Prettier
- **Documentation**: Markdown, docstrings

### Testing
```bash
# Run tests
pytest

# Check code style
black src/
flake8 src/
```

## 📝 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👥 Team

- **AI Challenge HCM Team**
- **Multimodal Search System**
- **Vietnamese AI Assistant**
- **FastAPI + React Stack**

## 📞 Liên hệ

- **Email**: team@aichallengehcm.com
- **GitHub**: https://github.com/your-username/ai-challenge-hcm
- **Project**: AI Challenge HCM 2025
- **Documentation**: https://github.com/your-username/ai-challenge-hcm#readme

## 🙏 Acknowledgments

- **PhoBERT** cho Vietnamese text processing
- **CLIP** cho image understanding
- **FAISS** cho vector search
- **FastAPI** cho backend API
- **React** cho frontend UI
- **AI Challenge HCM 2025** cho cơ hội phát triển

---

**Made with ❤️ for AI Challenge HCM 2025**

*Hệ thống tìm kiếm đa phương thức thông minh cho tương lai AI Việt Nam* 