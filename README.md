# AI Challenge HCM - Hệ thống tìm kiếm đa phương thức

## 🎯 Tổng quan

Hệ thống tìm kiếm đa phương thức (multimodal search) có khả năng tìm kiếm qua text, hình ảnh và video. Hệ thống sử dụng:

- **Text Embedding**: SentenceTransformer với model tiếng Việt
- **Image Embedding**: CLIP model cho xử lý hình ảnh
- **FAISS**: Facebook AI Similarity Search cho tìm kiếm nhanh
- **FastAPI**: Backend API
- **React**: Frontend giao diện

## 🚀 Khởi động nhanh

### Cách 1: Sử dụng file batch (Khuyến nghị)
```bash
# Chạy file batch để tự động setup và khởi động
run_system.bat
```

### Cách 2: Chạy từng bước
```bash
# 1. Tạo dữ liệu mẫu
python create_sample_data.py

# 2. Build indexes
python src/build_index_fixed.py

# 3. Khởi động backend
python -m uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload

# 4. Khởi động frontend (trong terminal khác)
cd frontend
npm install
npm start
```

## 📁 Cấu trúc dự án

```
Chatbot/
├── data/                    # Dữ liệu
│   ├── text/               # File text
│   ├── vid/                # Hình ảnh và video
│   └── audio/              # File audio
├── src/                    # Backend code
│   ├── api.py              # FastAPI endpoints
│   ├── build_index_fixed.py # Build FAISS indexes
│   ├── text_pipeline.py    # Text processing
│   ├── image_pipeline.py   # Image processing
│   └── faiss_pipeline.py   # FAISS operations
├── frontend/               # React frontend
│   └── src/
│       └── App.js          # Main React component
├── requirements.txt         # Python dependencies
├── create_sample_data.py   # Tạo dữ liệu mẫu
├── build_and_run.py        # Script khởi động hoàn chỉnh
└── run_system.bat          # Batch file khởi động
```

## 🎬 Dataset mẫu

Hệ thống bao gồm dataset mẫu về các chủ đề công nghệ:

### Text Files
- `t1.txt`: AI và Machine Learning
- `t2.txt`: Blockchain và Tiền điện tử  
- `t3.txt`: IoT và Thành phố thông minh

### Images
- `ai_ml.jpg`: Hình ảnh về AI
- `blockchain.jpg`: Hình ảnh về Blockchain
- `iot_smartcity.jpg`: Hình ảnh về IoT
- `data_science.jpg`: Hình ảnh về Data Science

### Video
- `ai_demo.mp4`: Video demo về AI

## 🔧 API Endpoints

### Text Search
```bash
POST /search_text
{
  "query": "AI machine learning",
  "top_k": 5
}
```

### Image Search
```bash
POST /search_image
Content-Type: multipart/form-data
file: [image_file]
top_k: 5
```

### Health Check
```bash
GET /health
```

## 🎯 Tính năng chính

### 1. Text Search
- Tìm kiếm semantic trong các file text
- Hỗ trợ tiếng Việt
- Kết quả được sắp xếp theo độ tương đồng

### 2. Image Search
- Tìm kiếm hình ảnh tương tự
- Hỗ trợ nhiều định dạng: JPG, PNG, MP4
- Trích xuất frame từ video
- Hiển thị hình ảnh base64

### 3. Unified Results
- Kết quả kết hợp từ cả text và image
- Hiển thị score tương đồng
- Metadata chi tiết cho mỗi kết quả

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Node.js 14+
- Git

### Dependencies

#### Python
```bash
pip install -r requirements.txt
```

#### Node.js
```bash
cd frontend
npm install
```

## 🧪 Testing

### Test API
```bash
# Test text search
curl -X POST "http://localhost:8001/search_text" \
  -H "Content-Type: application/json" \
  -d '{"query": "AI", "top_k": 3}'

# Test image search
curl -X POST "http://localhost:8001/search_image" \
  -F "file=@data/vid/ai_ml.jpg" \
  -F "top_k=3"
```

### Test Frontend
1. Mở http://localhost:3000
2. Thử text search với từ khóa "AI", "blockchain", "IoT"
3. Upload hình ảnh để test image search

## 🔍 Debug và Troubleshooting

### Kiểm tra trạng thái hệ thống
```bash
python check_status.py
```

### Logs
- Backend logs: Terminal chạy uvicorn
- Frontend logs: Terminal chạy npm start
- Browser console: F12 trong trình duyệt

### Các lỗi thường gặp

#### 1. "Failed to fetch"
- Kiểm tra backend có chạy không (port 8001)
- Kiểm tra CORS settings

#### 2. "Index not found"
- Chạy lại `python src/build_index_fixed.py`
- Kiểm tra file trong thư mục `data/`

#### 3. "Module not found"
- Cài đặt lại dependencies: `pip install -r requirements.txt`

## 📊 Performance

### Index Sizes
- Text index: ~768 dimensions
- Image index: ~512 dimensions
- Search speed: < 100ms cho 1000+ documents

### Memory Usage
- Backend: ~500MB RAM
- Frontend: ~100MB RAM
- Indexes: ~50MB RAM

## 🔮 Roadmap

### V1.1 (Planned)
- [ ] Audio search support
- [ ] Advanced filtering
- [ ] Batch upload
- [ ] Export results

### V1.2 (Future)
- [ ] Real-time search
- [ ] Multi-language support
- [ ] Cloud deployment
- [ ] Mobile app

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 📞 Support

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra logs
2. Chạy `python check_status.py`
3. Tạo issue với thông tin chi tiết

---

**🎉 Chúc bạn sử dụng hệ thống hiệu quả!** 