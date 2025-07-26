import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64

def create_sample_images():
    """Tạo các hình ảnh mẫu cho test"""
    
    # Tạo thư mục nếu chưa có
    os.makedirs("data/vid", exist_ok=True)
    
    # 1. Hình ảnh về AI
    img1 = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img1)
    draw.text((50, 50), "AI & Machine Learning", fill='black')
    draw.text((50, 100), "Artificial Intelligence", fill='blue')
    draw.text((50, 150), "Deep Learning", fill='green')
    draw.text((50, 200), "Neural Networks", fill='red')
    img1.save("data/vid/ai_ml.jpg")
    
    # 2. Hình ảnh về Blockchain
    img2 = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img2)
    draw.text((50, 50), "Blockchain Technology", fill='black')
    draw.text((50, 100), "Bitcoin", fill='orange')
    draw.text((50, 150), "Ethereum", fill='purple')
    draw.text((50, 200), "Smart Contracts", fill='darkgreen')
    img2.save("data/vid/blockchain.jpg")
    
    # 3. Hình ảnh về IoT
    img3 = Image.new('RGB', (400, 300), color='lightgreen')
    draw = ImageDraw.Draw(img3)
    draw.text((50, 50), "IoT & Smart City", fill='black')
    draw.text((50, 100), "Internet of Things", fill='darkblue')
    draw.text((50, 150), "Smart Sensors", fill='darkred')
    draw.text((50, 200), "Connected Devices", fill='darkgreen')
    img3.save("data/vid/iot_smartcity.jpg")
    
    # 4. Hình ảnh về Data Science
    img4 = Image.new('RGB', (400, 300), color='lightyellow')
    draw = ImageDraw.Draw(img4)
    draw.text((50, 50), "Data Science", fill='black')
    draw.text((50, 100), "Big Data Analytics", fill='purple')
    draw.text((50, 150), "Machine Learning", fill='blue')
    draw.text((50, 200), "Predictive Analytics", fill='green')
    img4.save("data/vid/data_science.jpg")
    
    print("✅ Đã tạo 4 hình ảnh mẫu")

def create_sample_video():
    """Tạo video mẫu"""
    
    # Tạo video về AI
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/vid/ai_demo.mp4', fourcc, 1.0, (400, 300))
    
    for i in range(30):  # 30 frames = 30 giây
        # Tạo frame với text khác nhau
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Thêm text
        cv2.putText(frame, f"AI Demo Frame {i+1}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Artificial Intelligence", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Machine Learning", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Deep Learning", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("✅ Đã tạo video mẫu ai_demo.mp4")

def create_audio_sample():
    """Tạo file audio mẫu (placeholder)"""
    # Tạo file text để đại diện cho audio
    audio_content = """
    Đây là file audio mẫu về công nghệ AI.
    Trí tuệ nhân tạo đang thay đổi thế giới.
    Machine learning và deep learning là những công nghệ quan trọng.
    Việt Nam đang phát triển mạnh mẽ trong lĩnh vực AI.
    """
    
    with open("data/audio/audio_sample.txt", "w", encoding="utf-8") as f:
        f.write(audio_content)
    
    print("✅ Đã tạo file audio mẫu (text format)")

if __name__ == "__main__":
    print("🎬 Đang tạo dữ liệu mẫu...")
    
    # Tạo các thư mục cần thiết
    os.makedirs("data/text", exist_ok=True)
    os.makedirs("data/vid", exist_ok=True)
    os.makedirs("data/audio", exist_ok=True)
    
    # Tạo dữ liệu mẫu
    create_sample_images()
    create_sample_video()
    create_audio_sample()
    
    print("🎉 Hoàn thành tạo dữ liệu mẫu!")
    print("\n📁 Cấu trúc dữ liệu:")
    print("data/text/ - Chứa các file text")
    print("data/vid/ - Chứa hình ảnh và video")
    print("data/audio/ - Chứa file audio") 