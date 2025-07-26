from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os

# Sử dụng CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb[0].cpu().numpy()

if __name__ == "__main__":
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.jpg')
    if os.path.exists(sample_path):
        emb = get_image_embedding(sample_path)
        print("CLIP Embedding shape:", emb.shape)
    else:
        print("Vui lòng đặt file ảnh mẫu tại data/sample.jpg để test.") 