from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os

clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

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
        print("Embedding shape:", emb.shape)
    else:
        print("Vui lòng đặt file ảnh mẫu tại data/sample.jpg để test.") 