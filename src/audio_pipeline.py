import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import os

processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vi")
model = Wav2Vec2Model.from_pretrained("nguyenvulebinh/wav2vec2-base-vi")

def get_audio_embedding(audio_path):
    speech, sr = sf.read(audio_path)
    if len(speech.shape) > 1:  # stereo -> mono
        speech = speech.mean(axis=1)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1)
    return emb[0].cpu().numpy()

if __name__ == "__main__":
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.wav')
    if os.path.exists(sample_path):
        emb = get_audio_embedding(sample_path)
        print("Embedding shape:", emb.shape)
    else:
        print("Vui lòng đặt file audio mẫu tại data/sample.wav để test.") 