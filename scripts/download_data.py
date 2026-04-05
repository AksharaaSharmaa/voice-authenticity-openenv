from datasets import load_dataset
import soundfile as sf
import os

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

dataset = load_dataset("garystafford/deepfake-audio-detection", split="train")

real_count = 0
fake_count = 0

for item in dataset:
    audio = item["audio"]
    label = item["label"]  # 0=real, 1=fake
    
    if label == 0 and real_count < 250:
        sf.write(f"data/real/real_{real_count:04d}.wav", 
                 audio["array"], audio["sampling_rate"])
        real_count += 1
        
    elif label == 1 and fake_count < 250:
        sf.write(f"data/fake/fake_{fake_count:04d}.wav",
                 audio["array"], audio["sampling_rate"])
        fake_count += 1
    
    if real_count >= 250 and fake_count >= 250:
        break

print(f"Downloaded: {real_count} real, {fake_count} fake")