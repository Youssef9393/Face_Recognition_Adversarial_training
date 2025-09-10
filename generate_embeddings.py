# generate_embeddings.py
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

model = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

names = ['youssef', 'hisham']
for name in names:
    img = Image.open(f"data/{name}.jpg")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor)
    torch.save(embedding, f"embeddings/{name}.pt")
    print(f"Saved embedding for {name}")
