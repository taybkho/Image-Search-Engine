from fastapi import FastAPI, File, UploadFile
import faiss
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import uvicorn
import torchvision.models as models

# Initialize FastAPI app
app = FastAPI()

# Load precomputed embeddings & filenames
embeddings = np.load("car_embeddings.npy").astype("float32")
file_names = np.load("car_filenames.npy")

# Load FAISS index
index = faiss.read_index("faiss_index_optimized.bin")

# Load ResNet50 for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

# Define preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image: Image.Image):
    """Extract feature vector from an image using ResNet50."""
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image).view(1, -1)
    return features.cpu().numpy()

@app.post("/search/")
async def search_similar(file: UploadFile = File(...)):
    """Search for similar images using FAISS."""
    # Load and process the uploaded image
    image = Image.open(file.file).convert("RGB")
    query_vector = extract_features(image)

    # Search in FAISS index
    distances, indices = index.search(query_vector, 5)
    similar_images = [file_names[i] for i in indices[0]]

    return {"query": file.filename, "matches": similar_images}

# Run the API with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
