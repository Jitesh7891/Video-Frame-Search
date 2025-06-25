import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

# Qdrant setup
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "frames"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Load CLIP model once
_model_name = "openai/clip-vit-base-patch32"
_processor = CLIPProcessor.from_pretrained(_model_name)
_model     = CLIPModel.from_pretrained(_model_name).eval().to("cpu")  # or "cuda"

def create_collection():
    # Recreate with CLIP’s 512-dim vectors
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

def compute_feature_vector(image_path: str) -> list[float]:
    """
    Load image, preprocess via CLIPProcessor, encode with CLIPModel,
    L2-normalize and return a 512-dim list.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = _model.get_image_features(**inputs)
    # L2 normalize
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.squeeze().cpu().tolist()

def store_vector(point_id: int, vector: list[float]):
    pt = PointStruct(
        id=point_id,
        vector=vector,
        payload={"image": f"frame_{point_id}.jpg"}
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[pt])

def search_similar(vector: list[float], limit: int = 5):
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
        with_payload=True
    )
