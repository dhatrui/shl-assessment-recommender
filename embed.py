import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

if os.path.exists("shl.index"):
    print("Index already exists, regenerating...")

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/shl_catalogue.json", "r") as f:
    catalogue = json.load(f)

documents = []
for item in catalogue:
    documents.append(
        f"{item['name']} | {item['category']} | "
        f"{', '.join(item['measures'])} | "
        f"{', '.join(item['use_cases'])}"
    )

embeddings = model.encode(documents, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "shl.index")

print("Local embeddings + FAISS index created")
