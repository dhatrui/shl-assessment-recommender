import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/shl_catalogue.json", "r") as f:
    catalogue = json.load(f)

index = faiss.read_index("shl.index")

def recommend_assessments(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(catalogue[idx])

    return results


if __name__ == "__main__":
    query = "Graduate hiring for analytical roles"
    recs = recommend_assessments(query)

    for r in recs:
        print("\n - ", r["name"])
        print("Category:", r["category"])
        print("Use cases:", ", ".join(r["use_cases"]))
