import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


with open("data/shl_catalogue.json", "r") as f:
    catalogue = json.load(f)

documents = []

for item in catalogue:
    text = f"""
    Name: {item['name']}
    Category: {item['category']}
    Measures: {', '.join(item['measures'])}
    Job Levels: {', '.join(item['job_levels'])}
    Use Cases: {', '.join(item['use_cases'])}
    Description: {item['description']}
    """
    documents.append(text.strip())


model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} documents")


def search_assessments(query, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "score": float(score),
            "document": documents[idx]
        })

    return results

def rag_recommendation(query, top_k=3):
    retrieved = search_assessments(query, top_k=top_k)

    response = "Recommended Assessments:\n\n"
    for i, res in enumerate(retrieved, 1):
        response += f"{i}. {res['document'].splitlines()[0]}\n"
        response += "   Reason: High semantic similarity to query requirements.\n\n"

    return response

query = "Assessment for graduate roles requiring numerical reasoning"
results = search_assessments(query, top_k=3)

for i, res in enumerate(results, 1):
    print(f"\nResult {i} (Score: {res['score']:.3f})")
    print(res["document"])


def precision_at_k(retrieved_docs, relevant_names, k):
    """
    retrieved_docs: list of retrieved result dicts
    relevant_names: set of ground-truth assessment names
    k: cutoff
    """
    retrieved_top_k = retrieved_docs[:k]

    hits = 0
    for res in retrieved_top_k:
        first_line = res["document"].splitlines()[0]
        name = first_line.replace("Name:", "").strip()
        if name in relevant_names:
            hits += 1

    return hits / k


evaluation_set = [
    {
        "query": "Assessment for graduate roles requiring numerical reasoning",
        "relevant": {
            "Graduate Potential Assessment",
            "Verify Numerical Ability"
        }
    },
    {
        "query": "Cognitive tests for problem solving and logical reasoning",
        "relevant": {
            "Verify Inductive Reasoning",
            "Graduate Potential Assessment"
        }
    },
    {
        "query": "Assessments for entry-level analytical and finance roles",
        "relevant": {
            "Verify Numerical Ability"
        }
    }
]


k = 3
scores = []

for sample in evaluation_set:
    retrieved = search_assessments(sample["query"], top_k=k)
    p_at_k = precision_at_k(retrieved, sample["relevant"], k)
    scores.append(p_at_k)

    print(f"\nQuery: {sample['query']}")
    print(f"Precision@{k}: {p_at_k:.2f}")

avg_precision = sum(scores) / len(scores)
print(f"\nAverage Precision@{k}: {avg_precision:.2f}")