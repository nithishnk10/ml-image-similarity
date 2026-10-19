import json
from pathlib import Path

import numpy as np

from model import get_embedding


EMBEDDINGS_FILE = Path("embeddings/embeddings.npy")
PATHS_FILE = Path("embeddings/paths.json")

TOP_K = 5


def load_data():
    embeddings = np.load(EMBEDDINGS_FILE)

    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        paths = [p.replace("\\", "/") for p in json.load(f)]

    return embeddings, paths


def search(query_image, top_k=TOP_K):
    embeddings, paths = load_data()

    query_embedding = get_embedding(query_image)

    # Embeddings are normalized, so dot product = cosine similarity
    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "image": paths[index],
            "similarity": float(scores[index])
        })

    return results


def main():
    query_image = input("Enter query image path: ").strip()

    results = search(query_image)

    print("\nTop 5 Similar Images:")
    print("-" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['image']}")
        print(f"   Similarity: {result['similarity']:.4f}")


if __name__ == "__main__":
    main()