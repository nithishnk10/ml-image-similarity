from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


COLLECTION_NAME = "wikiart_embeddings"

client = QdrantClient(
    host="localhost",
    port=6333
)


def create_collection():
    existing_collections = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        )
    )

    print(f"Created collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    create_collection()