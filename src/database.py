import os

import psycopg2
from qdrant_client import QdrantClient


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "image_similarity")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


def get_qdrant_client():
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT
    )


def get_postgres_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )


if __name__ == "__main__":
    qdrant = get_qdrant_client()
    print("Qdrant connection: OK")

    postgres = get_postgres_connection()
    print("PostgreSQL connection: OK")

    postgres.close()