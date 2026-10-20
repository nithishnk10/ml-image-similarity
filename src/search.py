from database import get_qdrant_client, get_postgres_connection
from model import get_embedding
from qdrant_setup import COLLECTION_NAME


TOP_K = 5


def search(query_image, top_k=TOP_K):
    qdrant = get_qdrant_client()
    postgres = get_postgres_connection()

    try:
        query_embedding = get_embedding(query_image)

        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
        ).points

        if not search_results:
            return []

        qdrant_ids = [result.id for result in search_results]

        with postgres.cursor() as cursor:
            cursor.execute(
                """
                SELECT qdrant_id, image_path, filename, artist, style
                FROM images
                WHERE qdrant_id = ANY(%s)
                """,
                (qdrant_ids,)
            )

            rows = cursor.fetchall()

        metadata = {
            row[0]: {
                "image": row[1],
                "filename": row[2],
                "artist": row[3],
                "style": row[4],
            }
            for row in rows
        }

        results = []

        for result in search_results:
            item = metadata.get(result.id)

            if item:
                results.append({
                    "image": item["image"],
                    "filename": item["filename"],
                    "artist": item["artist"],
                    "style": item["style"],
                    "similarity": float(result.score),
                })

        return results

    finally:
        postgres.close()


def main():
    query_image = input("Enter query image path: ").strip()

    results = search(query_image)

    print("\nTop 5 Similar Images:")
    print("-" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['image']}")
        print(f"   Artist: {result['artist']}")
        print(f"   Style: {result['style']}")
        print(f"   Similarity: {result['similarity']:.4f}")


if __name__ == "__main__":
    main()