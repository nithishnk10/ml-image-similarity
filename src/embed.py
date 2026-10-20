from pathlib import Path
import hashlib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from qdrant_client.models import PointStruct

from model import model, preprocess, DEVICE
from database import get_qdrant_client, get_postgres_connection
from qdrant_setup import COLLECTION_NAME


DATASET_DIR = Path("dataset_10k")

BATCH_SIZE = 16


def get_image_paths():
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    return [
        path
        for path in DATASET_DIR.rglob("*")
        if path.suffix.lower() in extensions
    ]


def extract_metadata(image_path):
    filename = image_path.name
    style = image_path.parent.name

    artist = filename.split("_", 1)[0] if "_" in filename else None

    return {
        "filename": filename,
        "artist": artist,
        "style": style,
        "image_path": image_path.as_posix(),
    }


def generate_id(image_path):
    path_string = image_path.as_posix()

    hash_bytes = hashlib.sha256(
        path_string.encode("utf-8")
    ).digest()

    return int.from_bytes(hash_bytes[:8], byteorder="big") & ((1 << 63) - 1)


def generate_embeddings(image_paths):
    qdrant = get_qdrant_client()
    postgres = get_postgres_connection()

    all_embeddings = []
    all_metadata = []

    try:
        with postgres.cursor() as cursor:

            for start in tqdm(
                range(0, len(image_paths), BATCH_SIZE),
                desc="Generating embeddings"
            ):
                batch_paths = image_paths[start:start + BATCH_SIZE]

                images = []
                metadata_batch = []

                for image_path in batch_paths:
                    try:
                        image = Image.open(image_path).convert("RGB")
                        image_tensor = preprocess(image)

                        images.append(image_tensor)
                        metadata_batch.append(
                            extract_metadata(image_path)
                        )

                    except Exception as e:
                        print(f"Skipping {image_path}: {e}")

                if not images:
                    continue

                batch = torch.stack(images).to(DEVICE)

                with torch.no_grad():
                    embeddings = model.encode_image(batch)

                    embeddings = embeddings / embeddings.norm(
                        dim=-1,
                        keepdim=True
                    )

                embeddings_np = embeddings.cpu().numpy()

                points = []

                for embedding, metadata in zip(
                    embeddings_np,
                    metadata_batch
                ):
                    qdrant_id = generate_id(Path(metadata["image_path"]))

                    points.append(
                        PointStruct(
                            id=qdrant_id,
                            vector=embedding.tolist(),
                        )
                    )

                    cursor.execute(
                        """
                        INSERT INTO images
                        (qdrant_id, image_path, filename, artist, style)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (qdrant_id) DO NOTHING
                        """,
                        (
                            qdrant_id,
                            metadata["image_path"],
                            metadata["filename"],
                            metadata["artist"],
                            metadata["style"],
                        )
                    )

                    all_metadata.append(metadata)

                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )

                postgres.commit()

                all_embeddings.append(embeddings_np)

    finally:
        postgres.close()

    if not all_embeddings:
        return np.empty((0, 512), dtype=np.float32)

    return np.vstack(all_embeddings).astype(np.float32)


def main():
    image_paths = get_image_paths()

    print(f"Found {len(image_paths)} images.")

    if not image_paths:
        print("No images found in the dataset folder.")
        return

    embeddings = generate_embeddings(image_paths)

    print("\nEmbedding generation complete.")
    print(f"Embeddings shape: {embeddings.shape}")
    print("Embeddings stored in Qdrant.")
    print("Metadata stored in PostgreSQL.")


if __name__ == "__main__":
    main()