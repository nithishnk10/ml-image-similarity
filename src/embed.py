import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model import model, preprocess, DEVICE


# Dataset folder
DATASET_DIR = Path("dataset_10k")

# Output folder
OUTPUT_DIR = Path("embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
PATHS_FILE = OUTPUT_DIR / "paths.json"

# CPU-friendly batch size
BATCH_SIZE = 16


def get_image_paths():
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    return [
        path
        for path in DATASET_DIR.rglob("*")
        if path.suffix.lower() in extensions
    ]


def generate_embeddings(image_paths):
    all_embeddings = []
    valid_paths = []

    for start in tqdm(
        range(0, len(image_paths), BATCH_SIZE),
        desc="Generating embeddings"
    ):
        batch_paths = image_paths[start:start + BATCH_SIZE]

        images = []
        current_paths = []

        for image_path in batch_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image)

                images.append(image_tensor)
                current_paths.append(str(image_path))

            except Exception as e:
                print(f"Skipping {image_path}: {e}")

        if not images:
            continue

        batch = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            embeddings = model.encode_image(batch)

            # Normalize for cosine similarity
            embeddings = embeddings / embeddings.norm(
                dim=-1,
                keepdim=True
            )

        all_embeddings.append(embeddings.cpu().numpy())
        valid_paths.extend(current_paths)

    if not all_embeddings:
        return np.empty((0, 512), dtype=np.float32), []

    return np.vstack(all_embeddings).astype(np.float32), valid_paths


def main():
    image_paths = get_image_paths()

    print(f"Found {len(image_paths)} images.")

    if not image_paths:
        print("No images found in the dataset folder.")
        return

    embeddings, valid_paths = generate_embeddings(image_paths)

    np.save(EMBEDDINGS_FILE, embeddings)

    with open(PATHS_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_paths, f, indent=2)

    print("\nEmbedding generation complete.")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved to: {EMBEDDINGS_FILE}")
    print(f"Paths saved to: {PATHS_FILE}")


if __name__ == "__main__":
    main()