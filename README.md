# 🎨 WikiArt Image Similarity Search

A visual similarity search engine for artwork images using pretrained image embeddings from OpenCLIP.

Given a query image, the system generates its visual embedding and retrieves the Top 5 most visually similar artworks from a 10,000-image WikiArt index using cosine similarity.

## Features

- Pretrained OpenCLIP image embeddings
- 10,000-image WikiArt search index
- Batch embedding generation
- Cosine similarity based image retrieval
- Top 5 similar artwork results with similarity scores
- Streamlit web demo
- Docker support
- CPU-compatible implementation

## Architecture

```text
                    Query Image
                         │
                         ▼
                 OpenCLIP ViT-B/32
                         │
                         ▼
                  Image Embedding
                         │
                         ▼
              Cosine Similarity Search
                         │
                         ▼
                 Top 5 Similar Images
                         │
                         ▼
                  Streamlit Demo
```

## Project Structure

```text
ml-image-similarity/
│
├── src/
│   ├── embed.py        # Generate embeddings for dataset images
│   ├── search.py       # Similarity search
│   ├── model.py        # OpenCLIP model and preprocessing
│   └── test_model.py   # Model embedding test
│
├── embeddings/
│   ├── embeddings.npy  # Precomputed 10k image embeddings
│   └── paths.json      # Image paths corresponding to embeddings
│
├── demo/
│   └── app.py          # Streamlit demo
│
├── requirements.txt
├── README.md
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Model

The project uses:

- **Model:** OpenCLIP ViT-B/32
- **Pretrained weights:** `laion2b_s34b_b79k`
- **Embedding dimension:** 512
- **Similarity metric:** Cosine similarity

The image embeddings are L2-normalized. Therefore, cosine similarity can be efficiently calculated using the dot product.

## Dataset

The project uses a 10,000-image subset of the processed WikiArt dataset.

The original dataset contains approximately 30,000 processed WikiArt images. A 10,000-image subset was selected for practical local indexing and demonstration.

The dataset itself is **not included in this repository** because of its size and applicable licensing restrictions.

Place the dataset in:

```text
dataset_10k/
```

with the artwork folders inside it.

Example:

```text
dataset_10k/
├── Abstract_Expressionism/
├── Baroque/
├── Cubism/
├── Impressionism/
├── Renaissance/
└── ...
```

The dataset should be obtained and used according to its applicable licensing and usage terms.

## Precomputed Embeddings

The repository includes the generated embedding index for the 10,000-image dataset:

```text
embeddings/
├── embeddings.npy
└── paths.json
```

`embeddings.npy` contains the 512-dimensional normalized embedding for each indexed image.

`paths.json` contains the corresponding image paths.

The actual dataset images are not included in the repository, so the dataset must be placed locally if the search or demo is to be executed.

## Installation

Clone the repository:

```bash
git clone https://github.com/nithishnk10/ml-image-similarity.git
cd ml-image-similarity
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Generate Embeddings

If you want to regenerate the embedding index from the dataset, place the images inside:

```text
dataset_10k/
```

Then run:

```powershell
python src/embed.py
```

This generates:

```text
embeddings/
├── embeddings.npy
└── paths.json
```

The repository already contains the generated 10,000-image embedding index, so regeneration is not required for the existing index.

## Test the Model

To verify that the OpenCLIP model can generate an image embedding:

```powershell
python src/test_model.py
```

The test prints the embedding shape and the first few embedding values.

Expected embedding shape:

```text
(512,)
```

## Command-Line Search

Run:

```powershell
python src/search.py
```

Enter the path to a query image:

```text
Enter query image path: test.jpg
```

The system returns the Top 5 most similar artworks with their cosine similarity scores.

Example:

```text
Top 5 Similar Images:
------------------------------------------------------------

1. dataset_10k/Northern_Renaissance/...
   Similarity: 0.7441

2. dataset_10k/Early_Renaissance/...
   Similarity: 0.7360

3. dataset_10k/High_Renaissance/...
   Similarity: 0.7313

4. dataset_10k/Early_Renaissance/...
   Similarity: 0.7303

5. dataset_10k/Northern_Renaissance/...
   Similarity: 0.7220
```

## Streamlit Demo

The main demonstration application is located in:

```text
demo/app.py
```

Start the demo with:

```powershell
streamlit run demo/app.py
```

The interface allows the user to:

1. Upload an image.
2. Generate its visual embedding.
3. Search the precomputed embedding index.
4. Retrieve the Top 5 visually similar artworks.
5. View the similarity score for each result.


## Docker

A Dockerfile is included for containerized deployment.

Build the Docker image:

```bash
docker build -t ml-image-similarity .
```

### Windows PowerShell

```powershell
docker run --rm -p 8501:8501 -v "${PWD}\dataset_10k:/app/dataset_10k" ml-image-similarity
```

### Linux / macOS

```bash
docker run --rm -p 8501:8501 -v "$(pwd)/dataset_10k:/app/dataset_10k" ml-image-similarity
```

The dataset images are intentionally excluded from the repository.

For a complete containerized demo, the dataset should be made available to the container at:

```text
/app/dataset_10k/
```

The pretrained OpenCLIP model weights may also be downloaded when the application first loads the model.
```


## Similarity Search Process

```text
Query Image
     │
     ▼
Image Preprocessing
     │
     ▼
OpenCLIP ViT-B/32
     │
     ▼
512-D Normalized Embedding
     │
     ▼
Compare Against Dataset Embeddings
     │
     ▼
Cosine Similarity
     │
     ▼
Sort Scores
     │
     ▼
Top 5 Results
```

Since both query and dataset embeddings are normalized:

```text
cosine_similarity = query_embedding · image_embedding
```

This allows efficient similarity calculation using a matrix-vector multiplication.

## Performance

The embedding generation was performed on a CPU-based laptop.

For the selected 10,000-image dataset:

- Images indexed: 10,000
- Embedding dimension: 512
- Batch size: 16
- Embedding generation time: approximately 20 minutes on the local CPU setup

Once the embeddings are generated, similarity search operates against the precomputed embedding matrix rather than re-embedding the entire dataset for every query.

## Limitations

- The current implementation performs a flat search over the complete embedding matrix.
- The dataset images are not included in the repository.
- The repository includes the generated 10,000-image embedding index.
- First-time model execution requires downloading the pretrained OpenCLIP weights.
- Retrieval quality depends on the pretrained embedding model and available dataset.
- The implementation is intended as a lightweight demonstration rather than a production-scale vector search system.

## Future Improvements

Possible extensions include:

- FAISS or another vector database for large-scale retrieval
- GPU acceleration
- Larger WikiArt index
- Metadata filtering by artist or art style
- Alternative embedding models such as DINOv2
- Result caching
- FastAPI backend
- Production deployment

## Technologies

- Python
- PyTorch
- OpenCLIP
- NumPy
- Pillow
- Streamlit
- Docker
