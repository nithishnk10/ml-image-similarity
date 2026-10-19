# 🎨 WikiArt Image Similarity Search

A visual similarity search engine for artwork images using pretrained image embeddings from OpenCLIP.

Given a query image, the system generates its visual embedding and retrieves the Top 5 most visually similar artworks from a WikiArt image collection using cosine similarity.

## Features

- Pretrained OpenCLIP image embeddings
- 10,000-image WikiArt subset for the search index
- Batch embedding generation
- Cosine similarity based image retrieval
- Top 5 similar artwork results with similarity scores
- Streamlit web interface
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
                  Streamlit Interface
```

## Project Structure

```text
ml-image-similarity/
│
├── src/
│   ├── app.py          # Streamlit web application
│   ├── embed.py        # Generate embeddings for dataset images
│   ├── model.py        # OpenCLIP model and preprocessing
│   ├── search.py       # Similarity search
│   └── test_model.py   # Model embedding test
│
├── embeddings/         # Generated embeddings (not committed)
├── dataset_10k/        # 10,000-image dataset (not committed)
│
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Model

The project uses:

- **Model:** OpenCLIP ViT-B/32
- **Pretrained weights:** `laion2b_s34b_b79k`
- **Embedding dimension:** 512
- **Similarity metric:** Cosine similarity

The image embeddings are L2-normalized, allowing cosine similarity to be calculated efficiently using the dot product.

## Dataset

The project uses a 10,000-image subset of the WikiArt dataset.

The original downloaded dataset contains approximately 30,000 processed WikiArt images. A 10,000-image subset was selected for a practical local demonstration and indexing time.

The dataset is not included in this repository because of its size and licensing restrictions.

Place the dataset in:

```text
dataset_10k/
```

The expected structure is:

```text
dataset_10k/
├── Abstract_Expressionism/
├── Baroque/
├── Cubism/
├── Impressionism/
├── Renaissance/
└── ...
```

The WikiArt dataset should be used according to its applicable licensing and usage terms.

## Installation

Clone the repository:

```bash
git clone https://github.com/nithishnk10/ml-image-similarity.git
cd ml-image-similarity
```

Create a virtual environment:

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

After placing the dataset inside `dataset_10k/`, run:

```powershell
python src/embed.py
```

This generates:

```text
embeddings/
├── embeddings.npy
└── paths.json
```

The embedding matrix contains one 512-dimensional vector per valid image.

## Test the Model

To verify that the OpenCLIP model can generate an embedding:

```powershell
python src/test_model.py
```

The test prints the embedding shape and the first few embedding values.

## Command-Line Search

Run:

```powershell
python src/search.py
```

Enter the path to a query image when prompted:

```text
Enter query image path: test.jpg
```

The system returns the Top 5 most similar artworks with their cosine similarity scores.

Example:

```text
Top 5 Similar Images:
------------------------------------------------------------

1. dataset_10k\Northern_Renaissance\...
   Similarity: 0.7441

2. dataset_10k\Early_Renaissance\...
   Similarity: 0.7360

3. dataset_10k\High_Renaissance\...
   Similarity: 0.7313

4. dataset_10k\Early_Renaissance\...
   Similarity: 0.7303

5. dataset_10k\Northern_Renaissance\...
   Similarity: 0.7220
```

## Streamlit Demo

Start the web application:

```powershell
streamlit run src/app.py
```

Then open the local Streamlit URL shown in the terminal.

The interface allows the user to:

1. Upload an image.
2. Generate its visual embedding.
3. Search the precomputed embedding index.
4. Display the Top 5 visually similar artworks.
5. View the similarity score for each result.

## Docker

A Dockerfile is included for containerized execution.

Build the image:

```bash
docker build -t ml-image-similarity .
```

Run the application:

```bash
docker run -p 8501:8501 ml-image-similarity
```

For a complete local deployment, the dataset and generated embeddings should be made available to the container because these large files are intentionally excluded from the Git repository.

## Similarity Search

The search process is:

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

This makes the similarity computation efficient using a matrix-vector multiplication.

## Performance

The embedding generation was performed on a CPU-based laptop.

For the selected 10,000-image dataset:

- Images indexed: 10,000
- Embedding dimension: 512
- Batch size: 16
- Embedding generation time: approximately 20 minutes on the local CPU setup

Once embeddings are generated, similarity search is performed against the precomputed embedding matrix without re-embedding the dataset.

## Limitations

- The current implementation uses a flat similarity search over the complete embedding matrix.
- The dataset and generated embeddings are not included in Git because of their size.
- First-time model execution requires downloading the pretrained OpenCLIP weights.
- Retrieval quality depends on the pretrained embedding model and the available dataset.
- The current implementation is designed as a lightweight demonstration rather than a production-scale vector search system.

## Future Improvements

Possible extensions include:

- FAISS or another vector database for large-scale retrieval
- GPU acceleration
- Larger WikiArt index
- Metadata filtering by artist or art style
- More embedding models such as DINOv2
- Result caching
- Improved Docker deployment
- API endpoint using FastAPI

## Technologies

- Python
- PyTorch
- OpenCLIP
- NumPy
- Pillow
- Streamlit
- Docker
```