# ML Image Similarity Search

A visual similarity search engine for WikiArt images using OpenCLIP embeddings, Qdrant vector search, and PostgreSQL metadata storage.

## Overview

This project takes an input image and returns the Top 5 visually similar artworks from a 10,000-image WikiArt dataset.

The system uses:

- OpenCLIP ViT-B/32 for image embeddings
- Qdrant for vector storage and similarity search
- PostgreSQL for image metadata
- Streamlit for the web interface
- Docker Compose for running the complete application stack

## Architecture

```text
                    Query Image
                         |
                         v
                  +-------------+
                  |  Streamlit  |
                  +------+------+
                         |
                         v
                  +-------------+
                  |   OpenCLIP  |
                  |  ViT-B/32   |
                  +------+------+
                         |
                    512-d embedding
                         |
                         v
                  +-------------+
                  |   Qdrant    |
                  |  Vector DB  |
                  +------+------+
                         |
                    Top-K IDs
                         |
                         v
                  +-------------+
                  | PostgreSQL  |
                  |  Metadata   |
                  +------+------+
                         |
                         v
                    Top 5 Results
```

## Embedding Model

The project uses:

```text
Model: ViT-B/32
Pretrained: laion2b_s34b_b79k
Embedding dimension: 512
Distance metric: Cosine similarity
```

Embeddings are L2-normalized before being stored in Qdrant.

## Dataset

The dataset consists of 10,000 processed WikiArt images.

The subset was selected from a larger 30,000-image dataset using a deterministic random seed (42).

Dataset structure:

```text
dataset_10k/
├── Abstract_Expressionism/
├── Baroque/
├── Cubism/
├── Impressionism/
├── Northern_Renaissance/
└── ...
```

The dataset is not included in the repository because of its size and image licensing restrictions.

The artwork images originate from WikiArt.org and are used for non-commercial research purposes.

## Project Structure

```text
ml-image-similarity/
│
├── src/
│   ├── model.py
│   ├── embed.py
│   ├── search.py
│   ├── database.py
│   ├── qdrant_setup.py
│   ├── postgres_setup.py
│   └── test_model.py
│
├── demo/
│   └── app.py
│
├── dataset_10k/
│   └── ...                  # Local dataset, not tracked
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── .gitattributes
└── README.md
```

## How It Works

### 1. Generate Embeddings

Images are loaded from the WikiArt dataset and passed through OpenCLIP.

```text
Image
  |
  v
Preprocessing
  |
  v
OpenCLIP ViT-B/32
  |
  v
512-dimensional embedding
  |
  v
L2 normalization
```

Run:

```bash
python src/embed.py
```

The generated embeddings are stored in Qdrant and the corresponding metadata is stored in PostgreSQL.

### 2. Qdrant

Qdrant stores the 512-dimensional image vectors.

Collection:

```text
wikiart_embeddings
```

Configuration:

```text
Vector size: 512
Distance: COSINE
```

### 3. PostgreSQL

PostgreSQL stores metadata associated with each image.

Schema:

```text
images
├── id
├── qdrant_id
├── image_path
├── filename
├── artist
└── style
```

Qdrant stores the vector while PostgreSQL stores the relational metadata.

### 4. Similarity Search

For a query image:

```text
Query image
    |
    v
OpenCLIP embedding
    |
    v
Qdrant similarity search
    |
    v
Top-K vector IDs + similarity scores
    |
    v
PostgreSQL metadata lookup
    |
    v
Top 5 artworks
```

Run the command-line search:

```bash
python src/search.py
```

Then provide an image path:

```text
test.jpg
```

Example:

```text
Top 5 Similar Images:
------------------------------------------------------------

1. dataset_10k/Northern_Renaissance/albrecht-durer_virgin-suckling-the-child-1503.jpg
   Artist: albrecht-durer
   Style: Northern_Renaissance
   Similarity: 0.7441

2. dataset_10k/Early_Renaissance/pietro-perugino_portrait-of-a-young-man-1.jpg
   Artist: pietro-perugino
   Style: Early_Renaissance
   Similarity: 0.7360
```

## Streamlit Demo

Run locally:

```bash
streamlit run demo/app.py
```

Then open:

```text
http://localhost:8501
```

The interface allows an image to be uploaded and displays the Top 5 visually similar artworks with:

- Artwork title
- Artist
- Similarity score
- Artwork image

## Docker

The complete system can be run using Docker Compose.

The stack contains:

```text
                 Streamlit Application
                         |
             +-----------+-----------+
             |                       |
             v                       v
          Qdrant                PostgreSQL
        Vector Store             Metadata
```

### Start the services

```bash
docker compose up -d
```

Check the services:

```bash
docker compose ps
```

Expected services:

```text
image-similarity-app
image-similarity-qdrant
image-similarity-postgres
```

The application is available at:

```text
http://localhost:8501
```

### First-Time Setup

For a fresh environment, Qdrant and PostgreSQL need to be initialized and the dataset needs to be indexed before performing searches.

Start Qdrant and PostgreSQL:

```bash
docker compose up -d qdrant postgres
```

Initialize the Qdrant collection:

```bash
python src/qdrant_setup.py
```

Initialize the PostgreSQL schema:

```bash
python src/postgres_setup.py
```

Generate and index the image embeddings:

```bash
python src/embed.py
```

The indexing pipeline:

1. Loads images from `dataset_10k/`
2. Generates 512-dimensional OpenCLIP embeddings
3. L2-normalizes the embeddings
4. Stores vectors in Qdrant
5. Stores image metadata in PostgreSQL

After indexing is complete, start the application:

```bash
docker compose up -d app
```

Then open:

```text
http://localhost:8501
```

### Stop the Services

```bash
docker compose down
```

## Environment Configuration

When running inside Docker Compose, the application connects to the services using their Compose service names:

```text
QDRANT_HOST=qdrant
QDRANT_PORT=6333

POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=image_similarity
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

When running locally, the default values are:

```text
QDRANT_HOST=localhost
QDRANT_PORT=6333

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=image_similarity
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

These values can be overridden using environment variables.

## Technology Stack

| Component | Technology |
|---|---|
| Image Embeddings | OpenCLIP ViT-B/32 |
| Vector Database | Qdrant |
| Relational Database | PostgreSQL |
| Similarity Metric | Cosine similarity |
| Web Interface | Streamlit |
| Containerization | Docker |
| Language | Python |

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For local development, make sure Qdrant and PostgreSQL are running.

Start them with:

```bash
docker compose up -d qdrant postgres
```

Initialize the databases:

```bash
python src/qdrant_setup.py
python src/postgres_setup.py
```

Generate the image embeddings:

```bash
python src/embed.py
```

Run the application:

```bash
streamlit run demo/app.py
```

## Notes

The current implementation indexes a deterministic 10,000-image subset for practical local development and testing.

The indexing pipeline can be scaled to the complete dataset without changing the search architecture.

The previous NumPy-based embedding storage was replaced by Qdrant for vector storage and PostgreSQL for metadata management.

The architecture separates vector search from relational metadata storage:

```text
OpenCLIP
   |
   v
Qdrant
(vector similarity)
   |
   v
PostgreSQL
(metadata)
```

This allows the vector search layer and metadata layer to be managed independently as the system scales.
