# Vector Databases and How Similarity Search Works  

## 1. Introduction  

- **Why now?**  
  The volume of unstructured data—text, images, audio—has exploded. Traditional keyword search cannot capture the underlying *meaning* of these assets, which has driven the rise of **semantic search**.  

- **What is a vector?**  
  In machine learning, an *embedding* is a dense, fixed‑length numeric array (e.g., a 384‑dim float32) that encodes the semantic content of a raw object. Similar items map to nearby points in this high‑dimensional space.  

- **Roadmap**  
  1. Turning raw data into embeddings.  
  2. Core similarity‑search concepts.  
  3. Architecture of vector databases.  
  4. Real‑world patterns and operational tips.  
  5. Closing thoughts and next steps.  

---  

## 2. From Raw Data to Embeddings  

### 2.1. Embedding models  

| Modality | Popular models | Typical output dim |
|----------|----------------|-------------------|
| Text     | BERT, RoBERTa, Sentence‑Transformers | 256 – 768 |
| Image    | CLIP (ViT‑B/32), OpenCLIP            | 512 |
| Audio    | wav2vec 2.0, Whisper encoder          | 256 – 1024 |

These models are usually **pre‑trained** and can be fine‑tuned for domain‑specific vocabularies.  

### 2.2. Dimensionality and normalization  

- **Dimensionality** determines storage cost (`dim × 4 bytes` per vector) and index quality.  
- **L2‑norm / unit vectors**: many similarity metrics (e.g., cosine) assume vectors are normalized to unit length (`‖v‖₂ = 1`). Normalization also stabilizes ANN indexing.  

```python
import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    """L2‑normalize a batch of vectors."""
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norm, 1e-12, None)
```

### 2.3. Practical pipeline  

1. **Pre‑processing** – tokenization for text, resizing/cropping for images, resampling for audio.  
2. **Model inference** – run the chosen encoder (often on GPU) to obtain a dense vector.  
3. **Vector storage** – write the vector (and optional metadata) to a vector DB via its client SDK.  

A typical end‑to‑end flow in Python (using the open‑source **Milvus** client) looks like:

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
import sentence_transformers

# 1️⃣ Load model
model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

# 2️⃣ Encode documents
docs = [
    "Machine learning is ...",
    "Vector databases enable ..."
]
vectors = model.encode(docs, normalize_embeddings=True)  # already L2‑normed

# 3️⃣ Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1]),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields, "semantic text collection")
collection = Collection("semantic_text", schema)

# 4️⃣ Insert
collection.insert([vectors, docs])
```

---  

## 3. Fundamentals of Similarity Search  

### 3.1. Distance metrics  

| Metric          | Formula (vectors *a*, *b*)                     | Typical use |
|-----------------|-----------------------------------------------|-------------|
| Euclidean (L2)  | `‖a ‑ b‖₂`                                     | Raw distance; works with un‑normalized vectors |
| Cosine          | `1 ‑ (a·b) / (‖a‖‖b‖)`                         | Preferred after L2‑normalization |
| Manhattan (L1) | `∑|a_i ‑ b_i|`                                 | Sparse embeddings |
| Inner product   | `‑a·b` (maximization)                         | Equivalent to cosine for unit vectors; used by many ANN libraries |

### 3.2. Exact vs. Approximate NN  

- **Exact (brute‑force)** – scans every vector; guarantees true nearest neighbors but costs `O(N·dim)`. Viable for < 10⁵ vectors on modern CPUs/GPUs.  
- **Approximate NN (ANN)** – sacrifices a small amount of recall for orders‑of‑magnitude speedup. Common algorithms: HNSW, IVF‑PQ, ScaNN, DiskANN.  

### 3.3. Query workflow  

1. **Vectorize** the user query (same model, same preprocessing).  
2. **Compute distances** between the query vector and indexed vectors (exact or ANN).  
3. **Rank** results, apply any **metadata filters**, and return the top‑*k*.  

---  

## 4. Vector Database Architectures  

### 4.1. Core components  

| Layer          | Responsibility |
|----------------|-----------------|
| **Storage**    | Persists raw vectors, metadata, and index structures (often column‑oriented for compression). |
| **Indexing**   | Builds and maintains ANN structures (e.g., HNSW graph, IVF lists). |
| **Query engine** | Orchestrates vectorization, distance computation, filtering, and result aggregation. |

### 4.2. Popular indexing structures  

#### 4.2.1. Flat (brute‑force)  
No extra structure; simply scans all vectors. Best for small datasets or when exact recall is mandatory.  

#### 4.2.2. KD‑Tree / Ball Tree (low‑dimensional)  
Partitions space recursively; efficient up to ≈ 30 dimensions. Degrades sharply in high‑dimensional regimes (the curse of dimensionality).  

#### 4.2.3. HNSW (Hierarchical Navigable Small World)  
Builds a multi‑layer graph where higher layers contain a subset of nodes.  
- **Search complexity** ≈ `O(log N)` with high recall (> 0.95) and sub‑millisecond latency on million‑scale data.  

#### 4.2.4. IVF‑PQ / OPQ (product quantization)  
- **IVF** (Inverted File) clusters vectors (k‑means) → coarse search over centroids.  
- **PQ** compresses residuals into short codes (e.g., 8 bytes per vector).  
Enables billions of vectors on commodity RAM/SSD with a controllable recall‑vs‑speed trade‑off.  

### 4.3. Data‑management features  

- **Metadata filtering** – Boolean, range, or full‑text predicates on scalar fields (e.g., `category='news' AND timestamp>2024-01-01`).  
- **Hybrid queries** – combine vector similarity with scalar filters in a single request.  
- **CRUD** – upserts, deletes, and batch imports; most DBs support soft‑delete to keep index consistency.  

### 4.4. Scaling considerations  

| Aspect                | Strategies |
|-----------------------|------------|
| **Sharding**          | Horizontal partitioning by vector hash or logical tenant; each shard hosts its own index. |
| **Replication**       | Leader‑follower or quorum‑based replication for HA and read scaling. |
| **GPU vs. CPU**       | GPU accelerates both embedding inference and batch distance kernels. CPU‑only deployments rely on SIMD (AVX‑512) and multi‑threading. |
| **Hybrid storage**    | Keep raw vectors on SSD, compressed PQ codes in RAM; reduces memory footprint while preserving latency. |

---  

## 5. Real‑World Use Cases & Best Practices  

### 5.1. Semantic text search  

- **Scenario**: Retrieve relevant knowledge‑base articles for a user question.  
- **Pattern**: Encode both questions and documents with a Sentence‑Transformer; store document vectors; at query time, embed the question and perform ANN search.  

### 5.2. Visual similarity  

- **Scenario**: Detect duplicate product images or recommend visually similar items.  
- **Pattern**: Use CLIP or a ResNet‑based encoder; optionally combine with metadata (category, price) in a hybrid query.  

### 5.3. Audio & speech  

- **Scenario**: Speaker verification or music recommendation.  
- **Pattern**: Encode short audio snippets with wav2vec 2.0; store embeddings; query with a live recording to find matching speakers or tracks.  

### 5.4. Operational tips  

| Tip | Why it matters |
|-----|-----------------|
| **Choose the right dimensionality** | Higher dimensions improve expressiveness but increase storage (`4 bytes × dim`) and indexing time. Empirically, 256 – 512 works well for most text and image tasks. |
| **Tune the index** | For HNSW, adjust `M` (connectivity) and `efConstruction` (build time vs. recall). For IVF‑PQ, pick `nlist` (cluster count) and `nprobe` (search breadth). |
| **Monitor recall vs. latency** | Periodically compute *Recall@k* on a held‑out query set; set alerts if latency exceeds your SLA. |
| **Batch queries** | Vector DBs can compute distances for many queries in a single call, leveraging SIMD/GPU for higher throughput. |
| **Version embeddings** | When you upgrade the encoder, keep the old vectors for backward compatibility or re‑encode offline and swap indices atomically. |

---  

## 6. Conclusion  

- **Recap** – Vector databases turn high‑dimensional embeddings into searchable entities by coupling efficient ANN indexes with rich metadata handling. This enables real‑time semantic retrieval across text, images, and audio.  

- **Emerging trends**  
  - **Multimodal vectors** – joint embeddings that capture cross‑modal relationships (e.g., CLIP‑style text‑image).  
  - **AI‑native stores** – databases that expose model inference as first‑class operations, reducing pipeline friction.  
  - **Retrieval‑augmented generation (RAG)** – coupling vector search with large language models to ground generation in factual data.  

- **Call to action** – Spin up an open‑source vector DB (Milvus, Qdrant, or Weaviate), ingest a modest dataset, benchmark *latency* vs. *recall* across index types, and iterate on dimensionality and hyper‑parameters. The insights you gain will translate directly into faster, more accurate semantic applications.  