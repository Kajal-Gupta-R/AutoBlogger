# Vector Databases and How Similarity Search Works  

---  

## 1. Introduction  

- **Why the shift?** Traditional keyword indexes work well on structured text but break down with unstructured modalities—free‑form prose, photographs, audio clips, or video frames. Users now expect *semantic* retrieval: “find pictures that look like this” or “retrieve documents discussing the same concept”.  
- **Vector embeddings** are dense numeric representations (typically 64–1,024 dimensions) that capture the semantics of raw data. The core problem becomes **similarity search**: given a query vector **q**, locate the database vectors **v₁ … vₙ** with the smallest distance according to a chosen metric.  
- **What you’ll learn**  
  1. How raw data is turned into vectors.  
  2. The storage and indexing building blocks of a vector DB.  
  3. Exact vs. approximate nearest‑neighbor (ANN) algorithms and the end‑to‑end query flow.  
  4. Practical deployment tips and real‑world use cases.  

---  

## 2. From Raw Data to Vectors  

### 2.1 Embedding models  

| Modality | Popular encoders | Typical output dim |
|----------|------------------|--------------------|
| Text     | word2vec, GloVe, BERT, Sentence‑Transformers | 128‑1,024 |
| Image    | CLIP (ViT‑B/32), ResNet‑50, EfficientNet | 256‑1,024 |
| Audio    | YAMNet, wav2vec 2.0, OpenL3 | 128‑512 |
| Multimodal| CLIP (joint), FLAVA | 512‑1,024 |

These models are usually frozen; you feed the raw artifact and receive a deterministic embedding vector.

### 2.2 Dimensionality, normalization, and metric choice  

| Metric | Formula | When to use |
|--------|---------|-------------|
| **L2 (Euclidean)** | \(\|q - v\|_2\) | Raw embeddings, distance‑sensitive tasks |
| **Cosine** | \(1 - \frac{q \cdot v}{\|q\|\|v\|}\) | Direction matters more than magnitude; often applied after L2‑normalizing vectors |
| **Inner product** | \(-q \cdot v\) (treated as a distance) | Equivalent to cosine when vectors are normalized; useful for maximum‑similarity search |

**Normalization tip:** `v = v / np.linalg.norm(v)` makes cosine and inner‑product searches interchangeable and improves ANN index quality.

### 2.3 Pre‑processing pipelines and batch indexing  

1. **Batch ingestion** – Collect raw items and run them through the encoder in parallel (GPU/CPU workers).  
2. **Vector post‑processing** – Normalize; optionally apply dimensionality reduction (PCA, OPQ) to speed up indexing.  
3. **Bulk write** – Most vector DBs expose an `insert_many` API that writes vectors and associated scalar metadata in a single transaction.  

---  

## 3. Core Components of a Vector Database  

### 3.1 Storage layer  

| Representation | Pros | Cons |
|----------------|------|------|
| **Dense float32** | Simple; no loss of precision | 4 B × dim per vector → large memory footprint |
| **Dense float16 / bfloat16** | 2× compression; GPU‑friendly | Slight recall drop for very high‑dim data |
| **Sparse (e.g., CSR)** | Ideal for one‑hot or bag‑of‑words | Inefficient for typical dense embeddings |
| **Compressed (PQ, OPQ)** | 10‑100× size reduction | Requires decoding during distance computation |

### 3.2 Indexing structures  

| Structure | Typical use case | Strengths | Weaknesses |
|-----------|------------------|-----------|------------|
| **Flat (brute‑force)** | Small collections (< 10 k vectors) | Exact results; trivial to implement | O(N·dim) cost; not scalable |
| **Tree‑based** (KD‑Tree, Ball‑Tree) | Low‑dimensional data (≤ 20 d) | Logarithmic query time in low dimensions | Curse of dimensionality; degrades sharply beyond ~30 d |
| **Hash‑based** (LSH, Product Quantization) | Medium‑size, moderate‑dim data | Sub‑linear query time; modest memory | Approximate; recall depends on hash parameters |
| **Graph‑based** (HNSW, NSG) | High‑dimensional data (100‑1,000 d) | Near‑exact recall with logarithmic query complexity; robust to dimensionality | More expensive to build; higher memory for adjacency lists |

### 3.3 Metadata coupling & hybrid queries  

Vector DBs store **scalar fields** (e.g., timestamps, categories, user IDs) alongside each embedding. A hybrid query can filter on both vector similarity and scalar predicates:

```sql
SELECT *
FROM vectors
WHERE cosine_distance(query_vec, embedding) < 0.2
  AND category = 'product'
  AND created_at > '2024-01-01';
```

The engine first applies the scalar filter (fast B‑tree lookup) and then runs the ANN search on the reduced candidate set.

---  

## 4. Similarity Search Algorithms & Query Flow  

### 4.1 Exact NN vs. Approximate NN (ANN)  

| Aspect | Exact NN | ANN |
|--------|----------|-----|
| Guarantee | Returns true nearest neighbor(s) | Returns *likely* nearest neighbor(s) with configurable recall |
| Complexity | O(N) for flat scans; O(log N) for low‑dim trees | Sub‑linear (often O(log N) or O(1) per query) |
| Typical use | Small datasets, evaluation, precision‑critical workloads | Production‑scale, latency‑sensitive services |

### 4.2 Query processing pipeline  

1. **Vectorization** – Encode the query payload (text → BERT, image → CLIP).  
2. **Candidate generation** – Use the chosen index to retrieve *k* ≈ 100‑1,000 approximate neighbors.  
3. **Re‑ranking** – Compute the exact distance on the candidate set (often with higher‑precision arithmetic).  
4. **Filtering** – Apply scalar predicates, pagination, or custom business rules.  

### 4.3 Distance‑calculation optimizations  

- **Product Quantization (PQ)** – Split each vector into sub‑vectors, quantize each with a codebook, and pre‑compute distance tables.  
- **Inverted File (IVF)** – Cluster vectors (e.g., k‑means) and search only within the nearest clusters.  
- **Cache‑aware tricks** – Keep centroids and PQ tables in CPU cache; align data structures to 64‑byte boundaries to avoid false sharing.  

### 4.4 Handling dynamic data  

| Operation | Impact on index | Typical strategy |
|-----------|-----------------|------------------|
| **Insert** | May require adding a node (graph) or updating IVF posting lists | Batch inserts → periodic background rebuild; most DBs support online HNSW insertion |
| **Delete** | Invalidation of graph edges or posting‑list entries | Lazy deletion + tombstones; compactation jobs run nightly |
| **Update** | Equivalent to delete + insert | Treat as two separate operations; keep versioning for auditability |

---  

## 5. Practical Considerations & Deployments  

### 5.1 Scaling strategies  

- **Sharding** – Partition vectors by a hash of a primary key or by semantic region (e.g., per‑tenant). Each shard runs its own index; a query router merges results.  
- **Replication** – Read‑only replicas reduce latency for hot queries; a write‑ahead log (WAL) guarantees durability.  
- **Multi‑tenant isolation** – Namespace‑level quotas, per‑tenant resource pools, and row‑level security on scalar metadata.  

### 5.2 Consistency, durability, and latency trade‑offs  

| Consistency level | Write path | Read path |
|-------------------|------------|-----------|
| **Strong** (synchronous replication) | Higher write latency | Fresh data guaranteed |
| **Eventual** (asynchronous replication) | Low write latency | Possible staleness (acceptable for many recommendation workloads) |

Choosing **disk‑backed storage** (e.g., RocksDB) adds durability at the cost of higher query latency; many vector DBs keep the index in memory and periodically flush to SSD for crash recovery.

### 5.3 Monitoring & observability  

- **Latency histograms** – Bucketed per operation (vectorize, candidate‑gen, rerank).  
- **Recall metrics** – Run a background job that compares ANN results against a ground‑truth flat scan on a sample; expose `recall@k`.  
- **Resource usage** – Track RAM consumption of the graph, CPU utilization of PQ tables, and I/O rates of the WAL.  

### 5.4 Integration patterns  

| Pattern | Typical stack | Example |
|---------|---------------|---------|
| **REST / GraphQL** | FastAPI, Express, Go Gin | `POST /search` with JSON `{ "text": "...", "k": 10 }` |
| **LangChain** | Python, LLM orchestrators | `retriever = MilvusRetriever(collection_name="docs")` |
| **Retrieval‑Augmented Generation (RAG)** | LLM + vector store | Query vector → top‑k docs → prompt LLM with retrieved snippets |

---  

## 6. Real‑World Use Cases  

1. **Semantic text search** – Customer‑support portals that answer queries by retrieving the most relevant knowledge‑base articles (RAG pipelines).  
2. **Visual similarity** – E‑commerce platforms that recommend visually similar products or detect duplicate images during ingestion.  
3. **Audio/video fingerprinting** – Content‑ID systems that match short audio clips against a catalog of millions of tracks using embeddings from wav2vec.  
4. **Anomaly detection** – High‑dimensional sensor streams (e.g., IoT) where outliers are identified by low similarity to the bulk of historical embeddings.  

---  

## 7. Conclusion  

### Key takeaways  
- Vector embeddings turn unstructured data into a uniform numeric space.  
- Vector databases combine dense storage, sophisticated ANN indexes (HNSW, IVF‑PQ, etc.), and scalar metadata to deliver low‑latency semantic search.  
- Exact and approximate search differ mainly in recall vs. performance; most production systems settle on ANN with a configurable recall target.  

### Future outlook  
- **Hardware acceleration** – GPUs and TPUs are being integrated directly into index traversal (e.g., GPU‑HNSW).  
- **Unified multimodal stores** – Emerging standards (VectorDB API, Vectordb‑SQL) aim to treat text, image, and audio vectors uniformly.  
- **On‑device indexes** – Edge‑optimized graph structures enable offline similarity search for mobile AR apps.  

### Call to action  

1. **Clone an open‑source vector DB** (e.g., Milvus or Qdrant).  
2. **Load a benchmark dataset** (MS‑MARCO for text, DeepFashion for images).  
3. **Measure latency and recall** for flat vs. HNSW indexes.  

```python
# Minimal Milvus example (Python SDK)
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections
import numpy as np

connections.connect("default", host="localhost", port="19530")

# 1. Define schema
id_field = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True)
vec_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
meta_field = FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64)

schema = CollectionSchema(fields=[id_field, vec_field, meta_field],
                         description="demo collection")
collection = Collection(name="demo_text", schema=schema)

# 2. Insert data (replace with real BERT embeddings)
embs = np.random.rand(5_000, 768).astype(np.float32)
cats = ["news"] * 5_000
collection.insert([embs.tolist(), cats])

# 3. Create HNSW index
collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
)

# 4. Query
query_vec = np.random.rand(1, 768).astype(np.float32)
search_params = {"metric_type": "IP", "params": {"ef": 64}}

results = collection.search(
    data=query_vec,
    anns_field="embedding",
    param=search_params,
    limit=5,
    expr="category == 'news'"
)

print(results)
```

Experiment with the index parameters, observe how recall and latency move together, and you’ll quickly internalize how similarity search works under the hood.  

---  

*Happy vector searching!*