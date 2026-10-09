# Vector Databases and How Similarity Search Works  

## Introduction  

- **Why similarity search matters** – It powers recommendation engines, image/audio retrieval, semantic search, and anomaly detection.  
- **Limits of relational databases** – Traditional indexes are built for equality or range predicates on low‑dimensional columns. High‑dimensional vectors explode index size and make distance calculations impractical.  
- **What you’ll learn** – How raw data becomes embeddings, the math behind similarity, the main index families, scaling patterns, and production‑ready best practices.  

---  

## 1. Vector Representations: From Raw Data to Embeddings  

### Definition  
A *vector* (or *embedding*) is a fixed‑length array of floating‑point numbers that captures the semantic information of an object in a continuous space.  

### Common Sources  

| Modality | Typical Model(s)                | Output dimension |
|----------|--------------------------------|-------------------|
| Text     | BERT, GPT, Sentence‑Transformers | 256 – 1 024 |
| Images   | CLIP, ResNet, ViT               | 512 – 1 024 |
| Audio    | Wav2Vec, YAMNet                 | 128 – 512 |
| Graphs   | GraphSAGE, Node2Vec             | 64 – 256 |

### Dimensionality & Pre‑processing  

| Step | Reason | Typical practice |
|------|--------|------------------|
| **Normalization** | Makes cosine similarity equal to inner‑product after L2‑norm | `vector = vector / np.linalg.norm(vector)` |
| **Quantization** | Reduces storage and I/O cost | 8‑bit Product Quantization (PQ) or OPQ |
| **Batch encoding** | Keeps GPU memory usage stable | Encode in chunks (e.g., 1 000 vectors per batch) |

---  

## 2. Fundamentals of Similarity Search  

### Distance Metrics  

| Metric | Formula (vectors **a**, **b**) | Typical use |
|--------|--------------------------------|-------------|
| Euclidean (L2) | \\( \|a-b\|_2 = \sqrt{\sum_i (a_i-b_i)^2} \\) | General‑purpose, metric‑space indexes |
| Cosine | \\( 1 - \frac{a\cdot b}{\|a\|_2 \|b\|_2} \\) | Text & multimodal embeddings (after L2‑norm) |
| Manhattan (L1) | \\( \|a-b\|_1 = \sum_i |a_i-b_i| \\) | Sparse vectors |
| Inner product (IP) | \\( -a\cdot b \\) (minimised) | Max‑inner‑product search (MIPS) |

### Exact vs. Approximate Nearest Neighbour (ANN)  

| Approach | Guarantee | Complexity | When to use |
|----------|-----------|------------|-------------|
| **Exact** (brute‑force) | 100 % recall | \\(O(N \cdot d)\\) per query | Small datasets (< 10⁶ vectors) or when absolute correctness is required |
| **ANN** | High but not perfect recall | Sub‑linear (often < 1 ms for billions of vectors) | Production workloads where latency outweighs a few missed neighbors |

### Query Workflow  

1. **Encode** the query item → vector **q**.  
2. **Search** the index using the chosen metric.  
3. **Rank** the top‑k candidates and return their identifiers (and optionally distances).  

---  

## 3. Vector Database Architectures  

| Architecture | Core idea | Strengths | Weaknesses |
|--------------|-----------|-----------|------------|
| **Flat (brute‑force)** | Store raw vectors; linear scan on each query. | Simple; 100 % recall. | Latency grows linearly; impractical beyond ~10⁶ vectors. |
| **Partitioned (IVF, PQ, OPQ)** | Coarse quantizer creates inverted lists; optional product quantization compresses vectors. | Scales to billions; tunable recall‑latency trade‑off. | Requires training data; recall depends on `nprobe`. |
| **Graph‑based (HNSW, NSG)** | Constructs a small‑world proximity graph; greedy search follows neighbor links. | Very high recall at low latency; cheap dynamic updates. | Higher memory overhead; index build can be expensive. |
| **Hybrid / Cloud‑native** | Combines several structures, adds metadata filtering, autoscaling, and managed APIs. | Operational simplicity; SLA guarantees. | Vendor lock‑in; limited low‑level tuning. |

**Examples** – Pinecone (managed HNSW + IVF), Milvus (supports IVF, PQ, HNSW), Weaviate (graph + metadata).  

---  

## 4. Indexing, Querying, and Scaling  

### Building an Index (Milvus + HNSW example)

```python
# Milvus Python SDK – create a collection and HNSW index
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# 1️⃣ Define the schema
vectors = FieldSchema(name="embedding",
                      dtype=DataType.FLOAT_VECTOR,
                      dim=768)
ids     = FieldSchema(name="pk",
                      dtype=DataType.INT64,
                      is_primary=True,
                      auto_id=True)
schema = CollectionSchema(fields=[ids, vectors],
                          description="Image embeddings")

# 2️⃣ Create the collection
collection = Collection(name="imgs", schema=schema)

# 3️⃣ Insert data (list of vectors)
#   `vecs` is a list/np.ndarray of shape (N, 768)
collection.insert([list(range(len(vecs))), vecs])

# 4️⃣ Build an HNSW index
index_params = {
    "metric_type": "IP",          # inner product (max‑IP)
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="embedding", index_params=index_params)
```

| Step | Note |
|------|------|
| **Training** | IVF/PQ need a representative sample; HNSW does not. |
| **Parameter tuning** | `nlist`/`nprobe` for IVF; `M`, `efConstruction`, `ef` for HNSW. |
| **Re‑indexing** | Required when dimensionality changes or after massive data churn. |

### Query Execution Pipeline  

1. **Encode** the query → vector **q**.  
2. **Set search parameters** (`nprobe` for IVF, `ef` for HNSW).  
3. **Execute** the search → obtain candidate IDs + distances.  
4. **Post‑process** (optional) – re‑rank with a more expensive metric or apply business logic.  

#### Recall vs. latency  

| Parameter | Effect on recall | Effect on latency |
|-----------|------------------|-------------------|
| Increase `nprobe` / `ef` | ↑ recall | ↑ latency |
| Decrease `nprobe` / `ef` | ↓ recall | ↓ latency |

### Distributed Deployment Patterns  

| Pattern | Description | Ideal scenario |
|---------|-------------|----------------|
| **Sharding by vector ID** | Split the collection across nodes; each shard holds a disjoint subset of vectors. | Datasets > 10⁹ vectors. |
| **Replica sets** | Duplicate shards for read‑scaling and fault tolerance. | Low‑latency SLA, high QPS. |
| **Load balancer + query router** | Front‑end routes a query to relevant shards, merges partial results, and returns the final top‑k. | Multi‑region deployments, geo‑distribution. |

### Monitoring & Metrics  

- **Throughput & latency** – QPS, p99/p95 latency.  
- **Recall** – Sample queries compared against a brute‑force baseline.  
- **Resource usage** – Memory/disk (raw vectors + index overhead), CPU/GPU utilization (especially for on‑the‑fly encoding).  
- **Health alerts** – Trigger on latency spikes, recall drops after bulk updates, or storage saturation.  

---  

## 5. Real‑World Applications & Best Practices  

### Typical Use Cases  

| Domain | Example |
|--------|---------|
| **Semantic search** | Retrieve documents that match the *meaning* of a query. |
| **Recommendation** | Find nearest‑neighbor items or users in embedding space. |
| **Anomaly detection** | Flag vectors whose similarity to any “normal” vector falls below a threshold. |
| **Multimedia retrieval** | Image‑to‑image, audio‑to‑audio, or cross‑modal (text ↔ image) search. |

### Data Hygiene  

- **Embedding drift** – Retrain the upstream model periodically, then re‑encode and re‑index.  
- **Updates** – Prefer upserts when supported; otherwise perform delete‑then‑insert.  
- **TTL / expiration** – Schedule background jobs to purge stale vectors (e.g., session embeddings).  

### Security & Privacy  

| Concern | Mitigation |
|---------|------------|
| **At‑rest encryption** | Enable DB‑level encryption or encrypt vectors client‑side before ingestion. |
| **Access control** | Role‑based policies per collection; isolate tenant data in multi‑tenant SaaS. |
| **Differential privacy** | Add calibrated noise to embeddings when regulatory privacy guarantees are needed. |

### Choosing & Tuning a Vector DB  

| Decision factor | Recommendation |
|-----------------|----------------|
| **Dataset size** | < 10 M → flat or HNSW; ≥ 10 M → IVF‑PQ or a managed service. |
| **Latency SLA** | Sub‑10 ms → HNSW with high `ef`; 50‑100 ms → IVF with moderate `nprobe`. |
| **Update frequency** | Frequent upserts → HNSW; bulk loads → IVF. |
| **Budget** | Open‑source (Milvus, FAISS) on self‑hosted infra; managed (Pinecone, Weaviate Cloud) for ops‑free deployments. |

**Production tip** – Start with a modest IVF index (`nlist=1024`). Benchmark recall at `nprobe=10`. Then iteratively increase `nlist` and `nprobe` until latency meets your target.  

---  

## Conclusion  

- **Key takeaways** – Vectors turn unstructured data into searchable numeric representations; similarity is measured with distance metrics; vector databases provide specialised indexes (IVF, PQ, HNSW) that make ANN feasible at scale.  
- **When to adopt** – Use a vector DB whenever you need fast, high‑dimensional similarity queries that relational indexes cannot support, especially for semantic or multimodal workloads.  
- **Future trends** – Unified multimodal embeddings, on‑device (edge) vector search, and tighter integration of large language models (LLMs) with vector stores.  

**Call to action** – Spin up a free Milvus (or any open‑source vector DB) instance, ingest a small text corpus, and run a semantic‑search query. Experiment with index parameters to observe the recall‑vs‑latency trade‑off in practice. Happy searching!