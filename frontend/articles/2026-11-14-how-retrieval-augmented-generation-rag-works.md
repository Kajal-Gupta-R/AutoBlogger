# Retrieval‑Augmented Generation (RAG)

## Introduction  

Large language models (LLMs) excel at fluent text generation, but they have three well‑known limitations:

| Limitation | Consequence |
|------------|--------------|
| **Fixed knowledge cutoff** | Facts become stale as the world changes. |
| **Hallucinations** | The model may fabricate details for niche or recent topics. |
| **Quadratic compute growth** | Expanding the context window quickly becomes expensive. |

**RAG** addresses these issues by coupling the parametric knowledge of an LLM with up‑to‑date external documents. The result is a system that produces grounded answers, reduces hallucinations, and keeps prompts short.

```
User query ──► Retriever ──► Top‑k passages ──► Prompt builder ──► Generator ──► Answer
```

**Who should read this?**  
- ML engineers building LLM‑powered services.  
- Data scientists who need reproducible, current QA.  
- Product teams designing chatbots, search assistants, or research tools.  

---

## 1. Retrieval vs. Generation  

### Retrieval  

| Concept | Description |
|---------|-------------|
| **Document** | Any chunk of text (FAQ, article, code snippet). |
| **Embedding** | Dense vector representation of a document or query. |
| **Similarity search** | Nearest‑neighbor lookup in the embedding space. |

**Retriever families**  

| Type | Typical algorithm | Strengths |
|------|-------------------|-----------|
| **Sparse** | BM25 (inverted index) | Fast, exact lexical matching. |
| **Dense** | Bi‑encoders (DPR, Sentence‑Transformers), ColBERT | Captures semantic similarity. |

**Index structures**  

- **Inverted index** – used by BM25.  
- **FAISS IVF‑PQ**, **HNSW** – popular for dense vectors.  

### Generation  

LLMs are autoregressive models that predict the next token given all previous ones. They come in two flavors:

| Architecture | Example | Conditioning |
|--------------|---------|--------------|
| **Decoder‑only** | GPT‑3, LLaMA | Prompt‑only or prompt + retrieved context. |
| **Encoder‑decoder** | T5, BART | Same conditioning options. |

### Why combine them?  

Retrieval provides factual grounding; generation provides fluent articulation. The combination reduces reliance on memorised facts, cuts token usage, and improves answer relevance.

---

## 2. RAG Architecture  

### Component overview  

1. **Retriever** – encodes the query and returns the top‑k passages.  
2. **Prompt builder** – merges the query and passages into a single prompt.  
3. **Generator** – the LLM consumes the prompt and produces the answer.  

### Retriever details  

| Step | Options | Notes |
|------|---------|-------|
| **Query encoding** | *Single‑tower* (same encoder for query & doc, e.g., SBERT) <br> *Dual‑tower* (separate encoders, trained with contrastive loss, e.g., DPR) | Dual‑tower usually yields higher recall after fine‑tuning. |
| **Top‑k selection** | Approximate nearest‑neighbor (ANN) search → retrieve *k* (5‑20). <br> Optional cross‑encoder re‑ranking for precision. | Re‑ranking adds latency but can boost `Recall@k`. |

### Generator / Reader  

**Prompt construction** (most common pattern)

```
Answer the question using ONLY the excerpts below.

Question: <query>

Context:
[1] <passage 1>
[2] <passage 2>
...
Answer:
```

**Fusion strategies**

| Strategy | How it works |
|----------|--------------|
| **Early‑fusion** | Concatenate all passages before feeding them to the LLM. |
| **Late‑fusion** | Generate an answer per passage, then merge the partial outputs. |
| **Knowledge‑aware attention** | Inject passage embeddings directly into the LLM’s attention layers (e.g., RAG‑token). |

### Training paradigms  

| Paradigm | Description |
|----------|-------------|
| **RAG‑token** | The LLM attends to retrieved passages token‑by‑token; the whole pipeline is fine‑tuned end‑to‑end with cross‑entropy. |
| **RAG‑sequence** | The generator sees a single concatenated context; fine‑tuning focuses on answer sequences. |
| **Supervised** | QA pairs with ground‑truth passages train both the retriever (contrastive loss) and the generator (generation loss). |
| **Self‑supervised** | Tasks such as Inverse Cloze (mask a passage and retrieve it) combined with language modeling. |

### Inference flow (step‑by‑step)

1. Encode the user query → dense vector.  
2. ANN search → retrieve *k* passages + similarity scores.  
3. (Optional) re‑rank with a cross‑encoder.  
4. Build the prompt using a template that explicitly requests grounding.  
5. Feed the prompt to the LLM (with or without extra passage embeddings).  
6. Post‑process the output (truncate, verify, log).  

---

## 3. Data Preparation & Indexing  

1. **Corpus selection** – internal knowledge bases, public FAQs, code repositories, web crawls.  
2. **Pre‑processing**  
   - Strip HTML and boiler‑plate.  
   - Chunk into passages of 200‑400 tokens with a 50‑token overlap to preserve context.  
   - Attach metadata (source ID, timestamps, tags).  
3. **Embedding generation**  
   - Choose an encoder (e.g., `sentence‑transformers/all‑mpnet‑base‑v2`).  
   - Batch size should respect GPU memory; store embeddings as `float16` to reduce footprint.  
4. **Vector store construction**  
   - **FAISS** (IVF‑PQ) for on‑premises, **Milvus** or **Pinecone** for managed cloud.  
   - Keep the index separate from metadata, which can live in PostgreSQL, Elastic, etc.  
5. **Dynamic data handling**  
   - *Incremental indexing*: embed new documents and add them via `add_vectors`.  
   - *Deletion*: mark stale IDs and periodically rebuild the affected segment.  
   - *Freshness boosting*: store `last_updated` and increase the score of recent passages at retrieval time.  

---

## 4. Prompt Engineering & Context Integration  

### Retrieval‑aware prompt template (Python)

```python
def build_prompt(query: str, passages: list[str]) -> str:
    ctx = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        "Answer the question using ONLY the provided excerpts.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{ctx}\n\n"
        "Answer:"
    )
```

### Managing token limits  

| Technique | When to use |
|-----------|-------------|
| **Truncation** | Keep highest‑scoring passages until `max_context_tokens` is reached. |
| **Summarization** | Run a lightweight summarizer on low‑score passages before concatenation. |
| **Hierarchical retrieval** | First retrieve coarse documents, then sub‑passages within the top document. |

### Reducing hallucinations  

1. Explicit grounding cue (`using ONLY the provided excerpts`).  
2. Post‑generation verification: run a cross‑encoder to check whether the answer is supported by any retrieved passage.  

### Prompt examples by use‑case  

| Use‑case | Template |
|----------|----------|
| **Q&A** | See the Python function above (includes “ONLY” clause). |
| **Summarization** | `"Summarize the following articles in three bullet points:\n{ctx}"` |
| **Code generation** | `"Write a Python function that implements the algorithm described below:\n{ctx}"` |

---

## 5. Evaluation, Scaling, and Production  

### Metrics  

| Layer | Metric | Typical target |
|-------|--------|----------------|
| **Retrieval** | `Recall@k`, `MRR`, `NDCG` | `Recall@5` ≥ 90 % for a well‑curated corpus |
| **Generation** | `BLEU`, `ROUGE‑L`, **Faithfulness** (e.g., FEVER score) | Faithfulness ≥ 0.8 |
| **End‑to‑end** | `RAG‑EM` (retrieval × generation accuracy) | > 0.75 in most QA benchmarks |

### Scaling guidelines  

| Aspect | Recommendation |
|--------|----------------|
| **Hardware** | Retrieval on CPU (FAISS IVF‑PQ); generation on GPU (mixed‑precision). |
| **Batch inference** | Group queries → shared ANN search → parallel LLM calls. |
| **Caching** | Cache top‑k results per query fingerprint; TTL based on data freshness. |
| **Latency budget** | ≤ 500 ms for retrieval + ≤ 800 ms for generation in interactive settings. |
| **Cost control** | Limit *k* and use a smaller LLM for fallback when retrieval confidence is low. |

### Production concerns  

- **Monitoring**: log retrieval recall, hallucination rate, per‑stage latency.  
- **Fallback**: if no passages are returned, switch to pure LLM mode with a disclaimer.  
- **Observability**: store `query → passages → answer` triplets for audits and debugging.  

### Real‑world case studies  

| Project | Corpus | Retrieval recall | End‑to‑end accuracy | Business impact |
|---------|--------|------------------|---------------------|-----------------|
| Customer‑support bot | Ticket archive (≈ 2 M docs) | 92 % `Recall@5` | 84 % exact‑match QA | 27 % reduction in escalations |
| Enterprise knowledge base | Internal wiki (≈ 500 k pages) | 90 % `Recall@5` | 80 % F1 | Faster issue resolution, lower support cost |
| Research assistant | arXiv PDFs (2020‑2024) | 95 % `Recall@5` | 90 % citation‑aligned answers | Accelerated literature review |

---

## Conclusion  

RAG unites the *static* knowledge of LLMs with *dynamic* information from external documents, delivering answers that are both fluent and factually grounded. Emerging trends include multimodal retrieval (image + text), tool‑use integration (function calling), and self‑retrieval where the model learns to query its own memory.

### Practical checklist  

1. **Corpus** – clean, chunked, and enriched with metadata.  
2. **Retriever** – start with a dense bi‑encoder; add cross‑encoder re‑ranking if latency permits.  
3. **Prompt** – make grounding explicit; stay within token limits.  
4. **Evaluation** – measure both retrieval (`Recall@k`) and generation (faithfulness, BLEU/ROUGE).  
5. **Deployment** – use caching, monitor key metrics, and provide a graceful fallback path.  

### Further reading  

- **Papers**:  
  - Lewis *et al.* “Retrieval‑Augmented Generation for Knowledge‑Intensive NLP Tasks” (2020).  
  - Izacard & Grave “RAG‑Token” (2022).  
- **Libraries**: `langchain`, `llama‑index`, `faiss`, `milvus`, `pinecone-client`.  
- **Tutorials**: Hugging Face “RAG tutorial”, LlamaIndex “Getting Started with Retrieval‑Augmented Generation”.  