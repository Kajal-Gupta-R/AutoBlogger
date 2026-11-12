# Embeddings: Turning Text into Numbers That Capture Meaning  

## Introduction  

- **Why numerical representations are essential for NLP**  
  Machine‑learning models operate on tensors. Converting words into numbers lets algorithms learn patterns, compute gradients, and scale to massive corpora.  

- **A brief history of word representations**  
  *One‑hot* vectors (high‑dimensional and sparse) were the first attempt, but they ignore similarity between words. The field quickly moved to *dense* vectors—continuous embeddings—that encode semantic relationships in a low‑dimensional space.  

- **What you’ll learn**  
  1. Core concepts behind embeddings.  
  2. Classic static methods (Word2Vec, GloVe, FastText).  
  3. Modern contextual embeddings from transformer models.  
  4. Real‑world use cases.  
  5. A quick hands‑on guide to load and use embeddings.  

---

## 1. Foundations of Text Embeddings  

### What is an embedding?  

An embedding is a mapping  

\[
f:\text{token}\;\rightarrow\;\mathbb{R}^{d}
\]

that places linguistic units (words, subwords, sentences) in a continuous vector space where geometric proximity reflects semantic similarity.  

### Core concepts  

| Concept | Description |
|---------|-------------|
| **Vector space** | All embeddings share a common coordinate system; arithmetic can reveal linguistic regularities (e.g., *king* − *man* + *woman* ≈ *queen*). |
| **Dimensionality** (\(d\)) | Typical values range from 50 to 1 024. Higher dimensions capture finer nuance but increase memory and compute costs. |
| **Similarity metric** | Cosine similarity \(\displaystyle \frac{u\cdot v}{\|u\|\|v\|}\) is the de‑facto standard; Euclidean distance is sometimes used for clustering. |

### Static embeddings vs. bag‑of‑words  

Bag‑of‑words (BOW) treats each token as an independent feature, discarding order and any notion of similarity. Embeddings provide a **dense** representation that captures both syntactic and semantic context, enabling downstream models to generalize beyond exact token matches.  

---

## 2. Classical Embedding Techniques  

### Word2Vec (CBOW & Skip‑gram)  

- **Objective**: Predict a target word from its surrounding context (CBOW) or predict surrounding words from a target (Skip‑gram).  
- **Strengths**: Fast, scalable, captures linear relationships.  
- **Limitations**: *Static*—the same vector is used for every occurrence of a word; no subword awareness.  

### GloVe (Global Vectors)  

- **Objective**: Factorize a global word‑co‑occurrence matrix; the loss minimizes the difference between the dot product of two vectors and the log of their co‑occurrence count.  
- **Strengths**: Leverages corpus‑wide statistics; often yields higher‑quality vectors on smaller datasets.  
- **Limitations**: Still static; building the co‑occurrence matrix can be memory‑intensive.  

### FastText  

- **Objective**: Extends Word2Vec by representing each word as a bag of character n‑grams.  
- **Strengths**: Handles out‑of‑vocabulary (OOV) words gracefully; captures morphological information.  
- **Limitations**: Larger model size; embeddings remain context‑independent.  

### Quick comparison  

| Method   | Training data | Static / Contextual | OOV handling | Typical dimensionality |
|----------|---------------|---------------------|--------------|------------------------|
| Word2Vec | Local windows | Static              | No           | 100 – 300 |
| GloVe    | Global counts | Static              | No           | 100 – 300 |
| FastText | Local windows + n‑grams | Static | Yes (subwords) | 100 – 300 |

---

## 3. Contextualized Embeddings and Transformers  

### From static to contextual  

Static vectors assign one representation per token. Contextual models generate a **different** vector for each token occurrence, conditioned on its surrounding words.  

### Transformer architecture (self‑attention)  

Transformers replace recurrence with self‑attention layers. For every token, attention weights compute a weighted sum of **all** other token representations, allowing the model to capture long‑range dependencies in parallel.  

### Key models  

| Model | Architecture | Training objective | Notable traits |
|-------|--------------|--------------------|----------------|
| **BERT** | Encoder‑only | Masked‑language modeling + next‑sentence prediction | Bidirectional, strong for understanding tasks |
| **RoBERTa** | Encoder‑only | Masked‑language modeling (larger batch, longer training) | No next‑sentence loss, generally outperforms BERT |
| **GPT series** | Decoder‑only | Autoregressive language modeling | Generates text; provides contextual embeddings at each generation step |
| **T5** | Encoder‑decoder | Text‑to‑text (span‑corruption) | Casts every NLP task as seq‑2‑seq; encoder yields versatile embeddings |

### Generating token‑level embeddings  

1. **Tokenization** → token IDs.  
2. **Embedding lookup** → initial token vectors.  
3. **Transformer layers** → repeated self‑attention and feed‑forward operations.  
4. **Final hidden state** of a token (often the last layer) is its contextual embedding.  

---

## 4. Practical Applications  

### Text classification, sentiment analysis, and topic modeling  

Replace sparse TF‑IDF vectors with sentence embeddings and feed them into a lightweight classifier (logistic regression, MLP). The result is higher accuracy with far less feature engineering.  

### Information retrieval and semantic search  

Encode documents and queries into the same embedding space; rank results by cosine similarity. This enables retrieval based on *meaning* rather than exact keyword overlap.  

### Question answering and conversational agents  

Contextual embeddings let models locate answer spans in a passage (e.g., BERT‑based QA) or generate coherent replies (GPT‑style chatbots).  

### Transfer learning & fine‑tuning  

Pre‑trained models act as universal feature extractors. Fine‑tuning a few transformer layers on a target dataset adapts the embeddings to domain‑specific nuances while preserving general language knowledge.  

---

## 5. Getting Started with Embeddings (Hands‑On Guide)  

### Selecting the right pre‑trained model  

| Criterion | Recommendation |
|-----------|----------------|
| **Speed vs. accuracy** | `distilbert-base-uncased` for rapid prototyping; `roberta-large` for state‑of‑the‑art performance. |
| **Domain** | `bioBERT` (biomedical), `FinBERT` (finance), `sentence‑transformers/all-MiniLM-L6-v2` (generic sentence similarity). |
| **License** | Prefer Apache‑2.0 or MIT for commercial use; verify any restrictive terms before deployment. |

### Popular libraries  

- **🤗 Hugging Face Transformers** – core API for loading any transformer model.  
- **Sentence‑Transformers** – simplifies sentence‑level encoding and provides ready‑made similarity models.  
- **Gensim** – still useful for static embeddings (Word2Vec, GloVe, FastText).  

### Code snippet: encode sentences and compute similarity  

```python
# Install once:
# pip install sentence-transformers torch

from sentence_transformers import SentenceTransformer, util

# 1️⃣ Load a pre‑trained sentence encoder
model = SentenceTransformer('all-MiniLM-L6-v2')   # 384‑dim vectors, fast & lightweight

# 2️⃣ Encode a batch of sentences
sentences = [
    "Artificial intelligence is transforming industry.",
    "Machine learning enables computers to learn from data.",
    "The weather today is sunny and warm."
]
embeddings = model.encode(
    sentences,
    normalize_embeddings=True,   # L2‑norm → cosine = dot product
    batch_size=32,
    show_progress_bar=False
)

# 3️⃣ Compute pairwise cosine similarity
similarity = util.cos_sim(embeddings, embeddings)
print(similarity.round(3))
```

**Sample output**

```
tensor([[1.000, 0.842, 0.112],
        [0.842, 1.000, 0.098],
        [0.112, 0.098, 1.000]])
```

The first two sentences are semantically close (both discuss AI/ML); the third is unrelated.  

### Tips for large corpora and performance  

1. **Batch encoding** – process texts in chunks (`batch_size=64` or higher) to keep GPU memory in check.  
2. **FAISS index** – for millions of vectors, build a FAISS IVF or HNSW index to achieve sub‑second nearest‑neighbor search.  
3. **Quantization** – use `torch.quantization` or `model.half()` to halve memory with minimal accuracy loss.  
4. **Caching** – store pre‑computed embeddings on disk (e.g., Parquet or Feather) to avoid repeated inference.  

---

## Conclusion  

- **Recap** – Embeddings translate raw text into a numeric space where distance reflects meaning, bridging language and machine learning. From static Word2Vec vectors to transformer‑based contextual representations, each generation has expanded the expressiveness and applicability of NLP models.  

- **Emerging trends**  
  - *Multimodal embeddings*: Jointly encode text with images, audio, or video (e.g., CLIP, Flamingo).  
  - *Retrieval‑augmented generation*: Combine dense retrieval with large language models to ground generation in external knowledge bases.  

- **Final thoughts** – Start with the simplest solution that satisfies your accuracy and latency constraints. A lightweight sentence‑transformer is ideal for rapid prototyping; move to a larger, domain‑specific model and fine‑tune when performance plateaus. Systematic experimentation—varying dimensionality, pooling strategies, and indexing methods—remains the most reliable path to effective embeddings in real projects.