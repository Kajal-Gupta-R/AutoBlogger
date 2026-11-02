## Embeddings – Turning Text into Numbers That Capture Meaning  

---  

### 1. Introduction  

- **Why raw text can’t be fed directly to most ML models**  
  Most machine‑learning algorithms expect numeric tensors. Plain strings have no inherent ordering, magnitude, or distance, so a model can’t compute gradients or similarity scores on them.  

- **The core idea of an *embedding***  
  An *embedding* is a dense, continuous‑valued vector \( \mathbf{v} \in \mathbb{R}^n \) that represents a token, phrase, sentence, or even an image. The vector space is constructed so that geometric relationships (e.g., cosine similarity) mirror linguistic relationships such as synonymy, analogy, or topical similarity.  

- **What you’ll learn**  
  1. A concise history from one‑hot to contextual vectors.  
  2. The mathematics that lets embeddings capture meaning.  
  3. The main families of embeddings and when to choose each.  
  4. A hands‑on pipeline that turns raw text into ready‑to‑use vectors.  
  5. Common pitfalls and best‑practice tips for production.  

---  

### 2. From Words to Vectors: A Brief History  

| Era | Technique | Key Idea |
|-----|-----------|----------|
| **Early approaches** | One‑hot, Bag‑of‑Words, TF‑IDF | Sparse, high‑dimensional vectors; similarity based on raw counts. |
| **Distributional hypothesis** | “You shall know a word by the company it keeps.” | Words that appear in similar contexts tend to have similar meanings. |
| **Neural breakthroughs** | **Word2Vec** (CBOW & Skip‑gram) | Predict surrounding words (or the target) with a shallow network → 100–300‑dim static vectors. |
| | **GloVe** (global co‑occurrence) | Factorises a word‑co‑occurrence matrix; balances local and global statistics. |
| **Contextual turn** | **ELMo**, **BERT**, **GPT**, etc. | Deep (bi)directional transformers produce a distinct vector for each token occurrence, jointly capturing syntax and semantics. |

The trajectory is clear: **bag‑of‑words → static distributional vectors → contextual vectors**.  

---  

### 3. How Embeddings Capture Meaning  

#### 3.1 Mathematical foundation  

- **Mapping**: a token *t* → **v**ₜ ∈ ℝⁿ.  
- **Similarity** (most downstream tasks):  

\[
\text{cosine}(\mathbf{v}_a,\mathbf{v}_b)=\frac{\mathbf{v}_a\!\cdot\!\mathbf{v}_b}{\|\mathbf{v}_a\|\;\|\mathbf{v}_b\|}
\]

  Values close to 1 indicate high semantic similarity.  
- **Distance**: Euclidean distance is useful for clustering or anomaly detection.  

#### 3.2 Training objectives  

| Objective type | Example | Core loss |
|----------------|---------|-----------|
| **Predict‑based** | Word2Vec Skip‑gram with negative sampling | \(-\log\sigma(\mathbf{v}_c^\top\mathbf{v}_w) -\sum_{i=1}^k\log\sigma(-\mathbf{v}_{c_i}^\top\mathbf{v}_w)\) |
| **Reconstruction‑based** | Auto‑encoding sentence embeddings | \(\| \mathbf{x} - \hat{\mathbf{x}} \|_2^2\) |

Predict‑based methods pull vectors of co‑occurring tokens together; reconstruction‑based methods force the model to preserve information when compressing the input.  

#### 3.3 Semantic properties  

- **Linear relationships**  

\[
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}
\]  

- **Clustering**: Synonyms and domain‑specific jargon form tight neighbourhoods.  
- **Domain transfer**: A well‑trained embedding can be reused on a downstream task with little fine‑tuning.  

#### 3.4 Visual illustration  

A t‑SNE or UMAP plot of a 300‑dim Word2Vec space typically shows:

1. **Colour‑coded clusters** (animals, colours, professions).  
2. **Analogical vectors** forming linear “arrows”.  

When you generate such a plot, look for:  

- **Compactness** of semantically related groups.  
- **Separation** between unrelated groups.  
- **Outliers**, which may indicate noisy tokens or OOV words.  

---  

### 4. Types of Embeddings and When to Use Them  

| Category | Description | Typical dimensionality | Best use‑cases |
|----------|-------------|------------------------|---------------|
| **Static word embeddings** (Word2Vec, GloVe, FastText) | Same vector for every occurrence of a token. | 100–300 | Small datasets, fast prototyping, classic pipelines (e.g., TF‑IDF + linear classifier). |
| **Subword / character embeddings** (FastText, Byte‑Pair) | Builds word vectors from n‑grams or characters; handles OOV words. | 100–300 | Low‑resource languages, domain‑specific vocabularies, misspelling‑rich corpora. |
| **Contextual embeddings** (ELMo, BERT, RoBERTa, GPT) | Token vector depends on surrounding tokens; deep transformer encoders. | 768–2048+ | Sentence‑level tasks (QA, NER, sentiment), transfer learning, multilingual models. |
| **Sentence / document embeddings** (Sentence‑BERT, Universal Sentence Encoder, Doc2Vec) | Fixed‑size representation for longer spans (sentence, paragraph, whole document). | 256–1024 | Retrieval, clustering, similarity search, semantic indexing. |
| **Multimodal embeddings** (CLIP, ALIGN) | Jointly trained text‑image space; the same vector can be compared across modalities. | 512–1024 | Cross‑modal retrieval, zero‑shot image classification, caption generation. |

#### Decision flow (ASCII)

```
               +-------------------+
               | What do you need? |
               +-------------------+
                         |
          +--------------+--------------+
          |                             |
   "Word‑level"                     "Sentence/Doc‑level"
          |                             |
   +------+-------+            +--------+--------+
   |              |            |                 |
"Static?"      "Context?"   "Static?"        "Context?"
   |              |            |                 |
   |   +----------+----+   +---+---+        +----+----+
   |   |               |   |       |        |         |
  Yes  No            Yes  Yes    No       Yes       No
   |   |               |   |       |        |         |
Word2Vec/GloVe   FastText   BERT   ELMo   SBERT   USE   Doc2Vec
```

---  

### 5. Practical Guide: From Raw Text to Ready‑to‑Use Vectors  

Below is a minimal end‑to‑end pipeline that uses **sentence‑transformers** (a wrapper around BERT‑style models) for contextual sentence vectors and **gensim** for static word vectors.

#### 5.1 Pre‑processing  

```python
import re, unicodedata

def simple_clean(text: str) -> str:
    """Lower‑case, normalize Unicode, and strip punctuation."""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", "", text)   # keep only alphanumerics + space
    return text.strip()
```

#### 5.2 Loading the models  

```python
# Static word vectors (gensim)
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True)

# Contextual sentence vectors (sentence‑transformers)
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-MiniLM-L6-v2')   # ~384‑dim, fast & small
```

#### 5.3 Caching embeddings  

```python
import os, pickle
from typing import List
import numpy as np

CACHE_PATH = "cached_vectors.pkl"

def embed_sentences(
    sentences: List[str],
    model: SentenceTransformer = sbert,
    cache_path: str = CACHE_PATH
) -> np.ndarray:
    """Encode `sentences` and cache the result on disk."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    vectors = model.encode(
        sentences,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(vectors, f)
    return vectors
```

*Tip:* For corpora larger than a few hundred thousand sentences, store vectors in a **FAISS** index or a database (e.g., SQLite BLOB) to avoid recomputation.  

#### 5.4 Plugging embeddings into a downstream model  

```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # x shape: (batch, embed_dim)
        return self.fc(x)

# Example: fine‑tune SBERT vectors for sentiment classification
model = SimpleClassifier(embed_dim=384, n_classes=2)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(sbert.parameters()),
    lr=2e-5,
)

# Pseudo‑training loop
for batch_text, batch_labels in dataloader:
    batch_vec = torch.tensor(
        sbert.encode(batch_text, convert_to_tensor=True)
    )
    logits = model(batch_vec)
    loss = nn.CrossEntropyLoss()(logits, batch_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

- **Frozen vs. trainable**: Freeze the embedding layer for small datasets; unfreeze (or partially unfreeze) when you have ample domain‑specific data.  

#### 5.5 Evaluation  

| Metric type | Example | When to use |
|-------------|---------|-------------|
| **Intrinsic** | WordSim‑353, Google Analogy | Quick sanity check of embedding quality. |
| **Extrinsic** | Accuracy / F1 on a downstream task (e.g., sentiment, NER) | Real‑world performance indicator. |

---  

### 6. Common Pitfalls & Best Practices  

| Pitfall | Why it hurts | Remedy |
|---------|--------------|--------|
| **Using cosine similarity on out‑of‑domain data** | The embedding space is biased toward its training corpus; scores become meaningless on unseen domains. | Validate on a domain‑specific probe set; consider domain‑adaptation (continue pre‑training on in‑domain text). |
| **Too high dimensionality → overfitting** | Large \(n\) increases model capacity and memory use, but may capture noise. | Apply PCA or TruncatedSVD to 50–100 dimensions; monitor validation loss. |
| **Embedded bias & fairness issues** | Training corpora contain gender, racial, or cultural stereotypes that surface in vectors. | Use debiasing methods (e.g., projection onto a gender‑neutral subspace) and audit downstream predictions. |
| **Version drift** | Pre‑trained models evolve; a model trained today may behave differently tomorrow. | Pin model hashes (e.g., `sentence-transformers/all-MiniLM-L6-v2@v1.0`) and store the binary in your artifact repository. |
| **Production drift** | Real‑time input distribution shifts; similarity thresholds become stale. | Deploy a monitoring job that computes average cosine distance to a reference set; trigger re‑training when drift exceeds a threshold. |

---  

### 7. Conclusion  

- **Recap** – Embeddings convert raw language into a numeric space where distance ≈ meaning. From one‑hot vectors to massive multimodal transformers, each generation has broadened what models can understand.  
- **Future direction** – Research is pushing toward *efficient* (quantised, LoRA‑adapted) and *multimodal* (text ↔ image ↔ audio) embeddings that retain rich semantics while fitting on‑device.  
- **Call to action** – Pick **two** families (e.g., a static Word2Vec model and a contextual SBERT model), run them on a small classification or similarity task, and compare:  

  1. **Performance** (accuracy / F1).  
  2. **Interpretability** (nearest‑neighbor inspection).  
  3. **Resource usage** (memory, latency).  

Seeing the trade‑offs firsthand will cement why embeddings are the lingua‑franca of modern NLP and where you should place them in your own pipelines.  

**Happy vectorizing!**  