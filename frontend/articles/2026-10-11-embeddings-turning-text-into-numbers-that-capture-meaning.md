# Embeddings – Turning Text into Numbers That Capture Meaning  

---  

## Introduction  

- **Why raw text isn’t enough** – Machines work with numbers. Plain strings have no numeric structure that algorithms can manipulate efficiently.  
- **What an embedding is** – A dense, continuous‑valued vector that represents a linguistic unit (word, subword, sentence, etc.). Geometric relationships between vectors reflect semantic relationships.  
- **What you’ll learn**  
  1. Core concepts behind token‑level vectors.  
  2. How embeddings are trained (static vs. contextual).  
  3. The main families of text embeddings.  
  4. Real‑world use cases.  
  5. A quick‑start guide with libraries, code, and evaluation tips.  

---  

## 1. Foundations – From Tokens to Vectors  

### 1.1 Tokens, vocabularies, and one‑hot encodings  

- **Token** – The smallest unit fed to a model (word, subword, character).  
- **Vocabulary** – The set of unique tokens observed during preprocessing.  
- **One‑hot** – A binary vector of length `|V|` with a single `1` at the token’s index.  

### 1.2 Limitations of sparse representations  

| Issue | Consequence |
|-------|-------------|
| **Dimensionality explosion** – vocabularies of 50 k–500 k → equally large vectors | High memory use and slow computation |
| **No notion of similarity** – all one‑hots are orthogonal | “king” and “queen” are as distant as “king” and “banana” |
| **Inefficient learning** – gradient updates touch almost every dimension | Slower convergence |  

### 1.3 The core idea of dense, continuous vectors  

Map each token to a low‑dimensional (e.g., 100–1 024) real‑valued vector whose values are learned from data. This allows the model to capture statistical regularities and semantic nuances.  

### 1.4 Intuition: geometric proximity ≈ semantic similarity  

- **Cosine similarity** or **Euclidean distance** between vectors correlates with meaning.  
- Classic example: `vec("king") - vec("man") + vec("woman") ≈ vec("queen")`.  
- Visualizations (t‑SNE, UMAP) often reveal clusters of synonyms, topics, or syntactic categories.  

---  

## 2. How Embeddings Are Learned  

### 2.1 Predictive vs. count‑based approaches  

| Approach | Core principle | Typical loss |
|----------|----------------|--------------|
| **word2vec (Skip‑Gram / CBOW)** | Predict surrounding words (or the target from context) | Negative‑sampling cross‑entropy |
| **GloVe** | Factorize a global word‑co‑occurrence matrix | Weighted least‑squares on log‑co‑occurrence counts |
| **SVD on PMI** | Decompose a Pointwise Mutual Information matrix | Implicit reconstruction error minimization |  

### 2.2 Training objectives  

- **CBOW** – Given context tokens, predict the centre word.  
- **Skip‑Gram** – Given a word, predict its surrounding context.  
- **Negative sampling** – Approximate softmax by contrasting true pairs with randomly sampled “negative” pairs.  
- **GloVe co‑occurrence loss** – Minimize  
  \[
  \bigl(w_i^{\top} w_j + b_i + b_j - \log X_{ij}\bigr)^2
  \]  
  where \(X_{ij}\) is the co‑occurrence count of tokens *i* and *j*.  

### 2.3 From static to contextual: the rise of Transformers  

| Model | Embedding type | Training scheme |
|-------|----------------|-----------------|
| **ELMo** | Contextual word vectors from a bidirectional LSTM | Forward + backward language modelling |
| **BERT / RoBERTa** | Deep contextual token embeddings | Masked Language Modeling (MLM) (+ Next Sentence Prediction for BERT) |
| **GPT** | Autoregressive contextual embeddings | Causal language modelling |  

Transformers replace a fixed lookup table with **dynamic** representations that depend on the entire input sequence.  

### 2.4 Fine‑tuning vs. frozen embeddings  

- **Frozen** – Use pretrained vectors as static features (e.g., feed them to a classifier).  
- **Fine‑tuned** – Back‑propagate the downstream task loss through the embedding layer, adapting vectors to the target domain.  

---  

## 3. Types of Text Embeddings  

### 3.1 Word‑level (static) embeddings  

- **Examples:** word2vec, GloVe, FastText (static mode)  
- **Pros:** Small memory footprint, fast inference  
- **Cons:** Cannot distinguish senses (e.g., “bank” → river vs. finance)  

### 3.2 Subword / character‑level embeddings  

- **FastText** – Represents a word as the sum of its character n‑gram vectors.  
- **Byte‑Pair Encoding (BPE)** – Learns a vocabulary of subword units; each subword has its own embedding.  

### 3.3 Sentence and paragraph embeddings  

| Method | Core idea |
|--------|-----------|
| **Sent2Vec** | Average word vectors plus n‑gram features, trained with a sentence‑level objective |
| **InferSent** | Supervised training on Natural Language Inference data using a BiLSTM encoder |
| **Universal Sentence Encoder (USE)** | Deep Averaging Network (DAN) or Transformer encoder trained on a variety of tasks |  

### 3.4 Contextual embeddings (BERT, GPT, ELMo)  

- Produce a distinct vector for each token position, conditioned on the full surrounding text.  
- Typically extracted from the hidden state of the last (or a middle) transformer layer.  

### 3.5 Multimodal embeddings (brief)  

Joint spaces where text vectors align with visual or acoustic vectors (e.g., **CLIP**, **AudioCLIP**).  
Enable cross‑modal retrieval such as “a photo of a dog” ↔ image embedding of a dog.  

---  

## 4. Practical Applications  

### 4.1 Information retrieval & semantic search  

Index document embeddings (e.g., with **FAISS**) and retrieve by cosine similarity to a query embedding.  

### 4.2 Text classification, sentiment analysis, and intent detection  

Feed sentence embeddings into a linear classifier or fine‑tune a transformer head.  

### 4.3 Clustering, topic modelling, and visualization  

Use embeddings as features for K‑means, DBSCAN, or visualise high‑dimensional structure with UMAP/t‑SNE.  

### 4.4 Transfer learning & downstream adaptation  

Pretrained models provide a strong initialization; only a small labelled set is needed for fine‑tuning.  

### 4.5 Real‑world case studies  

| Domain | Use case | Embedding choice |
|--------|----------|-----------------|
| **E‑commerce** | Product recommendation via similarity search | Sentence‑level USE + FAISS |
| **Customer support** | Intent detection for routing tickets | Fine‑tuned BERT classifier |
| **Media streaming** | Tag‑based playlist generation using multimodal CLIP embeddings | Text‑image joint space |  

---  

## 5. Getting Started – Hands‑On Guide  

### 5.1 Choosing the right technique  

| Task | Recommended embedding |
|------|------------------------|
| Short, domain‑specific vocabularies | FastText (subword) |
| Sentence‑level similarity | Universal Sentence Encoder or Sentence‑Transformers |
| Complex language understanding | Fine‑tuned BERT / RoBERTa |
| Low‑resource environment | Static GloVe + linear classifier |  

### 5.2 Popular libraries & APIs  

- **gensim** – word2vec, FastText, GloVe.  
- **spaCy** – Built‑in static vectors, easy pipeline integration.  
- **🤗 Transformers** – BERT, RoBERTa, GPT, etc.  
- **Sentence‑Transformers** – Ready‑to‑use sentence embeddings (SBERT).  

### 5.3 Loading, saving, and inspecting embeddings  

```python
# Load a Sentence‑Transformer model and inspect a vector
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
sentence = "Embeddings turn text into numbers that capture meaning."
vec = model.encode(sentence)          # → numpy array, shape (384,)

print("Shape:", vec.shape)
print("Norm :", np.linalg.norm(vec))
print("First 5 dimensions:", vec[:5])
```

- **Saving:** `np.save('my_vec.npy', vec)`  
- **Loading:** `vec = np.load('my_vec.npy')`  

### 5.4 Common pitfalls  

| Pitfall | Symptom | Mitigation |
|---------|---------|------------|
| **Out‑of‑vocabulary (OOV) tokens** | Random or zero vectors, degraded performance | Use subword models (FastText, BPE) or a dedicated `[UNK]` embedding |
| **Too high dimensionality** | Overfitting, unnecessary memory consumption | Start with 100–300 dimensions for static vectors; 256–768 for transformers |
| **Bias** | Embeddings encode gender, racial, or cultural stereotypes | Apply debiasing methods (e.g., Hard‑Debias) and monitor downstream impact |
| **Lack of interpretability** | Hard to explain predictions | Visualise nearest neighbours, project with UMAP, or use probing classifiers |  

### 5.5 Evaluation metrics  

- **Cosine similarity** – Primary metric for semantic similarity tasks.  
- **Intrinsic benchmarks** – Word‑Similarity (SimLex‑999, WS‑353), Analogy (Google Analogy).  
- **Extrinsic benchmarks** – Downstream task performance (accuracy, F1) after fine‑tuning.  

```python
from sklearn.metrics.pairwise import cosine_similarity

a = model.encode("machine learning")
b = model.encode("artificial intelligence")
print("Cosine similarity:", cosine_similarity([a], [b])[0][0])
```

---  

## Conclusion  

Embeddings bridge the gap between raw text and the numeric world of machine learning. By converting linguistic units into dense vectors, they enable:

- **Semantic reasoning** through geometric operations.  
- **Transfer learning** across tasks and domains.  
- **Scalable retrieval** and clustering in high‑dimensional spaces.  

**Future directions** include ever larger language models that learn unified multimodal embeddings, continual‑learning schemes that adapt vectors on‑the‑fly, and research on reducing bias while preserving expressive power.  

**Call to action:** Grab a small dataset, spin up a Sentence‑Transformer model, generate sentence vectors, and experiment with a downstream classifier or a semantic‑search index. The best way to internalise embeddings is to build a one‑off pipeline and watch the numbers you produce begin to capture meaning.  