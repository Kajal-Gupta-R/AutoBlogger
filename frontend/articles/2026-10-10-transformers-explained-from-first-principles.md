# Transformers Explained from First Principles  

## 1. Introduction  

- **Motivation** – Transformers dominate NLP, vision, and multimodal tasks because they scale efficiently with data and compute while capturing long‑range dependencies that recurrent networks struggle with.  
- **Historical context** – Early sequence models (RNNs, LSTMs) processed tokens strictly sequentially, limiting parallelism. The *attention* mechanism (Bahdanau 2015) decoupled content from position, and the 2017 “Attention Is All You Need” paper showed that a stack of attention layers alone can replace recurrence entirely.  
- **Goal** – Derive the Transformer from first principles, exposing every mathematical operation so you can reason about design choices instead of treating the model as a black box.  

---

## 2. Foundations: Linear Algebra & Probability Primer  

| Concept | Relevance in Transformers |
|---------|---------------------------|
| **Vector** \(\mathbf{x}\in\mathbb{R}^d\) | Token embedding, query/key/value vectors |
| **Matrix** \(\mathbf{W}\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}\) | Linear projections (e.g., \(\mathbf{Q}= \mathbf{X}\mathbf{W}_Q\)) |
| **Tensor** \(\mathcal{X}\in\mathbb{R}^{n\times d}\) | Batch of \(n\) token embeddings |
| **Dot‑product** \(\mathbf{a}^\top\mathbf{b}\) | Similarity between queries and keys |
| **Softmax** \(\operatorname{softmax}(\mathbf{z})_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}\) | Turns raw scores into a categorical probability distribution over tokens |
| **Gradient descent** – \(\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}\) | Optimises all weight matrices jointly |

*Key point*: The softmax converts the dot‑product similarity matrix into attention weights that sum to 1 for each query, i.e. a discrete probability distribution over sequence positions.  

---

## 3. The Core Idea: Scaled Dot‑Product Attention  

Given a sequence matrix \(\mathbf{X}\in\mathbb{R}^{n\times d_{\text{model}}}\):

1. **Linear projections**  

   \[
   \mathbf{Q}= \mathbf{X}\mathbf{W}_Q,\qquad
   \mathbf{K}= \mathbf{X}\mathbf{W}_K,\qquad
   \mathbf{V}= \mathbf{X}\mathbf{W}_V,
   \]

   where \(\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V\in\mathbb{R}^{d_{\text{model}}\times d_k}\).  

2. **Score matrix (raw logits)**  

   \[
   \mathbf{S}= \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{n\times n}.
   \]

   The factor \(\sqrt{d_k}\) prevents the dot‑product magnitude from growing with dimensionality, which would otherwise push the softmax into saturated regimes.  

3. **Softmax normalisation (row‑wise)**  

   \[
   \mathbf{A}= \operatorname{softmax}(\mathbf{S})\in\mathbb{R}^{n\times n}.
   \]  

   Each row \(\mathbf{a}_i\) is a probability distribution over all tokens for query \(i\).  

4. **Weighted sum of values**  

   \[
   \text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathbf{A}\mathbf{V}\in\mathbb{R}^{n\times d_v}.
   \]

### Visual intuition  

The matrix \(\mathbf{A}\) can be visualised as a heatmap: bright cells indicate strong focus of a query on particular keys.  

### Minimal PyTorch implementation  

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    mask: (batch, seq_len, seq_len) with 0 for allowed positions, -inf otherwise
    Returns (output, attention_weights)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
    if mask is not None:
        scores = scores + mask          # mask adds -inf where attention is prohibited
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

---

## 4. Multi‑Head Attention & Positional Encoding  

### Why multiple heads?  

A single attention head projects inputs into one sub‑space, limiting the relationships it can capture. By using \(h\) heads with independent projection matrices \(\mathbf{W}_Q^{(i)},\mathbf{W}_K^{(i)},\mathbf{W}_V^{(i)}\), the model learns *diverse* similarity metrics in parallel.

**Computation**  

1. Compute attention for each head:  
   \(\text{head}_i = \text{Attention}(Q_i, K_i, V_i)\).  
2. Concatenate the heads:  

   \[
   \mathbf{H}= \operatorname{concat}\big(\text{head}_1,\dots,\text{head}_h\big)
   \in\mathbb{R}^{n\times (h\cdot d_v)}.
   \]  

3. Final linear projection:  

   \[
   \mathbf{O}= \mathbf{H}\mathbf{W}_O,\qquad
   \mathbf{W}_O\in\mathbb{R}^{(h\cdot d_v)\times d_{\text{model}}}.
   \]

### Positional encodings  

Attention is permutation‑invariant; we must inject order information.

| Type | Formula (position \(p\), dimension \(i\)) |
|------|-------------------------------------------|
| **Sinusoidal** | \(\displaystyle \text{PE}_{p,2i}= \sin\!\Big(\frac{p}{10000^{2i/d_{\text{model}}}}\Big)\)  <br> \(\displaystyle \text{PE}_{p,2i+1}= \cos\!\Big(\frac{p}{10000^{2i/d_{\text{model}}}}\Big)\) |
| **Learned** | \(\mathbf{P}\in\mathbb{R}^{\text{max\_len}\times d_{\text{model}}}\) trained jointly with the model |

The embedding for token \(t\) becomes \(\mathbf{x}_t = \mathbf{e}_t + \text{PE}_t\).

---

## 5. The Transformer Architecture  

### Encoder block  

```
Input
   │
   ├─► Add & LayerNorm
   │        │
   │   Multi‑Head Self‑Attention
   │        │
   ├─► Add & LayerNorm
   │        │
   └─► Position‑wise Feed‑Forward
```

- **Self‑attention**: each token attends to *all* tokens in the same layer.  
- **Feed‑forward network (FFN)**: two linear layers with a non‑linearity (ReLU or GELU) applied position‑wise  

  \[
  \text{FFN}(x)=\max(0, x\mathbf{W}_1 + b_1)\mathbf{W}_2 + b_2.
  \]  

- **Residual connections** (the “Add” steps) and **LayerNorm** stabilise training.  

### Decoder block  

1. **Masked self‑attention** – a causal mask prevents a token from attending to future positions.  
2. **Encoder‑decoder (cross) attention** – queries come from the decoder, keys/values from the encoder output.  
3. **FFN** – identical to the encoder’s.  

Each sub‑layer is wrapped with residual + LayerNorm, exactly as in the encoder.  

### Stacking  

- **Depth** = number of encoder (or decoder) layers (e.g., 12, 24, 48).  
- **Width** = model dimension \(d_{\text{model}}\) (e.g., 768, 1024).  
Increasing depth improves abstraction; increasing width boosts capacity but also memory and compute.  

### Complexity  

| Operation | Time complexity | Memory complexity |
|-----------|----------------|-------------------|
| Self‑attention | \(O(n^2 d_{\text{model}})\) (because of the \(n\times n\) score matrix) | \(O(n^2)\) for the attention matrix |
| Feed‑forward | \(O(n d_{\text{model}}^2)\) | \(O(n d_{\text{model}})\) |

The total number of trainable parameters is roughly  

\[
\underbrace{(d_{\text{model}} d_k + d_{\text{model}} d_v + d_k d_v)}_{\text{per head}}
\times h
\;+\; 2d_{\text{model}}^2 \;\;(\text{FFN}) \;+\; \text{embeddings}.
\]

---

## 6. Training Objectives & Popular Variants  

| Model | Objective | Key architectural difference |
|-------|-----------|-------------------------------|
| **BERT** (Bidirectional Encoder Representations from Transformers) | Masked Language Modeling (MLM): predict randomly masked tokens. | Encoder‑only; bidirectional context. |
| **GPT** (Generative Pre‑trained Transformer) | Autoregressive LM: predict the next token given all previous ones. | Decoder‑only stack with causal mask; unidirectional. |
| **Vision Transformer (ViT)** | Image classification via patch embeddings + standard encoder. | Same architecture, different tokenisation (image patches). |
| **RAG** (Retrieval‑Augmented Generation) | Combine a frozen retriever with a generative decoder. | Adds a cross‑attention step over retrieved documents. |

**Fine‑tuning** – Replace the final head (e.g., classification, QA) and continue training on downstream data with a smaller learning rate.

---

## 7. Conclusion  

- **Core principles** – Scaled dot‑product attention turns similarity scores into a probability distribution, enabling differentiable weighted aggregation of information across any distance. Multi‑head attention enriches this by learning several sub‑spaces simultaneously, while positional encodings restore order.  
- **Practical takeaways**  
  1. Visualise the attention matrix to verify that the model focuses on expected tokens.  
  2. Reduce memory when \(n\) is large by using linear‑time approximations (e.g., Linformer, Performer).  
  3. When implementing from scratch, double‑check dimensions: \(\mathbf{Q},\mathbf{K},\mathbf{V}\) must share the same head dimension \(d_k\).  
- **Outlook** – Ongoing research is pushing beyond quadratic attention, integrating retrieval, and unifying modalities. Mastering the mathematics presented here remains the foundation for any future innovation.  