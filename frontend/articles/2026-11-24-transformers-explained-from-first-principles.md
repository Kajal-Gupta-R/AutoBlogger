# Transformers Explained from First Principles  

---  

## 1. Introduction  

- **Motivation** – Modern AI systems (LLMs, vision‑language models, speech recognizers) owe most of their performance to the *Transformer* architecture. Its ability to capture long‑range dependencies while remaining highly parallelizable makes it the work‑horse of today’s deep‑learning pipelines.  
- **Brief history** – Early sequence models relied on **RNNs** (LSTM, GRU) or **CNNs** to process tokens step‑by‑step. The 2017 paper *“Attention Is All You Need”* showed that a stack of attention layers alone can replace recurrence and convolution, achieving state‑of‑the‑art results on translation and beyond.  
- **Goal of this article** – Derive the Transformer from elementary concepts (vectors, similarity, linear maps) and build an intuition that survives the many variants that followed (BERT, GPT, ViT, etc.).  

---  

## 2. Foundations of Sequence Modeling  

### 2.1. Representing discrete data  

| Symbol | Meaning |
|--------|---------|
| \(x\) | raw token (word, sub‑word, pixel patch, …) |
| \(\mathbf{e}_x \in \mathbb{R}^{d}\) | *embedding* – learned dense vector for token \(x\) |
| \(X = [\mathbf{e}_{x_1},\dots,\mathbf{e}_{x_n}] \in \mathbb{R}^{n\times d}\) | input matrix (sequence length \(n\), hidden size \(d\)) |

Embedding layers map categorical indices to a continuous space where linear algebra can be applied.  

### 2.2. Limitations of traditional recurrent and convolutional approaches  

| Approach | Strength | Weakness |
|----------|----------|----------|
| RNN / LSTM | Handles variable‑length sequences; respects order | Sequential bottleneck → poor GPU utilization; gradients decay over long horizons |
| 1‑D CNN | Parallel across time; captures local n‑grams | Fixed receptive field; deep stacks needed for global context → high latency |

Both struggle to **simultaneously** provide *global context* and *full parallelism*.  

### 2.3. Why we need global context **and** parallelism  

- **Global context** – many tasks (translation, code completion) require a token to attend to any other token, regardless of distance.  
- **Parallelism** – modern hardware thrives on batched matrix multiplications; removing step‑wise dependencies yields massive speed‑ups.  

Attention satisfies both requirements, which leads to the Transformer’s core design.  

---  

## 3. The Attention Mechanism  

### 3.1. Query‑key‑value formulation  

For a sequence represented by \(X\) we compute three linear projections  

\[
\mathbf{Q}=XW_Q,\qquad 
\mathbf{K}=XW_K,\qquad 
\mathbf{V}=XW_V,
\]

with \(W_Q,W_K,W_V\in\mathbb{R}^{d\times d_k}\).  

- **Query** \(\mathbf{q}_i\): what we are looking for at position \(i\).  
- **Key**   \(\mathbf{k}_j\): what each token \(j\) offers.  
- **Value** \(\mathbf{v}_j\): the content that will be aggregated.  

Similarity between a query and a key is measured by a dot product; larger values indicate higher relevance.  

### 3.2. Scaled dot‑product attention  

Raw scores: \(s_{ij}= \mathbf{q}_i^{\top}\mathbf{k}_j\).  
When \(d_k\) is large the variance of \(s_{ij}\) grows, which can push the softmax into saturated regimes. To keep the gradients stable we **scale** the scores:

\[
\alpha_{ij}= \frac{\exp\!\bigl(s_{ij}/\sqrt{d_k}\bigr)}
                 {\sum_{j'=1}^{n}\exp\!\bigl(s_{ij'}/\sqrt{d_k}\bigr)}.
\]

The output for position \(i\) is the weighted sum of values  

\[
\mathbf{z}_i = \sum_{j=1}^{n}\alpha_{ij}\,\mathbf{v}_j .
\]

Collecting all positions gives the matrix‑wise formulation  

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V .
\]

### 3.3. Multi‑head attention  

Instead of a single \((Q,K,V)\) triple we split the embedding dimension into \(h\) *heads*:

\[
\text{head}_h = 
\operatorname{Attention}\bigl(QW_Q^{(h)},\,
                               KW_K^{(h)},\,
                               VW_V^{(h)}\bigr),
\]

where each head works in a sub‑space of size \(d_k = d/h\).  

**Why multiple heads?**  

1. **Diversity** – each head can focus on a different linguistic pattern (e.g., syntax vs. semantics).  
2. **Expressivity** – a single linear map cannot simultaneously capture several similarity notions.  
3. **Stability** – smaller \(d_k\) reduces the scaling factor, improving the softmax’s numerical behavior.  

The heads are concatenated and projected back to the original dimension with a learned matrix \(W_O\).  

### 3.4. Visual intuition (numeric example)  

```
Tokens:   T1   T2   T3
Embeds:  [1,0] [0,1] [1,1]   (d = 2)

Q = K = V = Embeds            # identity projection for illustration
sqrt(d_k)=sqrt(2)=1.414

Score matrix S = Q·Kᵀ
   [[1,0,1],
    [0,1,1],
    [1,1,2]]

Softmax rows (≈):
   [[0.42,0.15,0.42],
    [0.15,0.42,0.42],
    [0.21,0.21,0.58]]

Output Z = Softmax·V
   [[0.57,0.57],
    [0.57,0.57],
    [0.79,0.79]]
```

The third token (the sum of the first two) receives the strongest attention from all positions, illustrating **global context** in a single step.  

---  

## 4. The Transformer Architecture  

### 4.1. Encoder block  

```
Input
   → Multi‑Head Self‑Attention
   → Add & LayerNorm
   → Position‑wise Feed‑Forward
   → Add & LayerNorm
   → Output
```

- **Self‑attention** lets every token attend to all others (including itself).  
- **Feed‑forward** consists of two linear layers with a non‑linearity (typically GELU):  

\[
\text{FFN}(x)=\text{GELU}\bigl(xW_1+b_1\bigr)W_2+b_2 .
\]

- **Residual connections** + **LayerNorm** stabilize training and enable deep stacks.  

### 4.2. Decoder block  

```
Input
   → Masked Multi‑Head Self‑Attention
   → Add & LayerNorm
   → Encoder‑Decoder Attention
   → Add & LayerNorm
   → Position‑wise Feed‑Forward
   → Add & LayerNorm
   → Output
```

- **Masked self‑attention** uses a causal mask so that position \(i\) cannot attend to positions \(>i\).  
- **Encoder‑decoder attention** lets the decoder query the full encoder representation.  

### 4.3. Positional encodings  

Attention is permutation‑invariant, so order must be injected explicitly. Two widely used schemes:

| Type | Formula | Property |
|------|---------|----------|
| **Sinusoidal** | \(\displaystyle \text{PE}_{(pos,2i)} = \sin\!\bigl(\frac{pos}{10000^{2i/d}}\bigr),\quad \text{PE}_{(pos,2i+1)} = \cos\!\bigl(\frac{pos}{10000^{2i/d}}\bigr)\) | Fixed, extrapolates to longer sequences |
| **Learned** | \(\mathbf{p}_{pos}\in\mathbb{R}^{d}\) learned jointly with the model | Flexible, may overfit to the training length |

The final token representation is \(\mathbf{e}_x + \mathbf{p}_{pos}\).  

### 4.4. Overall model flow and parameter count  

A **base** Transformer (Vaswani *et al.*, 2017) uses  

| Component | Dimensions | Approx. parameters |
|-----------|------------|--------------------|
| Token embedding (vocab × \(d\)) | \(V \times 512\) | \(V\cdot512\) |
| Positional encoding (max‑len × \(d\)) | — | negligible |
| One encoder layer (6 heads, \(d=512\)) | — | ≈ 7 M |
| One decoder layer (6 heads, \(d=512\)) | — | ≈ 7 M |
| 6 encoder + 6 decoder layers | — | ≈ 85 M total (including output projection) |

Parameter count scales **linearly** with depth, width, and number of heads.  

---  

## 5. Training Objectives and Variants  

### 5.1. Autoregressive language modeling (teacher forcing)  

During training the decoder receives the *ground‑truth* previous tokens (teacher forcing) and predicts the next token via a softmax over the vocabulary.  

\[
\mathcal{L}_{\text{LM}} = -\sum_{t}\log p(x_t\mid x_{<t}) .
\]

### 5.2. Masked language modeling (BERT)  

BERT randomly masks ≈ 15 % of input tokens and trains the encoder to predict them, allowing **full bidirectional attention** (each token sees both left and right context).  

\[
\mathcal{L}_{\text{MLM}} = -\sum_{m\in\mathcal{M}}\log p(x_m\mid X_{\setminus m}) .
\]

### 5.3. Scaling laws  

Empirical studies (Kaplan *et al.*, 2020) show that loss \(L\) follows a power‑law relationship with model size, data, and compute:

\[
L(N, D, C) \approx \alpha N^{-\beta} + \gamma D^{-\delta} + \eta C^{-\zeta},
\]

where  

- \(N\) = number of training tokens,  
- \(D\) = number of model parameters,  
- \(C\) = FLOP budget,  

and \(\alpha,\beta,\gamma,\delta,\eta,\zeta > 0\). Balanced scaling (increasing all three together) yields the most efficient performance gains.  

### 5.4. Common extensions  

| Extension | Idea | Typical impact |
|-----------|------|----------------|
| **Relative positional bias** | Encode relative distance \(i-j\) instead of absolute positions | Improves extrapolation to longer sequences |
| **Sparse attention** (e.g., Longformer, BigBird) | Restrict each query to a subset of keys (local + global) | Reduces memory from \(O(n^2)\) to \(O(n)\) for very long inputs |
| **Adapters** | Insert small bottleneck modules per layer while freezing the base model | Parameter‑efficient fine‑tuning across many downstream tasks |
| **FlashAttention / xFormers kernels** | Fuse softmax and matmul, use tiling to avoid intermediate buffers | Up to 2–3× speed‑up and lower memory usage |

---  

## 6. From Theory to Practice  

### 6.1. Implementation sketch (PyTorch‑style) of a single multi‑head attention layer  

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (B, n_heads, seq_len, d_k)
    mask  : (B, 1, seq_len, seq_len) with 0 where attention is prohibited
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1