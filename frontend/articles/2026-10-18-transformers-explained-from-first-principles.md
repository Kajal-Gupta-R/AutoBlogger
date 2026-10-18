# Transformers Explained from First Principles  

---  

## 1. Introduction  

### Why Transformers matter  
* State‑of‑the‑art performance on NLP, vision, speech, and multimodal tasks.  
* Enable few‑shot learning, in‑context reasoning, and massive pre‑training.  

### Brief history: from recurrent models to attention  
| Era | Main model | Limitation |
|-----|------------|------------|
| Early sequence models | **RNN → LSTM → GRU** | Vanishing gradients, limited parallelism |
| 2015 | **Attention** (Bahdanau *et al.*) | Decoupled “memory” from recurrence, but still used inside RNNs |
| 2017 | **Transformer** (Vaswani *et al.*) | Fully attention‑based, no recurrence, dramatically faster and more accurate |

### Goal of this article  
Demystify every component of a Transformer by constructing it from the ground up, using only linear‑algebraic and probabilistic primitives.

---  

## 2. Foundations: Linear Algebra & Probability Basics  

| Concept | Formal definition | Typical use in Transformers |
|---------|-------------------|-----------------------------|
| **Vector** | \(\mathbf{x}\in\mathbb{R}^{d}\) | Token embedding, query/key/value vectors |
| **Matrix** | \(\mathbf{W}\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}\) | Linear projections, feed‑forward layers |
| **Tensor** | \(\mathcal{X}\in\mathbb{R}^{B\times L\times d}\) | Batch of sequences (B = batch size, L = length) |
| **Dot‑product** | \(\mathbf{a}^{\top}\mathbf{b}= \sum_i a_i b_i\) | Core of attention scoring |
| **Softmax** | \(\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}\) | Turns scores into a probability distribution over positions |
| **Cross‑entropy loss** | \(\mathcal{L}= -\sum_i y_i\log\hat{y}_i\) | Language‑modeling objective |
| **Gradient descent** | \(\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}\) | Parameter update rule |
| **Back‑propagation** | Chain rule applied to compute \(\nabla_\theta\mathcal{L}\) | Efficiently propagates gradients through the whole graph |

> **Key point** – All operations in a Transformer are **tensor‑wise linear transforms** followed by **element‑wise non‑linearities** (softmax, ReLU, etc.).

---  

## 3. The Core Idea: Self‑Attention  

### 3.1 Intuition  

Self‑attention lets each token **look at every other token** in the same sequence and decide how much information to borrow from each. Because the operation is fully parallel, it overcomes the sequential bottleneck of RNNs.

### 3.2 Scaled Dot‑Product Attention  

Given an input matrix \(\mathbf{X}\in\mathbb{R}^{L\times d}\):

1. **Linear projections**  

\[
\mathbf{Q}= \mathbf{X}\mathbf{W}_Q,\qquad
\mathbf{K}= \mathbf{X}\mathbf{W}_K,\qquad
\mathbf{V}= \mathbf{X}\mathbf{W}_V,
\]

where \(\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V\in\mathbb{R}^{d\times d_k}\).  

2. **Score matrix** (dot‑product)  

\[
\mathbf{S}= \frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\in\mathbb{R}^{L\times L}.
\]  

3. **Row‑wise softmax**  

\[
\mathbf{A}= \operatorname{softmax}(\mathbf{S})\quad(\text{each row sums to 1}).
\]  

4. **Weighted sum of values**  

\[
\operatorname{Attention}(\mathbf{X}) = \mathbf{A}\mathbf{V}\in\mathbb{R}^{L\times d_v}.
\]  

**Why divide by \(\sqrt{d_k}\)?**  
When \(d_k\) grows, the variance of the dot‑product increases proportionally to \(d_k\). The scaling keeps the softmax input in a region where gradients are neither vanishing nor exploding, stabilising training.

### 3.3 Multi‑Head Attention  

A single attention head can only attend within one sub‑space. By computing **\(h\) heads in parallel**, the model can capture diverse patterns (e.g., syntax vs. semantics).

\[
\text{head}_i = \operatorname{Attention}\big(\mathbf{X}\mathbf{W}_Q^{(i)},
                                         \mathbf{X}\mathbf{W}_K^{(i)},
                                         \mathbf{X}\mathbf{W}_V^{(i)}\big),
\qquad i=1,\dots,h
\]

The heads are concatenated and projected back to the model dimension:

\[
\mathbf{Z}= \operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)\,\mathbf{W}_O,
\qquad
\mathbf{W}_O\in\mathbb{R}^{hd_v\times d}.
\]

### 3.4 Visual Walkthrough (toy sentence)  

```
Sentence: "The cat sat"
Positions: 1   2    3
Embeddings: x1  x2   x3

Step 1: Q, K, V = X·WQ , X·WK , X·WV
Step 2: Scores = Q·Kᵀ / √d_k
          ┌─────┬─────┬─────┐
          │ 0.2 │ 0.8 │ 0.1 │   ← token 1 attends most to token 2
          │ 0.5 │ 0.1 │ 0.4 │
          │ 0.3 │ 0.6 │ 0.2 │
          └─────┴─────┴─────┘
Step 3: Softmax rows → attention weights
Step 4: Weighted sum of V → new token representations
```

---  

## 4. Building the Transformer Block  

### 4.1 Positional Encodings  

Attention is permutation‑invariant, so we must inject order information. Two common schemes:

* **Learned**: a matrix \(\mathbf{P}\in\mathbb{R}^{L_{\max}\times d}\) that is updated during training.  
* **Sinusoidal** (original paper):  

\[
\text{PE}_{(pos,2i)}   = \sin\!\Bigl(\frac{pos}{10000^{2i/d}}\Bigr),\qquad
\text{PE}_{(pos,2i+1)} = \cos\!\Bigl(\frac{pos}{10000^{2i/d}}\Bigr).
\]

The input to the first block is  

\[
\mathbf{X}_0 = \mathbf{E} + \mathbf{PE},
\]

where \(\mathbf{E}\) contains token embeddings.

### 4.2 Residual Connections & Layer Normalization  

For any sub‑layer \(\mathcal{F}(\cdot)\),

\[
\mathbf{Y}= \operatorname{LayerNorm}\bigl(\mathbf{X}+ \mathcal{F}(\mathbf{X})\bigr).
\]

* **Residual (skip) connection** preserves the original signal, improving gradient flow.  
* **LayerNorm** normalizes across the feature dimension, avoiding the batch‑dependence of BatchNorm.

### 4.3 Position‑wise Feed‑Forward Network (FFN)  

Applied independently to each token:

\[
\text{FFN}(\mathbf{x}) = \max\bigl(0,\;\mathbf{x}\mathbf{W}_1+\mathbf{b}_1\bigr)\mathbf{W}_2+\mathbf{b}_2,
\]

typically with dimensions \(d_{\text{model}}=512\) and \(d_{\text{ff}}=2048\).

### 4.4 Encoder and Decoder Stacks  

```
Encoder layer
-------------
Input → Add&Norm → Multi‑Head Self‑Attention → Add&Norm → FFN → Add&Norm → Output
```

```
Decoder layer
-------------
Input → Add&Norm → Masked Multi‑Head Self‑Attention → Add&Norm →
        Multi‑Head Cross‑Attention (keys/values from encoder) → Add&Norm →
        FFN → Add&Norm → Output
```

Both stacks are repeated **N** times (commonly \(N=6\)).

---  

## 5. Training the Model  

### 5.1 Objective Functions  

* **Causal language modeling** (decoder‑only):  

\[
\mathcal{L}= -\sum_{t=1}^{L}\log p\bigl(y_t \mid y_{<t}\bigr).
\]  

* **Sequence‑to‑sequence** (encoder‑decoder):  

\[
\mathcal{L}= -\sum_{t=1}^{L_{\text{tgt}}}\log p\bigl(y_t \mid y_{<t}, \text{src}\bigr).
\]  

Both are instantiated with **cross‑entropy** loss.

### 5.2 Masking Strategies  

| Mask type | Used in | Effect |
|-----------|---------|--------|
| **Padding mask** | Encoder self‑attention, decoder cross‑attention | Prevents attention to padded positions. |
| **Causal (look‑ahead) mask** | Decoder self‑attention | Enforces the autoregressive constraint (no peeking at future tokens). |

*PyTorch‑style implementation of the causal mask*:

```python
def subsequent_mask(size: int) -> torch.BoolTensor:
    """Mask out subsequent positions (size, size)."""
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return mask  # True entries are masked
```

### 5.3 Optimizer & Learning‑Rate Schedule  

* **Adam** with \(\beta_1=0.9,\ \beta_2=0.98,\ \epsilon=10^{-9}\) (as in the original paper).  
* **Warm‑up + inverse‑square‑root decay**  

\[
\text{lr}(step)= d_{\text{model}}^{-0.5}\,
\min\!\bigl(step^{-0.5},\; step\cdot\text{warmup}^{-1.5}\bigr).
\]  

After the warm‑up phase, the schedule behaves like \(step^{-0.5}\).  Cosine decay can be added for fine‑tuning.

### 5.4 Regularisation Techniques  

| Technique | Purpose |
|-----------|---------|
| **Dropout** (0.1 – 0.3) on attention weights and FFN activations | Mitigates over‑fitting and improves robustness. |
| **Label smoothing** (ε ≈ 0.1) | Prevents the model from becoming over‑confident, yielding higher BLEU/accuracy. |
| **Early stopping** on a held‑out validation set | Saves compute when further training offers negligible gains. |

---  

## 6. From Theory to Practice  

### 6.1 Popular Implementations  

| Library | Core API | Typical use case |
|---------|----------|------------------|
| **Hugging Face Transformers** | `AutoModel.from_pretrained` | Ready‑to‑use pretrained models for inference or fine‑tuning. |
| **TensorFlow / Keras** | `tf.keras.layers.MultiHeadAttention` | Seamless integration into TF pipelines. |
| **PyTorch** | `torch.nn.Transformer` / `torch.nn.MultiheadAttention` | Low‑level control for research prototypes. |

### 6.2 Mini‑Project: Implement a Single Transformer Layer from Scratch (PyTorch)

```python
import torch
import torch.nn as nn
import math

class SimpleTransformerLayer(nn.Module):
    """A minimal encoder‑style Transformer layer."""
    def __init__(self, d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: (seq_len, batch, d_model)
        Returns the same shape as src.
        """
        # ---- Self‑attention ----
        attn_output, _ = self.self_attn(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = self.norm1(src + self.dropout(attn_output))

        # ---- Feed‑forward ----
       