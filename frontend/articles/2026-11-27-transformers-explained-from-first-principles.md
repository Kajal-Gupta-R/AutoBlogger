## Transformers Explained from First Principles  

### 1. Introduction  
- **Why the Transformer matters** – It powers large language models (LLMs), vision‑transformers, and multimodal systems that dominate today’s state‑of‑the‑art benchmarks.  
- **Brief history** – Recurrent networks (LSTM, GRU) processed sequences step‑by‑step, limiting parallelism. The 2017 *Attention Is All You Need* paper replaced recurrence with **self‑attention**, enabling massive speed‑ups and scaling.  
- **Goal of this article** – Build intuition *from the ground up*: start with the algebraic primitives, derive the attention equations, and assemble a full Transformer block.  

---

### 2. Foundations: Linear Algebra & Probability Refresher  

| Concept | Quick reminder | Relevance to Transformers |
|---------|----------------|---------------------------|
| **Vectors & matrices** | \( \mathbf{x}\in\mathbb{R}^d,\;\mathbf{W}\in\mathbb{R}^{d\times d'} \) | Token embeddings and the linear projections that produce queries, keys, and values are matrix‑vector products. |
| **Tensors** | 3‑D tensor \( \mathbf{X}\in\mathbb{R}^{B\times L\times d} \) (batch, seq‑len, dim) | Whole batches are processed in a single GPU kernel. |
| **Dot product** | \( \mathbf{a}\!\cdot\!\mathbf{b}= \sum_i a_i b_i \) | Measures similarity between queries and keys. |
| **Softmax** | \( \sigma(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}} \) | Turns raw scores into a probability distribution over tokens. |
| **Gradient descent (optional)** | \( \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta) \) | Underlies all training; Adam W is a widely used variant. |

---

### 3. The Core Idea: Self‑Attention  

#### 3.1. What “attention” means in NLP  
Attention answers *“Which other tokens should I look at when processing this token?”* It produces a weighted sum of **value** vectors, where the weights reflect relevance measured by **query–key** similarity.

#### 3.2. Query, Key, and Value vectors – linear projections  
Given an input matrix \( \mathbf{X}\in\mathbb{R}^{L\times d_{\text{model}}} \) ( L tokens, hidden size \(d_{\text{model}}\) ):

\[
\mathbf{Q}= \mathbf{X}\mathbf{W}_Q,\qquad
\mathbf{K}= \mathbf{X}\mathbf{W}_K,\qquad
\mathbf{V}= \mathbf{X}\mathbf{W}_V
\]

where \( \mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V\in\mathbb{R}^{d_{\text{model}}\times d_k} \) are learned. Each token now has three *views*: a **query** (what it seeks), a **key** (how it can be found), and a **value** (what it contributes).

#### 3.3. Scaled dot‑product attention  

\[
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
= \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\!\top}}{\sqrt{d_k}}\right)\mathbf{V}
\]

* **\( \mathbf{Q}\mathbf{K}^{\!\top} \)** → similarity matrix ( L × L ).  
* **\( \sqrt{d_k} \)** scaling prevents large dot‑products from saturating the softmax.  
* **softmax** → row‑wise probability distribution → attention weights.  
* Multiplying by **V** aggregates information from all positions, weighted by relevance.

**PyTorch‑style implementation**

```python
import torch
def scaled_dot_product_attention(Q, K, V, mask=None):
    dk = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / dk**0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn
```

#### 3.4. Multi‑head attention – why multiple subspaces help  
Instead of a single \((\mathbf{Q},\mathbf{K},\mathbf{V})\) triple, we split the model dimension into \(h\) heads:

\[
\operatorname{MultiHead}(\mathbf{X})=
\operatorname{Concat}\bigl(\text{head}_1,\dots,\text{head}_h\bigr)\mathbf{W}_O,
\qquad
\text{head}_i = \operatorname{Attention}
\bigl(\mathbf{X}\mathbf{W}_Q^{(i)},
      \mathbf{X}\mathbf{W}_K^{(i)},
      \mathbf{X}\mathbf{W}_V^{(i)}\bigr)
\]

Each head learns its own projection matrices \((\mathbf{W}_Q^{(i)},\mathbf{W}_K^{(i)},\mathbf{W}_V^{(i)})\) and can focus on a different relational pattern (e.g., syntax vs. semantics). The concatenated output is linearly mixed back to the original dimension with \(\mathbf{W}_O\).

---

### 4. Architecture of a Transformer Block  

#### 4.1. Positional encoding – injecting order without recurrence  
Attention is permutation‑invariant, so we add a deterministic or learned **positional vector** \(\mathbf{p}_i\) to each token embedding:

\[
\tilde{\mathbf{x}}_i = \mathbf{x}_i + \mathbf{p}_i
\]

The original sinusoidal scheme (Vaswani et al.) is:

\[
p_{i,2k}   = \sin\!\Bigl(\frac{i}{10000^{2k/d_{\text{model}}}}\Bigr),\qquad
p_{i,2k+1} = \cos\!\Bigl(\frac{i}{10000^{2k/d_{\text{model}}}}\Bigr)
\]

#### 4.2. Residual connections & layer normalization – stabilizing training  

\[
\text{LayerNorm}(\mathbf{x}) =
\frac{\mathbf{x}-\mu}{\sigma}\,\boldsymbol{\gamma} + \boldsymbol{\beta}
\]

A **residual** (skip) connection wraps each sub‑layer:

\[
\mathbf{y}= \text{LayerNorm}\bigl(\mathbf{x}+ \text{Sublayer}(\mathbf{x})\bigr)
\]

This preserves gradient flow and lets the network learn identity mappings when needed.

#### 4.3. Feed‑forward network (FFN) – point‑wise transformation  

\[
\text{FFN}(\mathbf{x}) = \max\bigl(0,\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1\bigr)\mathbf{W}_2 + \mathbf{b}_2
\]

The same two‑layer MLP is applied **independently** to each token. The hidden dimension \(d_{ff}\) is typically \(4\times d_{\text{model}}\).

#### 4.4. Stacking blocks – depth vs. width  

| Dimension | Effect |
|-----------|--------|
| **Depth** (more blocks) | Improves hierarchical representation; increases memory and compute. |
| **Width** (larger \(d_{\text{model}}\), more heads) | Boosts per‑layer capacity. |
| **Modern LLMs** | Favor deep stacks (e.g., 96 layers) with moderate width, thanks to efficient parallelism. |

---

### 5. Training the Model  

#### 5.1. Objectives  

| Model | Objective | Details |
|-------|-----------|---------|
| **BERT** | Masked Language Modeling (MLM) + Next‑Sentence Prediction (NSP) | Randomly mask 15 % of tokens; predict originals. NSP encourages discourse‑level reasoning. |
| **GPT family** | Autoregressive language modeling | Predict the next token given all previous ones. Causal masking ensures each position only attends to earlier tokens. |

The autoregressive loss is  

\[
\mathcal{L}= -\sum_{t=1}^{L}\log p\bigl(x_t \mid x_{<t}\bigr).
\]

#### 5.2. Optimization tricks  

| Trick | Purpose |
|-------|---------|
| **AdamW** | Decouples weight decay from Adam’s adaptive moments. |
| **Learning‑rate warm‑up + cosine decay** | Prevents divergence in early steps; provides a smooth schedule later. |
| **Gradient clipping** | Controls exploding gradients in very deep stacks. |

#### 5.3. Scaling laws  

Empirical work (Kaplan *et al.*, 2020) shows that the test loss \(L\) follows a power‑law in three resources:

\[
L(N, D, C) \approx
\left(\frac{N}{N_0}\right)^{-\alpha} +
\left(\frac{D}{D_0}\right)^{-\beta} +
\left(\frac{C}{C_0}\right)^{-\gamma},
\]

where  

* \(N\) = number of model parameters,  
* \(D\) = number of training tokens,  
* \(C\) = compute (FLOPs),  

and \(\alpha,\beta,\gamma>0\). The law predicts an *optimal* balance: under‑training wastes parameters, over‑training wastes compute.

---

### 6. From Theory to Practice  

#### 6.1. Minimal Transformer (pseudocode)  

```python
import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, vocab_sz, d_model=64, n_head=4,
                 d_ff=256, n_layers=2, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, d_model)
        self.pos   = nn.Parameter(
            self._sinusoidal_positional_encoding(max_len, d_model),
            requires_grad=False)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn":  nn.MultiheadAttention(d_model, n_head,
                                               batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "ffn":   nn.Sequential(
                            nn.Linear(d_model, d_ff),
                            nn.GELU(),
                            nn.Linear(d_ff, d_model)
                         ),
                "norm2": nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])

    def _sinusoidal_positional_encoding(self, max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1)
        i   = torch.arange(d_model).unsqueeze(0)
        angle = pos / (10000 ** (2 * (i // 2) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe

    def forward(self, ids, attn_mask=None):
        # (B, L) -> (B, L, d_model)
        x = self.embed(ids) + self.pos[:ids.size(1)]
        for layer in self.layers:
            # Self‑attention + residual
            attn_out, _ = layer["attn"](x, x, x, attn_mask=attn_mask)
            x = layer["norm1"](x + attn_out)

            # Feed‑forward + residual
            ff_out = layer["ffn"](x)
            x = layer["norm2"](x + ff_out)
        return x
```

Key points illustrated: embedding + positional encoding, multi‑head attention, residual‑norm pattern, and stacked layers.

#### 6.2. Common pitfalls & debugging tips  

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| **NaN loss after a few steps** | Softmax overflow (missing \(\sqrt{d_k}\) scaling) | Verify the divisor; use `torch.nn.functional.scaled_dot_product_attention`. |
| **Very slow convergence** | Learning rate too low or missing warm‑up | Adopt a warm‑up schedule (e.g., 10 k steps) followed by cosine decay. |
| **Model ignores token order** | Positional encodings not added or zero‑initialized | Check that `x + pos` is performed before the first layer. |
| **Spiking gradients** | Deep stack without clipping | Apply `torch.nn.utils.clip_grad_norm_`. |

#### 6.3. Extensions beyond text  

* **Vision Transformers (ViT)** – Split an image into patches, treat patches as tokens.  
* **Audio Transformers** – Frame‑level spectrogram tokens with learned positional encodings.  
* **Decision Transformers** – Cast reinforcement‑learning trajectories as sequences of (state, action, return) triples and predict the next action.  

---

### 7. Conclusion  

1. **Recap** – A Transformer consists of (1) token embeddings + positional encodings, (2) stacked self‑attention + feed‑forward blocks wrapped with residual‑norm scaffolding, and (3) a training objective (MLM, causal LM, etc.) that exploits massive parallelism.  
2. **First‑principles view** – Every component reduces to familiar linear‑algebra (dot