# Transformers Explained from First Principles  

## Introduction  

- **Motivation** – Understanding the fundamentals lets you reason about model capacity, diagnose failures, and innovate beyond “black‑box” usage.  
- **Brief history** – Early sequence models used recurrent neural networks (RNNs) or convolutional networks (CNNs). Their sequential nature limited parallelism. The 2017 *Attention Is All You Need* paper replaced recurrence with self‑attention, unlocking unprecedented speed and scale.  
- **Roadmap** – This article proceeds as follows:  

  1. Foundations of sequence modeling  
  2. The attention mechanism  
  3. Encoder–decoder architecture  
  4. Training objectives and optimization tricks  
  5. From theory to practice – a minimal implementation  

---

## 1. Foundations of Sequence Modeling  

### 1.1. Variable‑length inputs  
Natural language, DNA, time series, and many other data types have no fixed length. A model must ingest a sequence  

\[
x = (x_1,\dots ,x_T)
\]

where the length \(T\) varies across examples.

### 1.2. Limitations of recurrent and convolutional approaches  

| Approach | Strength | Core limitation |
|----------|----------|-----------------|
| RNN / LSTM / GRU | Theoretically captures arbitrarily long dependencies | Inherently sequential → poor GPU utilization; gradients may vanish or explode |
| 1‑D CNN | Parallel convolutions, fixed receptive field | Needs many layers or dilations to see distant tokens; still biased toward locality |

### 1.3. Attention as a solution  
*Attention* lets every token directly “look at” every other token, weighting contributions by relevance. Because all pairwise interactions are computed in parallel, attention provides global context without the sequential bottleneck of RNNs.

---

## 2. The Attention Mechanism  

### 2.1. Scaled dot‑product attention  

For queries **Q**, keys **K**, and values **V**:

\[
\text{Attention}(Q,K,V)=\operatorname{softmax}\!\Bigl(\frac{QK^{\top}}{\sqrt{d_k}}\Bigr)V
\]

\(d_k\) is the dimensionality of the keys; the \(\sqrt{d_k}\) scaling prevents the softmax from saturating.

### 2.2. Queries, keys, and values – an analogy  
- **Key** – a “topic tag” attached to each token (what the token *offers*).  
- **Query** – a “question” a token asks of the sequence (what it *needs*).  
- **Value** – the actual information transmitted when a key matches a query.  

Think of a library: each book (token) has a subject label (key). When you search (query), the system returns the book’s content (value) weighted by how well the label matches the query.

### 2.3. Multi‑head attention  

- Split the embedding dimension \(d_{\text{model}}\) into \(h\) subspaces (\(d_k = d_v = d_{\text{model}}/h\)).  
- Each head learns a distinct relational pattern (e.g., syntax vs. semantics).  
- Concatenate the heads and apply a final linear projection back to \(d_{\text{model}}\).

### 2.4. Complexity and parallelism  

- **Time‑complexity** per layer: \(O(T^{2}\,d_{\text{model}})\) (quadratic in sequence length).  
- **Parallelism** – All matrix multiplications are batched; GPUs compute the full \(QK^{\top}\) matrix in a single pass, unlike the step‑wise recurrence of RNNs.

---

## 3. Architecture of the Transformer Encoder–Decoder  

### 3.1. Encoder layer  

1. **Multi‑head self‑attention** – each token attends to all others.  
2. **Add & Norm** – residual connection followed by layer normalization.  
3. **Position‑wise feed‑forward network (FFN)** – two linear layers with a ReLU or GELU non‑linearity.  
4. **Add & Norm** again.  

### 3.2. Decoder layer  

1. **Masked multi‑head self‑attention** – prevents a position from attending to future tokens (causal mask).  
2. **Add & Norm**.  
3. **Encoder‑decoder attention** – queries from the decoder attend to the encoder’s keys/values.  
4. **Add & Norm**.  
5. **FFN** + **Add & Norm**.  

### 3.3. Positional encodings – injecting order  

Attention is permutation‑invariant, so we add a positional vector \(p_i\) to each token embedding \(x_i\).

*Sinusoidal encoding* (Vaswani et al., 2017):

\[
p_{i,2k}= \sin\!\Bigl(\frac{i}{10000^{2k/d_{\text{model}}}}\Bigr),\qquad
p_{i,2k+1}= \cos\!\Bigl(\frac{i}{10000^{2k/d_{\text{model}}}}\Bigr)
\]

Learned positional embeddings are also common in modern variants.

### 3.4. Residual connections and layer normalization  

- **Residuals** (\(x + \text{Sublayer}(x)\)) preserve gradient flow across many layers.  
- **LayerNorm** stabilizes training by normalizing across the feature dimension after each sub‑layer.

---

## 4. Training Objectives and Optimization  

### 4.1. Teacher forcing  

During training the decoder receives the **ground‑truth** previous token \(y_{t-1}\) instead of its own prediction. This yields a well‑behaved loss surface and speeds convergence.

### 4.2. Cross‑entropy loss  

For each position \(t\):

\[
\mathcal{L}_t = -\log p_{\theta}\bigl(y_t \mid y_{<t}, X\bigr)
\]

The total loss is the mean (or sum) of \(\mathcal{L}_t\) over all positions and batch elements.

### 4.3. Regularization  

| Technique | Purpose |
|-----------|---------|
| **Dropout** (applied to attention weights and FFN) | Prevents co‑adaptation, improves robustness |
| **Label smoothing** (e.g., \(\epsilon=0.1\)) | Penalises over‑confident predictions, yields better calibration |
| **Weight decay (L2)** | Controls weight magnitude, aids generalisation |

### 4.4. Scaling laws and data efficiency  

Empirical studies show loss \(\approx A\,N^{-\alpha}\) where \(N\) is model size or token count. Practical take‑aways:  

- Doubling parameters yields diminishing returns unless the dataset size also grows.  
- Pre‑training on massive corpora and then fine‑tuning is far more data‑efficient than training from scratch.

---

## 5. From Theory to Practice: Building a Minimal Transformer  

### 5.1. Scaled‑dot‑product attention (PyTorch‑style pseudocode)

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot‑product attention."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # (B, h, T_q, T_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)                     # (B, h, T_q, T_k)
    attn_weights = torch.dropout(attn_weights, p=0.1, train=Q.requires_grad)

    return torch.matmul(attn_weights, V)                             # (B, h, T_q, d_v)
```

### 5.2. Minimal Transformer in PyTorch (high‑level skeleton)

```python
class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 512,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed + position
        src = self.tok_emb(src) + self.pos_emb(src)
        for layer in self.encoder:
            src = layer(src, src_mask)

        tgt = self.tok_emb(tgt) + self.pos_emb(tgt)
        for layer in self.decoder:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return self.fc_out(tgt)          # logits over the vocabulary
```

*`EncoderLayer`* and *`DecoderLayer`* implement the “Add & Norm → Multi‑head (self)‑attention → Add & Norm → FFN → Add & Norm” pattern described in Section 3.

### 5.3. Debugging checklist  

| Symptom | Likely cause | Remedy |
|---------|--------------|--------|
| NaN loss after a few steps | Missing \(\sqrt{d_k}\) scaling → softmax overflow | Add the scaling factor or use `torch.nn.functional.softmax` after scaling |
| Model ignores token order | Positional encodings not added or zero‑initialized | Verify that `pos_emb` returns non‑zero vectors and that broadcasting is correct |
| Gradient explosion early in training | Learning rate too high or missing residual connections | Lower the learning rate; ensure each sub‑layer is wrapped with `x + sublayer(x)` |
| Decoder sees future tokens | Mask shape mismatch or mask not applied | `tgt_mask` must be a lower‑triangular matrix of shape `(T, T)` with 1s on and below the diagonal |

### 5.4. Extending the base model  

| Extension | Core modification |
|-----------|-------------------|
| **BERT** (bidirectional encoder) | Drop the decoder, use masked language‑modeling and next‑sentence‑prediction objectives; optionally add segment embeddings |
| **GPT** (decoder‑only) | Keep only masked self‑attention stacks, train with causal language modelling loss |
| **Vision Transformer (ViT)** | Replace token embeddings with patch embeddings from images; keep positional encodings and the encoder stack |

---

## Conclusion  

1. **Recap** – We began with the fundamental problem of variable‑length sequence modeling, identified attention as the key abstraction, derived the scaled dot‑product formula, and assembled the full encoder–decoder stack (positional encodings, residuals, layer norm). We then covered the standard training pipeline (teacher forcing, cross‑entropy, regularization) and presented a compact implementation.  
2. **First‑principles perspective** – By tracing every architectural decision back to a concrete limitation (sequential bottleneck, lack of global context), the Transformer becomes a logical solution rather than a mysterious black box. This view demystifies later variants such as BERT, GPT, and ViT, which are merely re‑combinations of the same primitives.  
3. **Next steps**  

   1. **Experiment** – Train `MiniTransformer` on a toy corpus (e.g., WikiText‑2) and observe how loss scales with depth and data size.  
   2. **Read deeper** – *Attention Is All You Need* (Vaswani et al., 2017), *BERT: Pre‑training of Deep Bidirectional Transformers* (Devlin et al., 2019), *Scaling Laws for Neural Language Models* (Kaplan et al., 2020).  
   3. **Apply** – Extend the code to a decoder‑only language model, add relative positional encodings, or replace the FFN with a lightweight convolutional block for a hybrid architecture.  

Armed with these fundamentals, you can move beyond using pre‑trained APIs and start engineering your own Transformer‑based solutions. Happy modeling!