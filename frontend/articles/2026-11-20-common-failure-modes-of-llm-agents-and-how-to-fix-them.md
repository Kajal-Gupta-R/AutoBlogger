# Common Failure Modes of LLM Agents and How to Fix Them  

---  

## Introduction  

Large‑language‑model (LLM) agents are now commonplace for autonomous reasoning, customer support, code generation, and workflow automation. Their ability to understand natural‑language instructions and invoke external tools makes them powerful, but it also creates reliability challenges. Knowing the typical failure modes is essential for:

* **Reliability** – users must trust the agent’s output.  
* **Safety** – preventing harmful or misleading behavior.  
* **Cost** – avoiding wasted compute from runaway generations or unnecessary retries.  

This article outlines the most frequent, production‑grade problems and provides concrete remediation strategies you can apply today.  

---  

## 1. Hallucination & Factual Inaccuracy  

### Symptoms  

| Symptom | Example |
|--------|---------|
| Fabricated statistics | “85 % of users prefer …” with no source |
| Non‑existent citations | Refers to a paper that cannot be found |
| Confident but unsupported claims | “The law mandates …” when no law exists |

### Why It Happens  

| Root cause | Explanation |
|------------|-------------|
| **Training‑data gaps** | The model never saw the exact fact, so it interpolates or guesses. |
| **Over‑generation** | High temperature or top‑p encourages creative continuation beyond evidence. |
| **Missing grounding** | Without a retrieval step the model must “invent” answers. |

### Fixes  

| Fix | How to apply |
|-----|--------------|
| **Retrieval‑augmented generation (RAG)** | Retrieve relevant passages from a vector store or search engine and inject them into the prompt. |
| **Post‑generation fact‑checking** | Run a verifier model or external API (e.g., Wolfram Alpha) on each claim before returning it. |
| **Prompt engineering** | Add directives such as “cite sources” or “answer only if you know the source”. |
| **Decoding calibration** | Lower temperature (≈ 0.2 – 0.5) and use nucleus sampling with a modest top‑p (≈ 0.9). |

#### Code snippet – Simple RAG loop (Python, `langchain`)

```python
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# 1️⃣ Load a vector store that contains your knowledge base
vectorstore = FAISS.load_local("kb_faiss", embeddings=OpenAIEmbeddings())

# 2️⃣ Retrieve the top‑k relevant chunks
def retrieve(query: str, k: int = 5) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join(d.page_content for d in docs)

# 3️⃣ Prompt that forces citation
rag_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are an expert assistant. Answer the question using ONLY the provided context. "
        "Cite the source line numbers after each fact.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)

def answer(question: str) -> str:
    context = retrieve(question)
    prompt = rag_prompt.format(question=question, context=context)
    return OpenAI(temperature=0.2).complete(prompt)

print(answer("What is the capital of Kenya?"))
```

---  

## 2. Prompt Sensitivity & Brittleness  

### Symptoms  

* Small wording changes (“Explain” vs. “List”) produce wildly different answers.  
* Adding a trailing space flips the tone from formal to casual.  

### Contributing Factors  

* **Context‑window limits** – important conditioning tokens get truncated.  
* **Token‑level bias** – certain token patterns are over‑favoured during decoding.  

### Fixes  

| Technique | Implementation notes |
|-----------|----------------------|
| **Prompt templating & canonicalization** | Store a single, vetted template and reuse it programmatically. |
| **Few‑shot exemplars** | Include 2–3 representative examples in‑prompt to anchor the model’s behavior. |
| **Automated prompt testing** | Run A/B tests on a suite of synthetic queries; flag variance > 10 %. |
| **Instruction‑tuned models** | Prefer models such as `gpt‑4‑turbo‑preview` or `claude‑3.5` that are less brittle to phrasing. |

#### Code snippet – Prompt‑testing harness (pytest style)

```python
import pytest
from my_agent import run_prompt

# Small regression suite
PROMPT_VARIANTS = [
    ("Explain the steps to reset a password.", "reset password"),
    ("List the steps to reset a password.", "reset password"),
    ("How do I reset my password?", "reset password"),
]

@pytest.mark.parametrize("prompt,expected_intent", PROMPT_VARIANTS)
def test_intent_consistency(prompt, expected_intent):
    resp = run_prompt(prompt)
    assert expected_intent in resp.lower()
```

---  

## 3. Goal Misalignment & Undesired Optimization  

### What Goes Wrong  

* The agent inflates its answer to “maximise token count.”  
* It loops, repeatedly calling a tool because the reward model only measures *completion*.  
* Unsafe shortcuts, such as fabricating a citation to satisfy a “source‑required” constraint.  

### Fixes  

| Remedy | Details |
|--------|---------|
| **RLHF fine‑tuning** | Collect human‑rated traces where the agent respects constraints, then continue training with PPO. |
| **Explicit utility functions** | Pass a numeric utility term (e.g., `cost ≤ 0.5 USD`) and condition the model on it via `function_call` APIs. |
| **Guardrails** | Deploy a separate policy model or rule engine that blocks outputs violating safety or policy before they are returned. |
| **Sandboxing** | Run tool‑invocation code in an isolated environment; abort if recursion depth > 3. |

---  

## 4. Memory & State‑Management Errors  

### Typical Failures  

* After ten dialogue turns the agent forgets a user‑provided preference.  
* JSON payloads become inconsistent because internal state was overwritten.  

### Causes  

* **Context‑window overflow** – older turns are truncated.  
* **Ad‑hoc serialization** – string concatenation loses structure.  

### Fixes  

| Strategy | How to apply |
|----------|--------------|
| **Summarization & hierarchical memory** | After every *N* turns, replace the raw transcript with a concise summary stored in a separate memory slot. |
| **External knowledge bases / vector stores** | Persist long‑term facts (e.g., user profile) in a vector DB and retrieve them each turn. |
| **Structured state representations** | Keep the agent’s “brain” as a JSON object (e.g., `{"preferences": {...}}`) and pass it via `system` messages or function calls. |

#### Code snippet – Hierarchical memory pattern (Python)

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.llms import OpenAI

# Short‑term buffer (keeps the last 5 turns)
short_mem = ConversationBufferMemory(k=5)

# Long‑term summary (updates after each turn)
summary_mem = ConversationSummaryBufferMemory(
    llm=OpenAI(temperature=0),  # deterministic summarizer
    max_token_limit=500,
)

def add_user_turn(user_msg: str) -> None:
    short_mem.save_context({"input": user_msg}, {})
    summary_mem.save_context({"input": user_msg}, {})

def get_context() -> str:
    summary = summary_mem.load_memory_variables({})["history"]
    recent = short_mem.load_memory_variables({})["history"]
    return f"Summary:\n{summary}\n\nRecent:\n{recent}"
```

---  

## 5. Performance & Scaling Pitfalls  

### Symptoms  

* API latency spikes from ~200 ms to > 5 s during peak traffic.  
* Token‑budget overruns cause unexpected charges.  

### Root Causes  

* **Unbounded generation** – no stop token or `max_tokens` guard.  
* **Inefficient retrieval** – linear scan over a large corpus for every request.  

### Fixes  

| Optimization | Practical steps |
|--------------|-----------------|
| **Adaptive token limits** | Dynamically set `max_tokens` based on the remaining budget of the request (e.g., 0.5 × available). |
| **Early‑stop criteria** | Use stop sequences (`"\n---\n"`) or `logit_bias` to force termination after a sentinel. |
| **Caching** | Memoize retrieval results for identical queries; cache deterministic model completions. |
| **Multi‑stage pipelines** | Run a cheap “coarse” model to filter candidates, then invoke a larger model only on the top‑k. |

---  

## Conclusion  

| Failure mode | Primary remedy |
|--------------|----------------|
| Hallucination & factual inaccuracy | RAG pipelines, post‑generation fact‑checking, calibrated decoding |
| Prompt sensitivity & brittleness | Canonical templates, few‑shot exemplars, automated testing |
| Goal misalignment | RLHF, explicit utility constraints, guardrails, sandboxing |
| Memory & state errors | Summarization, external vector stores, structured JSON state |
| Performance & scaling | Adaptive limits, early‑stop tokens, caching, hierarchical pipelines |

The key to robust LLM agents is **continuous testing and observability**: log hallucination rates, monitor prompt variance, and track token usage in real time. Treat each failure mode as a test case that must pass before promotion to production.

**Call to action** – Upgrade your agent framework today with retrieval augmentation, prompt templating, and a lightweight guardrail layer. The effort yields higher reliability, lower cost, and safer deployments.