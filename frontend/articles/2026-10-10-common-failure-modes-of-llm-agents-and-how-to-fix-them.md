## Common Failure Modes of LLM Agents and How to Fix Them  

---  

### 1. Introduction  

Large Language Model (LLM) agents are autonomous—or semi‑autonomous—software components that receive a natural‑language prompt, perform reasoning (often over external tools or data), and return a response. Typical use‑cases include:

- Conversational assistants  
- Code‑generation copilots  
- Retrieval‑augmented question answering  
- Automated workflow orchestration (e.g., ticket triage)  

Reliability isn’t a nice‑to‑have feature; it directly influences user trust, safety compliance, and the bottom line. A single hallucinated fact or a missed constraint can cause misinformation, legal exposure, or lost revenue.  

**Goal of this post** – enumerate the most common failure modes observed in production LLM agents and provide concrete, implementable mitigation strategies.  

---  

### 2. Failure Mode #1 – Hallucination & Fabricated Information  

#### What it looks like  

- “The Eiffel Tower was built in **1885**” (invented date)  
- Citations to non‑existent papers or URLs  

#### Root causes  

| Cause | Why it matters |
|-------|----------------|
| **Training‑data noise** | The model learns spurious correlations and treats them as facts. |
| **Over‑confidence in decoding** | Greedy or high‑temperature sampling prefers plausible‑sounding tokens over “I don’t know”. |
| **Lack of grounding** | The model never checks its output against an authoritative source. |

#### Fixes  

1. **Retrieval‑augmented generation (RAG) & grounding checks**  
   - Retrieve relevant passages *before* generation and force the model to cite them.  
   - After generation, run a verification step that cross‑checks any claim against the retrieved corpus.  

2. **Post‑generation fact‑checking pipelines**  
   - Call external APIs (e.g., Wolfram Alpha, Crossref) or an internal knowledge‑base to validate named entities, dates, or numbers.  
   - If verification fails, either regenerate with an explicit “please cite sources” instruction or return a fallback message.  

3. **Prompt‑engineering tricks**  
   - Prefix: `You are a factual assistant. Only answer if you are 100 % certain; otherwise say "I don't know."`  
   - Add a “self‑critique” instruction: `After you answer, briefly list any assumptions you made.`  

#### Code snippet – Simple RAG loop (Python + LangChain)

```python
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# 1️⃣ Load a vector store that contains domain documents
vector_store = FAISS.load_local(
    "faiss_index", embedding_function="text-embedding-ada-002"
)

# 2️⃣ Retrieve the top‑k most similar chunks
def retrieve(query: str, k: int = 5) -> str:
    docs = vector_store.similarity_search(query, k=k)
    return "\n".join(doc.page_content for doc in docs)

# 3️⃣ Prompt that forces grounding
template = """You are a factual assistant. Use **only** the following excerpts to answer the question.
If you cannot answer with certainty, say "I don't know."

Excerpts:
{context}

Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

def answer(question: str) -> str:
    context = retrieve(question)
    llm = OpenAI(temperature=0)          # deterministic for factual QA
    return llm(prompt.format(context=context, question=question))

print(answer("When was the first iPhone released?"))
```

*Key points*: deterministic decoding (`temperature=0`), explicit grounding, and a fallback response.  

---  

### 3. Failure Mode #2 – Prompt Sensitivity & Inconsistent Outputs  

#### Symptom  

Re‑phrasing a request—e.g., “What are the risks?” → “List the risks.”—yields completely different enumerations.  

#### Why it happens  

- Token‑level stochasticity (temperature, top‑p) amplifies tiny wording changes.  
- The model does not maintain a persistent reasoning chain across calls.  

#### Fixes  

| Fix | How to apply |
|-----|--------------|
| **Chain‑of‑thought (CoT) prompting** | Ask the model to “think step‑by‑step” before answering. |
| **Self‑consistency sampling** | Generate *n* CoT samples, then pick the most frequent final answer. |
| **Few‑shot exemplars** | Provide 2–3 examples that encode the desired reasoning pattern. |
| **Temperature/seed control** | Set `temperature=0` (or a low value) and fix the random seed for critical paths. |
| **Deterministic decoding** | Use beam search with a fixed number of beams (`num_beams=4`). |

#### Example – Self‑consistency wrapper

```python
def self_consistent_answer(question: str, n: int = 5) -> str:
    llm = OpenAI(temperature=0.7, seed=42)
    # Generate n CoT samples
    samples = [
        llm(f"Q: {question}\nA: Let's think step by step.")
        for _ in range(n)
    ]
    # Extract the final answer from each sample
    finals = [s.split("\n")[-1].strip() for s in samples]
    # Return the most common answer
    return max(set(finals), key=finals.count)

print(self_consistent_answer("What is the capital of Canada?"))
```

---  

### 4. Failure Mode #3 – Goal Misalignment & Undesired Behaviors  

#### Examples  

- Ignoring a “do not mention politics” constraint.  
- Optimizing for token length (“short answer”) while the user asked for a detailed report.  

#### Sources  

- Ambiguous system prompt or missing constraints.  
- Reward model that over‑weights proxy metrics (e.g., brevity).  

#### Fixes  

1. **Hierarchical prompting**  
   - **System prompt** sets high‑level policy.  
   - **User prompt** conveys the concrete task.  
   - **Assistant (internal) prompt** reinforces constraints before generation.  

2. **RLHF loops**  
   - Collect human feedback on alignment failures.  
   - Fine‑tune a reward model and run PPO to bias the base LLM toward compliant outputs.  

3. **Runtime safety filters**  
   - Plug a classifier (e.g., OpenAI Moderation API) that blocks or rewrites disallowed content before it reaches the user.  

#### Prompt hierarchy example  

```
System: You are a helpful assistant. NEVER provide political opinions. ALWAYS cite sources when you state a fact.
User: Summarize the latest research on quantum error correction.
Assistant (internal): Follow system rules. If a source is unavailable, say "I don't have a reliable source."
```

---  

### 5. Failure Mode #4 – Context‑Window Overrun & Forgetting Important Details  

#### Problem  

Long chat histories or multi‑page documents exceed the model’s context window (e.g., 8 k tokens), causing earlier constraints or facts to be dropped.  

#### Causes  

- Fixed‑size context windows.  
- Naïve “keep the last N tokens” sliding window that discards earlier constraints.  

#### Fixes  

| Strategy | Implementation tip |
|----------|--------------------|
| **Summarization & hierarchical memory** | After each turn, summarise key facts into a short “memory prompt” that is always prepended. |
| **On‑demand retrieval** | Store the full transcript in a vector store; retrieve only the most relevant chunks for the current query. |
| **Adaptive chunking** | Split long inputs into semantic chunks, embed them, and fetch those with highest similarity to the current query. |
| **Memory prompts** | Keep a persistent “facts block” (e.g., `Facts: user is a data‑engineer, prefers Python examples.`) and prepend it to every call. |

#### Code snippet – Memory‑prompt pattern

```python
memory = []                     # list of dicts: {"role": "...", "content": "..."}
MAX_MEMORY_TURNS = 10           # keep at most 10 recent turns

def add_to_memory(role: str, content: str) -> None:
    memory.append({"role": role, "content": content})
    if len(memory) > MAX_MEMORY_TURNS:
        memory.pop(0)            # drop the oldest turn

def retrieve(query: str) -> str:
    """Placeholder for a RAG or vector‑store lookup."""
    # In practice, query a FAISS/Chroma index here.
    return ""

def chat(user_msg: str) -> str:
    # 1️⃣ Retrieve relevant external context (optional)
    external_context = retrieve(user_msg)

    # 2️⃣ Build the prompt
    prompt = [
        {"role": "system", "content": "You are a factual assistant. Never hallucinate."},
        *memory,
        {"role": "user", "content": f"{external_context}\n\nQuestion: {user_msg}"}
    ]

    # 3️⃣ Call the LLM
    response = OpenAIChat().chat_completion(prompt).choices[0].message["content"]

    # 4️⃣ Update memory
    add_to_memory("assistant", response)
    return response
```

---  

### 6. (Optional) Failure Mode #5 – Resource Exhaustion & Latency Spikes  

#### Symptoms  

- API timeouts during peak load.  
- Unexpected cost spikes in production.  

#### Contributors  

- Using the largest available model for every request.  
- Re‑generating the same answer for identical queries.  
- Serial processing of independent requests.  

#### Fixes  

1. **Model distillation / quantisation**  
   - Deploy a 3‑bit (or 4‑bit) quantised version for low‑latency paths; fall back to the full‑size model only for complex tasks.  

2. **Caching & token‑level memoisation**  
   - Cache embeddings and LLM outputs keyed by a hash of the normalised prompt.  
   - Re‑use the same completion for identical queries within a short TTL.  

3. **Asynchronous pipelines & load‑balancing**  
   - Use a task queue (e.g., Celery, RabbitMQ) to batch similar requests.  
   - Autoscale model workers based on latency metrics.  

#### Cache example (Python + Redis)

```python
import hashlib, json, redis
from openai import ChatCompletion

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def cached_chat(messages: list[dict]) -> str:
    # Create a deterministic key from the message list
    key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    cached = redis_client.get(key)
    if cached:
        return cached.decode()

    resp = ChatCompletion.create(model="gpt-4o-mini", messages=messages)
    answer = resp.choices[0].message["content"]
    redis_client.setex(key, 300, answer)   # 5‑minute TTL
    return answer
```

---  

### 7. Conclusion  

| Failure Mode | Core Remedy |
|--------------|--------------|
| Hallucination | Retrieval‑augmented generation + grounding + post‑generation fact‑checking |
| Prompt Sensitivity | Chain‑of‑thought, self‑consistency, deterministic decoding |
| Goal Misalignment | Hierarchical prompts, RLHF, runtime safety filters |
| Context Overrun | Summarisation, hierarchical memory, on‑demand retrieval |
| Resource Exhaustion (optional) | Distilled/quantised models, caching, asynchronous load‑balancing |

A **defence‑in‑depth** approach works best: start with robust prompt design, layer architectural safeguards (retrieval, memory, safety filters), and finish with continuous monitoring + human‑in‑the‑loop feedback.  

**Call to action** – audit the agents you own today: identify which of the above modes appear, apply the corresponding fixes, and share your findings with the community. A collective effort will raise the reliability bar for every LLM‑powered product.  