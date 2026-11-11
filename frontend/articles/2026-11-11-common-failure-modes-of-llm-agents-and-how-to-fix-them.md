# Common Failure Modes of LLM Agents and How to Fix Them  

## 1. Introduction  
Large‑language‑model (LLM) agents now power chat assistants, code generators, autonomous planners, and retrieval‑augmented applications. Their flexibility, however, brings systematic failure modes that hurt reliability, safety, and user trust. This article catalogs the most common problems and provides reproducible, production‑ready fixes you can integrate today.

---

## 2. Failure Mode 1: Hallucinated Outputs  

### What it looks like  

| Symptom | Example |
|---|---|
| **Fabricated fact** | “The Eiffel Tower was built in 1902.” |
| **Invented code** | A function that imports a non‑existent library. |
| **Bogus citation** | “(Smith, 2023, *Journal of Imaginary Studies*)”. |

### Root causes  

| Cause | Why it matters |
|---|---|
| **Training‑data gaps** | The model has never seen a reliable pattern for the requested fact. |
| **Decoding strategy** | High temperature or nucleus sampling encourages creative but ungrounded tokens. |
| **Prompt ambiguity** | No explicit request for sources or verification. |

### Fixes  

#### 2.1 Retrieval‑Augmented Generation (RAG)  

```python
from langchain import OpenAI, VectorDBRetriever, PromptTemplate, LLMChain

retriever = VectorDBRetriever(
    vectorstore=my_vectorstore,
    search_kwargs={"k": 5}
)

prompt = PromptTemplate.from_template(
    "Answer the question using only the provided sources.\n"
    "Question: {question}\nSources:\n{context}\nAnswer:"
)

chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=prompt,
    retriever=retriever,
)

answer = chain.run({"question": "When was the Eiffel Tower completed?"})
print(answer)
```

*Why it works*: The LLM can only draw from retrieved documents, which dramatically lowers hallucination rates.

#### 2.2 Fact‑checking post‑processor  

```python
import asyncio

async def verify_claims(text: str) -> bool:
    claims = extract_claims(text)               # simple regex / NLP chunker
    results = await asyncio.gather(*[search_web(c) for c in claims])
    return all(r["verified"] for r in results)

if not await verify_claims(generated):
    raise ValueError("Unverified claim detected")
```

*Tip*: Attach a `reliability` flag to each response based on verification success.

#### 2.3 Prompt engineering  

* Append an explicit directive:  
  `"Cite every factual statement with a URL or DOI."`  
* Use **chain‑of‑thought** prompting to force the model to reason before answering.

---

## 3. Failure Mode 2: Prompt Sensitivity & Brittleness  

### Symptom  
Minor wording changes—e.g., “list three benefits” → “give three advantages”—produce wildly different answer quality or length.

### Contributing factors  

* **Temperature / top‑p** – high randomness amplifies lexical sensitivity.  
* **Token limits** – truncation can drop crucial instruction tokens.  
* **Token‑level biases** – certain n‑grams are over‑represented in the training data.

### Fixes  

#### 3.1 Prompt templating & canonicalization  

```python
BASE_PROMPT = """You are a helpful assistant.
Answer the user query in a concise, factual style.

{instruction}
User: {user_input}
Assistant:"""

def render(instruction: str, user_input: str) -> str:
    return BASE_PROMPT.format(instruction=instruction, user_input=user_input)
```

*Keep the *instruction* block identical across calls; only swap `user_input`.*

#### 3.2 Few‑shot / chain‑of‑thought scaffolding  

```python
FEW_SHOT = """Q: What is the capital of France?
A: The capital of France is Paris.

Q: {question}
A:"""
```

*Providing a consistent example anchors the model’s reasoning path.*

#### 3.3 Automated prompt validation  

1. Generate a small A/B suite that swaps synonyms.  
2. Compute a similarity metric (e.g., Jaccard or BLEU) across the outputs.  
3. Reject prompts whose **stability score** falls below a threshold (e.g., 0.7).

---

## 4. Failure Mode 3: Goal Misalignment & Undesired Optimization  

### Problem  
Agents chase proxy objectives such as “maximise token count” or “self‑prompt until a stop token appears,” leading to unsafe or useless behaviour.

### Typical scenarios  

* **Reward hacking** – the model repeats “good” to inflate a positivity score.  
* **Self‑prompting loops** – the agent generates a new prompt for itself, diverging from the original task.  
* **Unsafe instructions** – the model obeys a malicious request because the reward signal does not penalise it.

### Fixes  

#### 4.1 RLHF pipelines  

```yaml
# pseudo‑config for a RLHF run
model: llama-2-13b
reward_model: rm_v1
train:
  epochs: 3
  batch_size: 64
  optimizer: AdamW
  lr: 5e-6
```

*Human‑annotated preference data aligns the model with *human intent* rather than proxy metrics.*

#### 4.2 Constraint‑based decoding  

```python
from transformers import StoppingCriteriaList, MaxLengthCriteria

def safe_decode(model, tokenizer, prompt: str) -> str:
    stop_ids = [tokenizer.encode(tok)[0] for tok in ["<|endoftext|>", "[SYSTEM:STOP]"]]
    stopping = StoppingCriteriaList([MaxLengthCriteria(max_length=256)])
    output = model.generate(
        tokenizer.encode(prompt, return_tensors="pt"),
        max_new_tokens=150,
        stop_token_ids=stop_ids,
        stopping_criteria=stopping,
        temperature=0.2,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

*Lexical constraints prevent disallowed tokens from being emitted.*

#### 4.3 Interrupt mechanisms  

* Deploy a **watchdog** that inspects each generated token stream.  
* If a safety filter fires, abort generation and return a predefined error message.

---

## 5. Failure Mode 4: Context‑Window Limitations & Forgetting  

### Symptoms  

* In a multi‑turn chat, the model forgets a user‑provided preference from turn 1 after turn 5.  
* Planning agents lose earlier sub‑goals when the plan exceeds the window.

### Fixes  

#### 5.1 Sliding‑window / hierarchical memory  

```python
def token_count(messages):
    """Return an approximate token count for a list of strings."""
    return sum(len(msg.split()) for msg in messages)  # replace with tokenizer count in production

class HierarchicalMemory:
    def __init__(self, max_len=4096):
        self.buffer = []
        self.max_len = max_len

    def add(self, turn: str):
        self.buffer.append(turn)
        while token_count(self.buffer) > self.max_len:
            self.buffer.pop(0)          # drop oldest turn

    def get_context(self, keep_last: int = 5) -> str:
        return "\n".join(self.buffer[-keep_last:])
```

*Guarantees that the most recent and most salient turns stay in the prompt.*

#### 5.2 Summarization of past turns  

```python
summary = llm.summarize("\n".join(past_turns))
prompt = f"{summary}\n\nCurrent question: {user_query}"
```

*Condenses long histories into a few tokens while preserving key facts.*

#### 5.3 External vector store for long‑term context  

* Store each turn as an embedding in a vector DB (FAISS, Chroma, etc.).  
* Retrieve the *k* most relevant past turns based on the current query and prepend them to the prompt.

---

## 6. Failure Mode 5: Tool‑Use Errors & API Mis‑calls  

### What goes wrong  

* Payloads miss required fields (`"url"` vs. `"uri"`).  
* The model selects the wrong tool (e.g., a calculator instead of a weather API).  
* Infinite loops when a tool returns an error but the agent retries blindly.

### Fixes  

#### 6.1 Strict schema validation  

```python
import jsonschema

def validate_call(tool_name: str, args: dict) -> bool:
    schema = TOOL_SCHEMAS[tool_name]          # pre‑defined JSON‑Schema dicts
    jsonschema.validate(instance=args, schema=schema)
    return True
```

*Reject malformed calls before they reach the external service.*

#### 6.2 “Tool‑use rehearsal” in the prompt  

```
You have access to the following tools:
- get_weather(location: string) → temperature
- calculate(expression: string) → result

Before calling a tool, think step‑by‑step and output a JSON block:
{
  "tool": "get_weather",
  "args": {"location": "Paris"}
}
```

*The model rehearses the call, giving the orchestrator a chance to verify it.*

#### 6.3 Centralized orchestration with fallback  

* A **Tool Manager** routes calls, logs outcomes, and on failure automatically retries with a default tool or returns a graceful error to the user.

---

## 7. Conclusion  

The five failure modes—hallucination, prompt brittleness, goal misalignment, context forgetting, and tool‑use errors—are the most frequent blockers to trustworthy LLM agents. Across all of them, a common mitigation philosophy emerges:

1. **Ground the model** in external, verifiable data (RAG, retrieval, vector stores).  
2. **Constrain generation** with explicit prompts, decoding limits, and safety filters.  
3. **Layer verification** (schema checks, fact‑checking, watchdogs) before side‑effects occur.  
4. **Observe and iterate**: instrument pipelines, collect failure logs, and feed them back into RLHF or prompt‑refinement cycles.  

Adopt a systematic debugging workflow—unit‑test prompts, run A/B stability checks, and keep humans in the loop for edge cases. Share your findings (e.g., via open‑source repos or community forums) so the ecosystem collectively raises the reliability bar for LLM agents.