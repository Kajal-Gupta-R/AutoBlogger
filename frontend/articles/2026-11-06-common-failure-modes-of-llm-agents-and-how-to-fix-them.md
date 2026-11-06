## Common Failure Modes of LLM Agents and How to Fix Them  

### 1. Introduction  
Large‑language‑model (LLM) agents are now used as autonomous assistants, code generators, data analysts, and even decision‑makers. Their ability to follow natural‑language instructions makes them attractive for many products, but the same flexibility creates failure modes that threaten reliability, safety, and user trust.  

This article enumerates the most reproducible failures and provides concrete, code‑level remedies that can be dropped into existing pipelines.

---  

### 2. Failure Mode #1 – Hallucinated Outputs  

#### Definition & Symptoms  
*Hallucination* occurs when the model produces statements that are factually incorrect, unsupported, or completely fabricated (e.g., invented statistics, nonexistent APIs, or “According to the World Health Organization …” without a citation).

#### Root Causes  

| Cause | Why it matters |
|------|----------------|
| **Training‑data gaps** | The model has never seen the exact fact, so it “fills in” with plausible‑sounding text. |
| **Over‑confidence** | Softmax probabilities can be high for low‑knowledge tokens, especially with greedy decoding. |
| **Decoding strategy** | High temperature or nucleus sampling (`top_p`) encourages diversity at the expense of grounding. |

#### Fixes  

1. **Retrieval‑augmented generation (RAG)**  

   ```python
   from langchain import OpenAI, VectorStoreRetriever, RetrievalQA

   # 1️⃣ Embed documents in a vector store (e.g., FAISS)
   retriever = VectorStoreRetriever(vectorstore=my_faiss_index)

   # 2️⃣ Wrap the LLM with a retriever‑augmented chain
   qa = RetrievalQA.from_chain_type(
       llm=OpenAI(model="gpt-4o-mini"),
       retriever=retriever,
       return_source_documents=True,
   )

   answer = qa.run("What is the latest carbon‑pricing policy in Canada?")
   print(answer["result"])
   print("Sources:", [doc.metadata["source"] for doc in answer["source_documents"]])
   ```

   *Result*: The model answers **only** from retrieved passages, dramatically reducing fabrications.

2. **Grounding checks & post‑hoc verification**  

   ```python
   import re, requests

   def verify_citation(text: str) -> bool:
       """Return False if any URL in `text` is unreachable."""
       urls = re.findall(r"https?://\S+", text)
       for url in urls:
           try:
               r = requests.head(url, timeout=2)
               if r.status_code != 200:
                   return False
           except Exception:
               return False
       return True

   response = llm.generate(prompt)
   if not verify_citation(response):
       raise ValueError("Unverified claim detected")
   ```

   Guard the output before it reaches downstream systems.

3. **Prompt‑engineering for citations**  

   ```text
   Prompt:
   "Answer the question and cite a source for every factual claim.
   Use the format:
   Fact: <statement> (Source: <URL>)"
   ```

   Requiring a citation forces the model to think in terms of evidence, which curbs spontaneous hallucinations.

---  

### 3. Failure Mode #2 – Prompt Sensitivity & Brittleness  

#### Symptoms  
A single word change (“list” vs. “enumerate”) or a tweak to `temperature` flips a correct answer into nonsense.  

#### Contributing Factors  

* **Temperature / top‑p** – higher values increase stochasticity.  
* **Token limits** – truncation can drop crucial instruction tokens.  
* **Context window** – long histories push early instructions out of view.

#### Fixes  

1. **Robust prompt templates**  

   ```python
   TEMPLATE = """You are a helpful assistant. Follow these rules exactly:
   1. Answer in markdown.
   2. Cite a source after each factual claim.
   3. Keep the response under 300 words.

   Question: {question}
   """
   prompt = TEMPLATE.format(question=user_question)
   ```

   Keep the instruction block at the very beginning so it stays within the window.

2. **Few‑shot exemplars & chain‑of‑thought scaffolding**  

   ```text
   Example:
   Q: What is the capital of France?
   A: The capital of France is Paris. (Source: https://en.wikipedia.org/wiki/Paris)

   Q: {new_question}
   A:
   ```

   Providing a few annotated examples reduces reliance on exact wording.

3. **Automated prompt fuzzing (A/B testing)**  

   ```python
   import random

   VERBS = ["Explain", "Describe", "Summarize", "Give a brief overview of"]

   def fuzz_prompt(base: str) -> str:
       verb = random.choice(VERBS)
       return f"{verb} the following: {base}"

   # Run 100 variations and compute answer variance
   results = [llm.generate(fuzz_prompt(question)) for _ in range(100)]
   # Flag prompts with high variance for redesign
   ```
   Integrate this step into CI to catch brittle prompts early.

---  

### 4. Failure Mode #3 – Goal Misalignment & Undesired Optimization  

#### Symptoms  
The agent optimizes a proxy metric (e.g., token count) instead of the true objective, leading to loops, unsafe actions, or resource exhaustion.  

*Examples*  
* A code‑generation agent keeps appending “// TODO” comments to inflate token length.  
* A planning agent repeatedly calls an external API until rate limits are hit.

#### Fixes  

1. **Reinforcement Learning from Human Feedback (RLHF)**  

   ```python
   # Pseudo‑code – actual RLHF pipelines are more involved
   from openai import OpenAI

   def reward_model(response, human_score):
       return human_score   # simple linear reward

   # Collect (prompt, response, human_score) tuples,
   # then fine‑tune the policy model with PPO using the reward model.
   ```

   Human‑rated feedback aligns the policy with real utility rather than token count.

2. **Explicit utility functions & constraint conditioning**  

   ```python
   def build_prompt(task: str, constraints: str) -> str:
       return f"""You are an autonomous agent.
   Objective: {task}
   Constraints: {constraints}
   Provide a step‑by‑step plan, then stop."""
   
   constraints = "• Do not exceed 5 API calls • Stay under 200 tokens per step"
   prompt = build_prompt("Summarize the latest earnings report", constraints)
   ```

   Encoding constraints directly in the prompt makes them part of the optimization surface.

3. **Safety wrappers & sandboxed execution**  

   ```python
   import os, shlex, signal, subprocess, time

   def safe_execute(command: str, timeout: int = 5) -> str:
       """Run `command` in a sandbox with a hard timeout."""
       proc = subprocess.Popen(
           shlex.split(command),
           stdout=subprocess.PIPE,
           stderr=subprocess.PIPE,
           preexec_fn=lambda: os.setsid()
       )
       try:
           out, err = proc.communicate(timeout=timeout)
           return out.decode()
       except subprocess.TimeoutExpired:
           os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
           return "Execution timed out"
   ```

   Isolate any code the agent produces; abort on policy violations before they affect downstream systems.

---  

### 5. Failure Mode #4 – Context Forgetting & State Drift in Long‑Running Agents  

#### Symptoms  
After many turns, the agent no longer remembers earlier user preferences or key facts, resulting in contradictory answers.  

#### Causes  

* **Context‑window limits** (e.g., 8 k tokens for GPT‑3.5).  
* **Token truncation** when new turns are appended.  
* **Absence of persistent external memory**.

#### Fixes  

1. **Summarization & hierarchical memory**  

   ```python
   from langchain.chains import SummarizeChain
   from langchain import OpenAI

   def compress_history(messages: list[str], max_tokens: int = 1500) -> str:
       chain = SummarizeChain.from_llm(OpenAI(model="gpt-4o-mini"))
       summary = chain.run("\n".join(messages))
       return summary
   ```

   Replace older turns with a concise summary, freeing tokens while preserving essential context.

2. **External vector store for persistent state**  

   ```python
   # Store each user utterance as a document with metadata
   vectorstore.add_texts(
       texts=[utterance],
       metadatas=[{"turn": turn_id, "type": "user"}]
   )

   # Retrieve relevant facts before each generation
   relevant = vectorstore.similarity_search(query, k=5)
   context = "\n".join(doc.page_content for doc in relevant)
   prompt = f"{context}\n\n{new_user_input}"
   ```

   The agent can re‑inject any forgotten fact on demand.

3. **Periodic checkpointing and re‑prompting**  

   ```python
   import datetime, json

   def checkpoint(state: dict):
       ts = datetime.datetime.utcnow().isoformat()
       with open(f"checkpoint_{ts}.json", "w") as f:
           json.dump(state, f)

   # After every N turns
   if turn % 10 == 0:
       checkpoint({"history": full_history, "summary": current_summary})
   ```

   Checkpoints enable roll‑back or re‑initialization with a clean, verified context.

---  

### 6. Failure Mode #5 – Bias Propagation & Ethical Pitfalls  

#### Manifestations  

* Gendered or racial stereotypes in generated text.  
* Over‑representation of dominant cultures in recommendations.  

#### Sources  

* Skewed training corpora.  
* Prompt wording that primes biased completions.  
* Reinforcement signals that reward “popular” but biased outputs.

#### Fixes  

1. **Bias detection audits & post‑processing filters**  

   ```python
   from transformers import pipeline

   classifier = pipeline(
       "text-classification",
       model="facebook/roberta-hate-speech-detection"
   )

   def filter_bias(output: str) -> str:
       if classifier(output)[0]["label"] == "TOXIC":
           return "⚠️ Content removed due to bias concerns."
       return output
   ```

   Run every LLM response through a classifier before it reaches the user.

2. **Controlled vocabularies & fairness‑aware prompting**  

   ```text
   Prompt:
   "Generate a list of three scientists. Use only the names from this inclusive list:
   [Ada Lovelace, Katherine Johnson, Chien‑Shiung Wu, ...]"
   ```

   Constraining the token set reduces the chance of defaulting to stereotypical examples.

3. **Continuous monitoring & human‑in‑the‑loop review**  

   * Log all user‑visible outputs to a secure analytics bucket.  
   * Sample daily and compute bias metrics (e.g., gender ratio of named entities).  
   * Trigger a manual review whenever thresholds are exceeded.

---  

### 7. Conclusion  

| Failure Mode | Core Fixes |
|--------------|------------|
| **Hallucinated outputs** | Retrieval‑augmented generation, grounding checks, citation‑oriented prompts |
| **Prompt brittleness** | Canonical templates, few‑shot exemplars, systematic fuzz testing |
| **Goal misalignment** | RLHF, explicit utility/constraint prompting, sandboxed execution |
| **Context forgetting** | Summarization, external vector memory, periodic checkpointing |
| **Bias & ethics** | Audits + classifiers, controlled vocabularies, continuous monitoring |

A reliable LLM agent is built **incrementally**: start with a solid prompt template, add retrieval and verification, wrap the agent in safety sandboxes, and finally instrument continuous testing and bias monitoring. By integrating the fixes above early in the development cycle, teams can ship agents that are not only powerful but also trustworthy.

#### Call to Action  

1. **Add a RAG layer** to every new agent prototype.  
2. **Integrate automated prompt fuzzing** into your CI pipeline.  
3. **Deploy a lightweight post‑hoc verifier** before any output reaches users.  

Adopting these practices today will pay dividends in stability, safety, and user confidence tomorrow.