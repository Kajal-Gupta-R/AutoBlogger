## Fine‑Tuning vs. Prompt Engineering: When to Use Which  

---  

### Introduction  

* **LLM customization** hinges on two levers:  
  1. **Fine‑tuning** – modify the model’s weights.  
  2. **Prompt engineering** – shape the input text that drives the model.  

* The choice influences **cost, latency, maintainability, and domain‑specific performance**.  

* **Goal** – give engineers a concrete decision framework and a set of best‑practice tips.  

---  

## 1. Foundations of Fine‑Tuning  

| Aspect | Key Points |
|--------|------------|
| **Definition & workflow** | Retrain a pre‑trained LLM on a downstream dataset so its parameters adapt to the new task. Typical pipeline: <br>1. Data preparation <br>2. Train/validation split <br>3. Trainer configuration <br>4. Training <br>5. Evaluation <br>6. Deployment |
| **Data requirements** | • **Size** – 10 k – 1 M examples (depends on model scale). <br>• **Quality** – clean, representative, low‑noise. <br>• **Labeling** – supervised tasks need explicit targets; instruction‑tuning can use prompt‑response pairs. |
| **Common tools & platforms** | • **OpenAI API** – `openai.FineTune.create` (e.g., `gpt‑3.5‑turbo`). <br>• **Hugging Face Transformers** + `Trainer` or `accelerate`. <br>• **Azure Machine Learning** – managed compute & experiment tracking. |
| **Pros** | • Full control over model behavior. <br>• Consistent outputs across calls. <br>• Higher accuracy on niche tasks. |
| **Cons** | • Compute‑intensive (GPU hours, possible cloud credits). <br>• May increase inference latency, especially with large checkpoints. <br>• Requires ongoing maintenance (re‑training, versioning). |

### Quick Python snippet (Hugging Face + PEFT)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

model_name = "meta-llama/Meta-Llama-3-8B"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer  = AutoTokenizer.from_pretrained(model_name)

# Low‑rank adapter (LoRA) – only a few MB
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(base_model, lora_cfg)

def tokenize(batch):
    return tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=256)

train_ds = dataset["train"].map(tokenize, batched=True)
val_ds   = dataset["validation"].map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./lora_finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

---  

## 2. Foundations of Prompt Engineering  

| Aspect | Key Points |
|--------|------------|
| **Definition & core techniques** | Design the input text to coax the desired behavior. Common patterns: <br>• **Few‑shot** – embed examples in the prompt. <br>• **Chain‑of‑thought** – ask the model to reason step‑by‑step. <br>• **System prompts** – set role, tone, or constraints at the start. |
| **Tooling & interfaces** | • **OpenAI Playground** – rapid iteration. <br>• **API calls** – embed prompts directly in code. <br>• **Prompt libraries** – LangChain, Promptify, or community repos for reusable templates. |
| **Pros** | • Zero‑cost iteration. <br>• No GPU training; latency identical to the base model. <br>• Can be tweaked per request. |
| **Cons** | • Brittle – small wording changes may flip the output. <br>• Limited for highly specialized vocab or logic. <br>• Harder to guarantee consistent behavior across many calls. |

### Example API call with a few‑shot prompt (OpenAI)

```python
import openai

prompt = """You are a medical coding assistant.

Q: What CPT code corresponds to a routine colonoscopy?
A: 45378

Q: What CPT code corresponds to a chest X‑ray?
A:"""

response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)

print(response.choices[0].message.content.strip())
```

---  

## 3. Decision Matrix – When to Fine‑Tune  

| Situation | Rationale for Fine‑Tuning |
|-----------|---------------------------|
| **Large, domain‑specific dataset** | The model can internalize terminology and patterns, reducing reliance on prompt tricks. |
| **Need for deterministic behavior across many calls** | Weight updates lock in behavior; prompts no longer drift. |
| **Regulatory or security constraints** | Owning a checkpoint enables on‑prem deployment, data isolation, and audit trails. |
| **High‑volume, long‑term usage** | Fixed training cost amortized over billions of tokens; per‑token price often lower than premium prompting models. |

**Rule of thumb** – If you have **≥ 10 k clean examples** *and* expect **> 10 M inference tokens per month**, fine‑tuning usually yields lower total cost and higher reliability.

---  

## 4. Decision Matrix – When to Use Prompt Engineering  

| Situation | Rationale for Prompt Engineering |
|-----------|-----------------------------------|
| **Rapid prototyping or one‑off tasks** | No time to collect data or launch a training job. |
| **Low‑resource environment** | No GPU budget; rely on hosted model APIs. |
| **On‑the‑fly adjustments** | A/B‑test prompt variants, personalize per user, or react to emerging requirements instantly. |
| **Unpredictable query volume or tight budget** | Pay‑as‑you‑go API usage avoids upfront compute spend. |

**Rule of thumb** – If you have **< 5 k labeled examples** *or* need **sub‑daily iteration**, start with prompting.

---  

## 5. Hybrid Strategies & Best Practices  

1. **Prompt + lightweight adapters (LoRA, IA³)**  
   - Train a low‑rank adapter (few hundred MB) on a small dataset while keeping the base model unchanged.  
   - Captures many benefits of fine‑tuning with minimal compute.  

2. **Iterative fine‑tuning after prompt discovery**  
   - Use prompt engineering to surface failure modes.  
   - Curate a targeted dataset of those cases and fine‑tune once.  

3. **Monitoring & fallback**  
   - Log prompt tokens, model responses, latency, and confidence scores.  
   - If confidence drops, fall back to a more robust fine‑tuned model or a vetted prompt template.  

4. **Governance**  
   - Version prompts (e.g., Git) and model checkpoints.  
   - Automate tests: unit tests for prompt output, regression tests for fine‑tuned models.  

### Hybrid code sketch (LoRA with 🤗 PEFT)

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(base, lora_cfg)

# `train_dataset` should contain a few hundred high‑quality examples.
args = TrainingArguments(
    output_dir="./lora_adapter",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
)

trainer = Trainer(model=model, args=args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset)
trainer.train()
```

---  

## 6. Quick Decision Checklist  

| ✅ | Question | Recommended Path |
|----|----------|-------------------|
| 1 | Do you have **≥ 10 k clean, domain‑specific examples**? | ✅ Fine‑tune (or LoRA). |
| 2 | Is **deterministic output** a hard requirement? | ✅ Fine‑tune. |
| 3 | Are **regulatory or data‑privacy constraints** present? | ✅ Fine‑tune and host on‑prem. |
| 4 | Do you need **sub‑daily iteration** or A/B testing? | ✅ Prompt engineering. |
| 5 | Is the **budget limited** and query volume unpredictable? | ✅ Prompt engineering (or lightweight adapters). |
| 6 | Would a **small dataset** (< 5 k) suffice for a quick win? | ✅ Prompt engineering first; consider fine‑tuning later. |

---  

## Conclusion  

* **Fine‑tuning** provides durability, higher accuracy, and compliance guarantees at the cost of compute and maintenance.  
* **Prompt engineering** offers agility and zero upfront cost but can be brittle and harder to scale.  
* **Hybrid approaches**—lightweight adapters, iterative prompt‑driven data collection, and robust monitoring—are increasingly the default for production LLM systems.  

As tooling matures (e.g., OpenAI function calling, LangChain’s “prompt‑to‑adapter” pipelines), the line between the two levers blurs, making a **flexible, data‑driven workflow** the most effective strategy.