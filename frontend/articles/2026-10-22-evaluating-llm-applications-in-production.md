# Evaluating LLM Applications in Production  

## Introduction  

- **Why rigorous evaluation matters** – In production an LLM directly impacts revenue, brand reputation, and user safety. A single hallucination or latency spike can cost dollars, customers, or regulatory penalties.  
- **Scope & audience** – This guide is for engineers building the service layer, MLOps teams maintaining pipelines, and product leaders defining success criteria.  
- **High‑level framework** – We walk through a repeatable lifecycle: define metrics → govern data → select & fine‑tune models → monitor continuously → enforce security & ethics.  

---  

## 1. Defining Success Metrics  

| Category               | Example KPI                     | Measurement approach |
|------------------------|---------------------------------|----------------------|
| **Business‑level**     | Conversion rate, ticket deflection, churn reduction | A/B test the LLM‑enabled feature against a baseline and compute lift. |
| **Technical performance** | p95 latency, throughput (req/s), cost per 1 k tokens | Instrument request timing; multiply token usage by provider pricing. |
| **Quality**            | Accuracy, relevance, factuality, toxicity, hallucination rate | Curated test sets + automated classifiers (e.g., factuality scorer, toxicity filter). |
| **Trade‑off matrix**   | Business impact vs. cost vs. risk | Assign a weight (0‑1) to each metric and compute a composite score:  

```python
def composite_score(metrics, weights):
    return sum(metrics[k] * weights[k] for k in metrics)
```  

*Tip*: Re‑evaluate weights each quarter as product priorities shift.  

---  

## 2. Data Governance & Pre‑processing  

1. **Training vs. inference data**  
   - *Training*: static, versioned, audited for bias.  
   - *Inference*: streaming user queries; must be sanitized in real time.  

2. **Data‑quality checks**  
   - **Bias audit** – run demographic‑parity checks on a representative sample.  
   - **Representativeness** – verify coverage of edge‑case intents (e.g., rare SKUs).  
   - **Freshness** – schedule a nightly diff against the source to flag stale entries.  

3. **Prompt engineering & context management**  
   - Store prompts in a version‑controlled repository.  
   - Use a template that injects system instructions, user context, and a token‑budget guard.  

   ```python
   PROMPT_TEMPLATE = """\
   You are a helpful support assistant. Answer concisely and cite sources.
   Context: {context}
   Question: {question}
   """
   def render_prompt(context, question):
       return PROMPT_TEMPLATE.format(context=context, question=question)
   ```  

4. **Versioning**  
   - **Datasets** – use DVC or MLflow artifacts (`dvc add data/train.json`).  
   - **Prompts** – tag commits (`v1.2-prompt`) and link them to model releases.  

---  

## 3. Model Selection & Fine‑tuning Strategies  

| Strategy                | When to use                              | Pros                                           | Cons                                 |
|-------------------------|------------------------------------------|------------------------------------------------|--------------------------------------|
| **Off‑the‑shelf** (e.g., GPT‑4o) | Low latency, generic tasks, limited budget | No training infra, rapid iteration            | Higher per‑token cost, limited domain adaptation |
| **Fine‑tuned** (API or self‑hosted) | Domain‑specific language, higher relevance | Improves KPI lift, reduces hallucinations      | Requires labeled data and compute   |
| **Custom‑trained** (full model) | Proprietary data, strict IP constraints | Full control over architecture & data          | Massive R&D cost, longer time‑to‑market |

### Parameter‑efficiency techniques  

- **LoRA / adapters** – inject low‑rank updates, saving GPU memory.  
- **Quantization (int8/4)** – cut inference cost by ~2‑3× with <1 % accuracy loss.  

### Evaluating fine‑tuning impact  

1. Run the test suite from **Section 1** on both the base and fine‑tuned model.  
2. Plot metric deltas and compute statistical significance (e.g., bootstrap).  

### Cost‑benefit analysis  

```python
cost_per_req = (tokens_per_req * price_per_1k) / 1000
roi = (business_lift - cost_per_req) / cost_per_req
```  

Choose the model variant with the highest positive ROI while satisfying latency caps.  

---  

## 4. Monitoring, Observability, and Continuous Evaluation  

1. **Real‑time logging** (structured JSON)  

   ```json
   {
     "timestamp": "2026-10-22T14:03:12Z",
     "model": "gpt-4o-finetuned",
     "latency_ms": 187,
     "tokens_input": 42,
     "tokens_output": 128,
     "error": null,
     "policy_violation": false
   }
   ```  

   Push logs to a centralized store (e.g., Loki, CloudWatch).  

2. **Automated quality checks**  
   - **Regression tests** – nightly run of ~5 k benchmark prompts; alert if the composite score drops >5 %.  
   - **A/B testing** – split traffic 80/20 between candidate and production models; track business KPI lift.  
   - **Shadow inference** – duplicate live requests to a “shadow” model; compare outputs without affecting users.  

3. **Alerting**  
   - Latency > p95 + 2 × MAD → PagerDuty.  
   - Factuality score < 0.7 for > 1 % of requests → Slack.  
   - Policy‑violation rate spikes → security incident.  

4. **Feedback loops**  
   - **Human‑in‑the‑loop (HITL)** – surface low‑confidence responses to reviewers; store corrections.  
   - **RLHF pipeline** – periodically retrain a reward model on collected human feedback and fine‑tune the LLM with PPO.  

---  

## 5. Security, Compliance, and Ethical Considerations  

| Area            | Controls |
|-----------------|----------|
| **Data privacy** | PII redaction via regex + NER before logging; encrypt logs at rest; honor GDPR/CCPA deletion requests. |
| **Model security** | Input sanitization to prevent prompt injection; run adversarial robustness tests (e.g., TextFooler). |
| **Ethical guardrails** | Deploy a toxicity classifier (e.g., Perspective API) as a post‑processor; maintain a bias‑mitigation checklist for each release. |
| **Governance** | Record an immutable audit trail: model version → dataset version → prompt version → evaluation results (e.g., AWS Glacier). |

*Sample policy‑enforcement snippet*  

```python
def enforce_policy(response):
    if toxicity_score(response) > 0.6:
        return "I’m sorry, I can’t help with that."
    return response
```  

---  

## Conclusion  

- **End‑to‑end lifecycle** – Business‑driven metrics, disciplined data handling, careful model selection, continuous observability, and robust security/ethics form a feedback loop where each step informs the next.  
- **Key takeaways**  
  1. Quantify success with a weighted metric matrix.  
  2. Treat prompts and datasets as first‑class, versioned artifacts.  
  3. Leverage parameter‑efficient fine‑tuning to balance cost and performance.  
  4. Automate monitoring and embed human feedback loops.  
  5. Embed privacy, security, and ethical guardrails from day one.  
- **Call to action** – Institutionalize a systematic evaluation pipeline, iterate on the composite score each sprint, and treat LLMs as a continuously evolving service rather than a one‑off model drop. A production‑ready LLM will then be trustworthy, performant, and aligned with business goals.