# How LangGraph Models Agent Workflows as Graphs  

## Introduction  

- **Why graph‑based visualisation matters** – Agent pipelines quickly become tangled when actions, tool calls, and branching logic are expressed as linear code. A graph makes control flow explicit, improves debugging, and enables automated analysis (e.g., dead‑end detection, cost estimation).  
- **LangGraph in a nutshell** – LangGraph is a lightweight SDK that lets you declare LLM‑driven agents as directed graphs. Nodes encapsulate atomic actions (prompts, tool calls, custom functions) and edges encode the transition rules that drive the workflow.  
- **What you’ll learn** – This article covers core graph concepts, node definition, edge wiring, building a complete workflow, and advanced patterns such as dynamic mutation, state handling, and testing.  

---  

## 1. Core Graph Concepts  

| Concept | Meaning in LangGraph |
|---------|----------------------|
| **Node** | An atomic agent action: a prompt to an LLM, a call to an external tool, a memory lookup, or any user‑defined function. |
| **Edge** | A directed transition from one node to another. Edges can carry **conditions** (e.g., `confidence > 0.8`), **triggers** (output tags), or **probabilities** for stochastic routing. |
| **State & Context** | A mutable dictionary (`ctx`) that travels with the token. Nodes read from and write to it, enabling data sharing without global variables. |
| **Supported Graph Types** | - **Static** – Fixed topology defined at import time. <br> - **Dynamic** – Nodes/edges can be added or removed during execution. <br> - **Hierarchical** – Sub‑graphs (modules) can be nested, allowing reuse of complex patterns. |

---  

## 2. Defining Agent Nodes  

### 2.1 Creating a node  

```python
from langgraph import Node, InputSchema, OutputSchema

class PromptNode(Node):
    class In(InputSchema):
        user_query: str

    class Out(OutputSchema):
        response: str
        intent: str | None = None   # optional metadata

    def run(self, ctx: dict, inputs: In) -> Out:
        # Send the prompt to the LLM (abstracted by LangGraph)
        llm_resp = self.llm.complete(
            f"User: {inputs.user_query}\nReply:"
        )
        return self.Out(
            response=llm_resp.text,
            intent=llm_resp.metadata.get("intent")
        )
```

- **Schema** – `In` and `Out` define the node’s contract, enabling type‑checking and automatic documentation.  
- **Atomicity** – The node does one thing (here: generate a response). Complex logic belongs in separate nodes.  

### 2.2 Built‑in node categories  

| Category | Typical use |
|----------|-------------|
| **LLM** | Prompt templates, chat completions, function‑calling LLMs. |
| **Tool** | External APIs, database queries, search services. |
| **Memory** | Vector‑store retrieval, short‑term conversation buffers. |
| **Custom** | Pure Python logic, data transformation, validation. |

### 2.3 Decorating nodes with metadata  

```python
@node_meta(
    tags=["nlp", "intent"],
    version="1.2",
    cost_estimate=0.0002
)
class IntentNode(PromptNode):
    pass
```

- **Tags** help with discovery and automated cost monitoring.  
- **Version** enables safe upgrades in production pipelines.  

---  

## 3. Wiring the Workflow: Edges and Transition Logic  

### 3.1 Declaring simple edges  

```python
graph = Graph()
graph.add_edge("intent_node", "search_node")   # source → target
graph.add_edge("search_node", "response_node")
```

### 3.2 Conditional routing  

```python
graph.add_edge(
    "intent_node",
    "escalation_node",
    condition=lambda ctx, out: out.intent == "complaint"
                               and out.confidence < 0.7
)
```

- The lambda receives the current context and the source node’s output.  
- LangGraph also supports `match` statements for richer pattern matching.  

### 3.3 Looping & recursion  

```python
# Re‑ask the user if confidence is low, up to 3 attempts
graph.add_edge(
    "clarify_node",
    "intent_node",
    condition=lambda ctx, out: ctx.get("retries", 0) < 3
                               and out.confidence < 0.6,
    on_enter=lambda ctx: ctx.update(
        {"retries": ctx.get("retries", 0) + 1}
    )
)
```

- Self‑edges create loops; a bounded counter prevents infinite recursion.  

### 3.4 Parallel branches & merging (fork/join)  

```python
graph.fork("intent_node", ["search_node", "knowledge_node"])
graph.join(["search_node", "knowledge_node"], "response_node")
```

- The fork spawns two independent sub‑paths; the join waits until both have produced output before proceeding.  

---  

## 4. Building a Complete LangGraph Workflow  

Below is a minimal **Customer Support Bot** that demonstrates the concepts introduced so far.

### 4.1 Define the nodes  

```python
class IntentNode(PromptNode):
    pass


class RetrievalNode(Node):
    class In(InputSchema):
        query: str

    class Out(OutputSchema):
        docs: list[str]

    def run(self, ctx: dict, inputs: In) -> Out:
        docs = self.vector_store.search(inputs.query, top_k=3)
        return self.Out(docs=docs)


class ResponseNode(Node):
    class In(InputSchema):
        intent: str
        docs: list[str] | None = None
        user_query: str

    class Out(OutputSchema):
        answer: str

    def run(self, ctx: dict, inputs: In) -> Out:
        # Build a prompt that combines intent and retrieved docs
        prompt = f"Intent: {inputs.intent}\n"
        if inputs.docs:
            prompt += "Relevant info:\n" + "\n".join(inputs.docs) + "\n"
        prompt += f"User: {inputs.user_query}\nAnswer:"
        resp = self.llm.complete(prompt)
        return self.Out(answer=resp.text)
```

### 4.2 Wire the graph  

```python
g = Graph(name="customer_support")

g.add_node("intent", IntentNode())
g.add_node("search", RetrievalNode())
g.add_node("response", ResponseNode())
g.add_node("escalate", EscalationNode())   # custom fallback node

# Conditional edges
g.add_edge(
    "intent",
    "search",
    condition=lambda ctx, out: out.intent in {"faq", "info"}
)
g.add_edge(
    "intent",
    "escalate",
    condition=lambda ctx, out: out.intent == "complaint"
)
g.add_edge("search", "response")
g.add_edge(
    "intent",
    "response",
    condition=lambda ctx, out: out.intent == "smalltalk"
)
```

### 4.3 Initialise and run  

```python
engine = GraphEngine(g)

user_input = "I can't reset my password."
result = engine.run(
    start_node="intent",
    inputs={"user_query": user_input}
)

print(result["answer"])
```

### 4.4 Visualise the graph  

```python
g.visualize(output="support_workflow.svg")
```

- `visualize` produces a Graphviz SVG, useful for documentation and debugging.  
- The SDK also ships with a live UI (`langgraph serve`) that shows node execution times, token usage, and context snapshots.  

---  

## 5. Advanced Features & Best Practices  

### 5.1 Dynamic graph modification  

```python
def add_feedback_node(ctx, out):
    if out.satisfaction < 0.5:
        g.add_node("feedback", FeedbackNode())
        g.add_edge("response", "feedback")
        g.add_edge("feedback", "escalate")
```

- Nodes can be injected at runtime based on real‑time signals (e.g., low satisfaction scores).  

### 5.2 State‑management strategies  

| Scope | Recommendation |
|------|----------------|
| **Global** | Store data that must persist across the whole conversation (e.g., user ID, auth tokens) in `ctx["session"]`. |
| **Node‑local** | Keep transient data (e.g., temporary API responses) in the node’s output; avoid polluting the global context. |

### 5.3 Error handling & fallback sub‑graphs  

```python
g.add_error_handler(
    target="response",
    fallback="fallback_node",
    condition=lambda exc: isinstance(exc, TimeoutError)
)
```

- Errors trigger a dedicated sub‑graph, guaranteeing graceful degradation.  

### 5.4 Performance considerations  

| Technique | When to use |
|-----------|-------------|
| **Caching** (`@cached`) | Expensive look‑ups such as vector‑store queries. |
| **Batch execution** (`batch=True`) | Multiple user queries hit the same node; reduces LLM request overhead. |
| **Parallelism** (forked branches) | Independent sub‑paths can run on separate workers; monitor CPU/GPU utilisation to avoid contention. |

### 5.5 Testing and validation  

```python
def test_intent_routing():
    ctx, out = engine.run_one(
        "intent",
        {"user_query": "My order is late."}
    )
    assert out.intent == "order_status"
    assert g.next_node("intent", out) == "search"
```

- **Unit tests** should target node logic (`run` methods).  
- **Graph‑routing tests** verify that edges fire as expected (`next_node` helper).  
- For end‑to‑end verification, use `engine.simulate(inputs, steps=10)` to step through the graph without external calls.  

---  

## Conclusion  

LangGraph transforms opaque LLM orchestration code into **explicit, maintainable graphs**. By modelling each atomic action as a node and every decision as an edge, developers gain:

- **Modularity** – Swap, version, or reuse nodes without touching surrounding logic.  
- **Observability** – Auto‑generated diagrams, per‑node metrics, and context inspection simplify debugging.  
- **Scalability** – Hierarchical sub‑graphs, parallel forks, and dynamic mutation support both prototypes and production‑grade pipelines.  

### Next steps  

1. Install the SDK (`pip install langgraph`) and explore the `examples/` directory.  
2. Contribute reusable patterns (e.g., retrieval‑augmented generation) to the community repo.  
3. Integrate LangGraph with your existing orchestrators (Airflow, Prefect) to run LLM workloads at scale.  

**Happy graph‑building!**