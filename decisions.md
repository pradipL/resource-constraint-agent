# Design Decisions

---

## 1. Custom ReAct Agent over Prebuilt Package

**Decision:** Build a custom ReAct agent using LangGraph instead of a prebuilt ReAct agent package.

**Reasoning:** The system enforces strict constraints — no more than 10 LLM calls per task and a maximum cost of $0.20. A custom implementation provides finer control over each reasoning step, making it straightforward to enforce call limits and prevent uncontrolled LLM usage that could cause cost leakage or inefficient execution. A prebuilt package does not expose the hooks needed to intercept and gate each call at this level of granularity.

---

## 2. Docker Sandbox over Daytona

**Decision:** Use Docker containers as the sandbox environment instead of Daytona.

**Reasoning:** The core requirement is to execute untrusted code that can generate artifacts such as PDFs and other files, which then need to be persisted and made available to the user.

In the current design, generated files are stored locally and served as downloadable links after extraction from the container. Alternatively, files could be pushed to S3 and served via pre-signed URLs, avoiding any coupling between the sandbox filesystem and the host.

Running untrusted code directly on the host is a security risk — malicious code could cause system compromise or data corruption. While Daytona provides a more managed sandbox abstraction, Docker was chosen for this assignment because it achieves process isolation and controlled execution in a lightweight manner, while still satisfying the functional and security requirements.

---

## 3. Hybrid Planning + ReAct + Reflection over Pure ReAct

**Decision:** Use a hybrid approach combining Planning, ReAct, and Reflection instead of a pure ReAct loop.

**Reasoning:** A pure ReAct loop reacts to observations one step at a time without an upfront plan, which makes it prone to drifting off course or repeating failed strategies. The hybrid approach works as follows:

1. **Plan** — the system formulates a structured numbered plan before taking any action.
2. **Execute** — the plan is executed step by step using a ReAct-style loop (Thought → Action → Observation).
3. **Reflect** — after each tool call, the agent evaluates whether meaningful progress was made.
4. **Replan** — if a step fails or stalls, the agent revises the plan and retries with a different approach.

This ensures more reliable execution, better adaptability to unexpected failures, and prevents the agent from getting stuck in an unproductive loop.

![Planning, ReAct, and Reflection architecture](materials/architecture/langgraph_architecture.png)

## 4. RAG over other Tools

**Decision:** Use RAG (Retrieval-Augmented Generation) instead of other tools.

**Reasoning:** RAG provides richer context to the LLM compared to basic keyword or document lookup tools. It reduces hallucination by grounding responses in retrieved content rather than relying solely on the model's parametric knowledge. It also enables querying an internal knowledge base, which is a key capability for domain-specific tasks.

As a practical example relevant to Jobins as a recruiting company — RAG could be used to store candidate CVs and semantically query them to find the best match for a given role, going far beyond what simple filtering or keyword search can achieve.

---

## 5. Choosing OpenAI Instead of a Local Ollama Model

**Decision:** Use OpenAI (`gpt-4o-mini`) as the LLM provider instead of a locally hosted Ollama model.

**Reasoning:** The initial implementation used Ollama with `qwen2.5:7b`. During testing, two problems became apparent: the model lacked the reasoning capability needed to reliably select the right tools, and inference was too slow for an interactive agent loop. Switching to OpenAI resolved both issues — the agent now makes accurate tool-use decisions and responds at a practical speed.

**Trade-off:** OpenAI incurs API costs, whereas Ollama is free. The architecture keeps the provider behind a Factory + Strategy abstraction, so switching back to a local model requires only a single `.env` change when a capable enough model becomes available locally (also can be inject at the runtime for routing).

**Cost Reduction Strategy:** A hybrid routing approach can reduce costs further — use OpenAI for reasoning-heavy tasks such as tool selection and decision-making, while delegating simpler tasks like planning to a local or lower-cost model.