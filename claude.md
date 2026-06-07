## Project Goal

Build a Resource-Constrained Agentic Planning Loop that satisfies all assignment requirements.

The implementation must prioritize correctness of budget enforcement, planning behavior, reflection, and traceability over feature richness.

---

## Mandatory Constraints

These constraints are non-negotiable.

### Budget Limits

Per task:

- Maximum 10 LLM calls
- Maximum cost of $0.20

Execution must stop immediately when either limit is reached.

Do not continue execution after a limit is exceeded.

When stopping:

- Return partial progress
- Return completed steps
- Return termination reason
- Save trace information

Warnings alone are insufficient.

---

### LLM Cost Simulation

The project uses Ollama.

Since Ollama is free, simulate costs using:

```python
cost = tokens_used * (0.01 / 1000)
```

Track:

- Total tokens
- Total cost
- Total LLM calls

for every task.

---

### Planning Loop

Implement a ReAct-style planning loop.

Loop structure:

```text
Thought → Action → Observation → Reflection
```

The loop must support replanning.

---

### Reflection Requirement

After every tool call the agent must evaluate:

> Am I making progress toward solving the task?

## Architecture

Create the following structure:

```text
app/
├── agent/
│   ├── planner.py
│   ├── reflector.py
│   ├── executor.py
│   ├── budget.py
│   ├── state.py
│   └── prompts.py
│
├── tools/
│   ├── base.py
│   ├── web_search.py
│   ├── code_executor.py
│   └── skill_matcher.py
│
├── llm/
│   ├── ollama_client.py
│   └── cost_tracker.py
│
├── tasks/
│   ├── sample_tasks.py
│   └── adversarial_tasks.py
│
├── traces/
│
└── tests/

README.md
decisions.md
test_results.md
Dockerfile
.env.example
requirements.txt
```

---

## State Design