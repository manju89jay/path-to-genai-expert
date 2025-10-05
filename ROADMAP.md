# One-Year Roadmap — Lead & Ship AI

## Outcomes

- **Fluent stakeholder talk:** translate goals ↔ constraints, estimate cost/latency.
- **Technical leadership:** architect agentic RAG systems with eval gates and ops.
- **Operational maturity:** ship with tests, safety, monitoring, and rollback plans.

---

## Quarter 1 — Core Fluency & Agentic Patterns

### Q1 Master/Advance

- LLM essentials (tokens, context windows, tool/function calling, structured JSON output, streaming)
- RAG basics (chunking, embeddings, hybrid search, reranking, vector stores, routing)
- Agents & orchestration (planning/critique loops, typed tools, guardrails)
- Latency/cost knobs (caching, model selection, response compression)

### Q1 Deliverable

- **Spec-to-Code Assistant**: RAG + function-calling agent to draft adapter code/diffs.

### Q1 KPIs

- Dev acceptance rate ≥ 30%
- P50 latency ≤ 2s; € per task tracked
- 20+ golden Qs; zero-shot eval baseline

---

## Quarter 2 — MLOps, Evaluation & Productionization

### Q2 Master/Advance

- Experiment tracking & versioning (datasets, prompts, models)
- Offline & online evals (faithfulness, answer relevance, Recall@k, function-call accuracy)
- Observability & tracing (prompt→tools→output), cost/latency dashboards
- CI/CD for AI apps, prompt pinning, shadow deploys, data governance basics

### Q2 Deliverable

- **Requirements Traceability Copilot** (req → design → code → test), drift alerts

### Q2 KPIs

- Eval suite ≥ 50 golden Qs; tracing enabled
- CI green; rollback plan defined
- Mean time to trace ↓ 50%

---

## Quarter 3 — Serving, Performance & Edge (Automotive-adjacent)

### Q3 Master/Advance

- Inference stacks (serverless vs self-host), scheduling, batching/multiplexing
- Optimization (quantization 8/4-bit, distillation, speculative decoding, KV caching)
- Serving platforms (Triton / TensorRT-LLM / vLLM / TGI)
- Edge patterns (on-device constraints, offline modes, safety fallbacks)
- Simulation testing (deterministic replays, failure injection)

### Q3 Deliverable

- **Simulation Test Synthesizer**: mines failure logs → proposes new scenarios + Jira MREs

### Q3 KPIs

- Throughput +50%, cost/task −30%
- Offline demo runs within memory/thermal budget

---

## Quarter 4 — Safety, Governance & Product Leadership

### Q4 Master/Advance

- Safety & policy (red-teaming prompt attacks/tool abuse, permissioning, HITL)
- Governance (model cards, data lineage, audit logs; NIST AI RMF; EU AI Act awareness)
- Automotive safety mindset (ISO 26262/21448 concepts for ML-adjacent tools)
- Product leadership (PRDs, ADRs, risk register, go/no-go criteria)

### Q4 Deliverable

- **Release-Readiness AI Gatekeeper**: pre-merge evals, safety checks, traceability report

### Q4 KPIs

- Defects caught pre-merge ↑; audit trail completeness ≥ 90%
- Red-team report delivered; governance checklist adopted

---

## Milestones

| Quarter | Demo Shipped | Key KPIs | Notes |
|---|---|---|---|
| Q1 |  | Acceptance ≥30%, P50≤2s |  |
| Q2 |  | Eval≥50 Qs, tracing on |  |
| Q3 |  | +50% throughput, −30% cost |  |
| Q4 |  | Audit≥90%, red-team report |  |

---

## Beginner Ramp (Weeks 1–6)

- **W1–2:** LLM basics; run one notebook end-to-end
- **W3:** RAG tiny Q&A with FAISS/pgvector
- **W4:** Evals (promptfoo/ragas) + fixes
- **W5–6:** Agents with two tools + guardrails
