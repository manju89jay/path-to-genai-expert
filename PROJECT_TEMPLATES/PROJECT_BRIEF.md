# Project Brief Template

Use this template to scope GenAI initiatives before committing engineering time. Keep it to 2–3 pages and link supporting docs instead of embedding them.

## 1. Project Snapshot
- **Project name:**
- **Prepared by / Date:**
- **Phase:** ▢ Discovery ▢ Scoping ▢ Delivery ▢ Launch ▢ Sustain
- **Linked experiment log:**
- **Related issues / OKRs:**

## 2. Problem & Context
- **Business problem:** What pain or opportunity does this address?
- **Target users / stakeholders:** Who benefits? Who signs off?
- **Current workflows:** What is happening today? Include baseline metrics.
- **Strategic alignment:** How does this support org goals or KPIs?

## 3. Success Criteria
- **Primary metric(s):** Quality, latency, cost, satisfaction, risk reduced, etc.
- **Guardrails:** Maximum latency, budget, legal or policy constraints.
- **Definition of done:** Checklist of what must be true for launch approval.

## 4. Solution Outline
- **User journey / scenarios:** Bullet the key interactions.
- **Architecture sketch:** Link to diagram or describe components (prompting, RAG, fine-tuning, agents, integrations).
- **Data sources:** Internal / external datasets, refresh cadence, privacy considerations.
- **Model strategy:** Prompt-only, RAG-first, fine-tune, hybrid. Note reasoning.

## 5. Delivery Plan
| Milestone | Owner | Deliverable | Target date | Dependencies |
|-----------|-------|-------------|-------------|--------------|
| Kickoff |
| Prototype |
| Evaluation | 
| Pilot |
| Launch |

- **Review cadence:** Standups, demos, stakeholder readouts.
- **Tooling:** Tracking (Notion/Jira), experiment logging, observability stack.

## 6. Evaluation Plan
- **Offline metrics:** e.g., BLEU, ROUGE, accuracy, F1, retrieval hit rate.
- **Online metrics:** e.g., CSAT, retention, task completion time.
- **Qualitative review:** SMEs, red-teaming, LLM-as-judge with human audit frequency.
- **Regression plan:** Link to evaluation scripts and acceptance thresholds.

## 7. Risks & Mitigations
| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| Data quality |
| Prompt injection |
| Cost overrun |
| Stakeholder adoption |

- **Open questions:**

## 8. Compliance & Safety
- **PII/Sensitive data handling:** Masking, encryption, retention.
- **Responsible AI review:** Bias checks, hallucination guardrails, escalation path.
- **Security requirements:** Secrets management, API scopes, access control.

## 9. Launch Readiness
- **Documentation:** User guides, runbooks, model cards.
- **Training / enablement:** Sessions scheduled? Materials linked?
- **Support model:** Owners for monitoring, incident response, feedback triage.

## 10. Post-Launch Plan
- **Success measurement window:** How long until final evaluation?
- **Iteration backlog:** Next experiments, stretch goals, phased rollout.
- **Sunset criteria:** When do we retire or replace this solution?

---
- **Last reviewed:**
- **Next review date:**
