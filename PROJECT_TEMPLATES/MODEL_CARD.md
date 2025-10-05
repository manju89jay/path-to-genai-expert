# Model Card Template

Summarize model intent, data lineage, evaluation, and caveats. Update this card whenever you ship a new version.

## 1. Model Overview
- **Model name / version:**
- **Release date:**
- **Authors / owners:**
- **Intended use cases:** List supported tasks and audiences.
- **Out-of-scope uses:** Explicitly state disallowed or unsafe scenarios.

## 2. Model Details
- **Base model / architecture:** Include size, provider, license.
- **Adaptation method:** Prompt-engineering, fine-tuning, LoRA/QLoRA, RLHF, etc.
- **Training / adaptation codebase:** Link to repo or scripts.
- **Hyperparameters:** Batch size, lr, max length, temperature, etc.
- **Compute footprint:** Hardware, GPU hours, energy estimates if available.

## 3. Data
- **Sources:** Datasets, internal corpora, scraping notes, collection dates.
- **Preprocessing:** Cleaning, filtering, chunking, augmentation steps.
- **Data balancing:** Coverage across domains, demographics, languages.
- **Sensitive attributes handled:** How bias and privacy were addressed.
- **Data quality checks:** Manual review, automated validation, lineage tracking.

## 4. Evaluation
| Task | Dataset / Split | Metric | Score | Threshold | Notes |
|------|-----------------|--------|-------|-----------|-------|
| Example | | | | | |

- **Prompt/RAG evals:** Describe rubrics, judges, cost/latency budgets.
- **Human evaluation:** SME reviewers, sample size, agreement scores.
- **Drift monitoring:** Tests scheduled post-launch, triggers for rollback.

## 5. Limitations & Risks
- **Known failure modes:** e.g., hallucinations, refusal to comply.
- **Bias & fairness concerns:** Populations or topics requiring caution.
- **Security considerations:** Prompt injection susceptibility, data leakage risks.
- **Mitigations:** Filters, guardrails, fallback flows, human-in-the-loop.

## 6. Usage Guidelines
- **Integration instructions:** APIs, SDKs, environment variables.
- **Input constraints:** Token limits, content restrictions.
- **Output post-processing:** Validation, moderation, grounding steps.
- **Support / escalation:** On-call contact, response SLA, incident playbook.

## 7. Version History
| Version | Date | Changes | Owner | Notes |
|---------|------|---------|-------|-------|
| 1.0 |

---
- **Last reviewed:**
- **Next review date:**
