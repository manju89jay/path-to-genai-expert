# Experiment Log Template

Use this log to capture every experiment, prompt iteration, or model evaluation. Duplicate the table per run.

## Experiment Metadata
- **Experiment ID:** `YYYYMMDD-<short-name>`
- **Owner:**
- **Related project / issue:**
- **Hypothesis:** What are you trying to learn or improve?
- **Baseline reference:** Link to prior run or control.

## Configuration Snapshot
| Parameter | Value |
|-----------|-------|
| Model / version |
| Prompt / system message |
| Retrieval config (if RAG) |
| Temperature / top-p / top-k |
| Max tokens |
| Tooling / agents |
| Dataset slice |
| Hardware |
| Cost controls |

## Procedure
1. **Setup:** Describe data prep, environment, and scripts used.
2. **Run steps:** Bullet the sequence of actions.
3. **Observability:** Note logging, tracing, telemetry captured.

## Results
| Metric | Score | Target | Delta vs. Baseline |
|--------|-------|--------|---------------------|
| | | | |

- **Qualitative notes:** Annotate failure modes, standout examples, user feedback.
- **Cost & latency:** Record per-call cost, throughput, GPU utilization.
- **Artifacts:** Links to notebooks, dashboards, pull requests.

## Decision
- **Outcome:** ▢ Accept ▢ Reject ▢ Needs follow-up
- **Next actions:** What should happen next? Include owners and due dates.
- **Risks / blockers:** Items to escalate.

## Reflection
- **What worked:**
- **What to improve:**
- **Questions for mentors / stakeholders:**

---
- **Log updated:**
