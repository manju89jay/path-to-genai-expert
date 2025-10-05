# Essential AI Roadmap (No Timeline)

Each section lists **what to learn** and multiple **high-quality resources**. Pick one primary resource per topic and go deep.

---

## 0) Prerequisites (light)
**Learn:** Python & Git basics; vectors & cosine similarity; reading API docs.  
**Resources:**
- Microsoft “AI for Beginners” — https://github.com/microsoft/AI-For-Beginners
- Data Science for Beginners — https://github.com/microsoft/Data-Science-For-Beginners

---

## 1) Generative AI & LLM Fundamentals
**Learn:** tokens & context windows; prompts vs. tools; embeddings; LLM lifecycle.  
**Resources:**
- Generative AI for Beginners (Microsoft, 21 lessons) — https://github.com/microsoft/generative-ai-for-beginners
- Google Cloud Intro to GenAI (concepts) — https://www.cloudskillsboost.google/journeys/118
- deeplearning.ai “Prompt Engineering for Developers” — https://www.deeplearning.ai/short-courses/

---

## 2) Prompting & Structured Outputs
**Learn:** task decomposition; few-shot patterns; JSON/typed outputs; function/tool calling basics.  
**Resources:**
- deeplearning.ai prompt engineering minis — https://www.deeplearning.ai/short-courses/
- Generative AI for Beginners (prompting lessons) — https://github.com/microsoft/generative-ai-for-beginners

---

## 3) Agents & Orchestration (how AI apps *do things*)
**Learn:** tool schemas; planning/critique loops; multi-agent vs single-agent; guardrails & handoffs.  
**Resources:**
- AI Agents for Beginners (Microsoft) — https://github.com/microsoft/ai-agents-for-beginners
- Semantic Kernel (orchestration SDK) — https://learn.microsoft.com/semantic-kernel/  • Repo: https://github.com/microsoft/semantic-kernel
- LangChain agents (LangGraph concepts) — https://python.langchain.com/

---

## 4) RAG (Retrieval-Augmented Generation) Essentials
**Learn:** chunking; embeddings; hybrid search (BM25 + embeddings); rerankers; context routing.  
**Resources:**
- LangChain RAG docs — https://python.langchain.com/
- LlamaIndex (indices, query engines) — https://docs.llamaindex.ai/

---

## 5) Evaluation, Testing & Observability
**Learn:** golden sets; hallucination/faithfulness; retrieval metrics (Recall@k, nDCG); function-call accuracy; traces & cost/latency dashboards.  
**Resources:**
- Ragas — https://docs.ragas.io/  • Repo: https://github.com/explodinggradients/ragas
- promptfoo (CI for prompts/agents) — https://www.promptfoo.dev/  • Repo: https://github.com/promptfoo/promptfoo
- MLflow for GenAI/Agents — https://mlflow.org/docs/latest/llms/index.html
- Langfuse (open-source tracing) — https://docs.langfuse.com/

---

## 6) Serving / Inference (pick one stack and master it)
**Learn:** server vs gateway; batching/multiplexing; streaming; OpenAI-compatible routes.  
**Options:**
- NVIDIA Triton Inference Server — https://docs.nvidia.com/deeplearning/triton/  • https://github.com/triton-inference-server/server
- TensorRT-LLM — https://nvidia.github.io/TensorRT-LLM/  • https://github.com/NVIDIA/TensorRT-LLM
- vLLM — https://vllm.ai/  • https://github.com/vllm-project/vllm
- Hugging Face TGI — https://huggingface.co/docs/text-generation-inference/index  • https://github.com/huggingface/text-generation-inference

---

## 7) Performance & Cost Optimization
**Learn:** quantization (8/4-bit), KV cache, speculative decoding; distillation/LoRA; routing small→large.  
**Resources:**
- TensorRT overview — https://developer.nvidia.com/tensorrt
- ONNX Runtime GenAI (efficient small/edge models) — https://onnxruntime.ai/docs/genai/  • https://github.com/microsoft/onnxruntime-genai  • https://pypi.org/project/onnxruntime-genai/

---

## 8) Vector Search (standardize on 1–2 backends)
**Learn:** exact vs ANN; HNSW/IVF/IVFPQ; GPU vs CPU; hybrid (BM25 + embeddings).  
**Options & docs:**
- FAISS — https://faiss.ai/  • https://github.com/facebookresearch/faiss
- pgvector — https://github.com/pgvector/pgvector
- Milvus — https://milvus.io/  • Docs: https://milvus.io/docs

---

## 9) Security & Safety
**Learn:** prompt injection; insecure output handling; tool permissioning; data leakage; red-team playbooks.  
**Resources:**
- OWASP Top-10 for LLM Applications — https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Cloudflare explainer (quick ref) — https://blog.cloudflare.com/owasp-top-10-for-llm-apps/

---

## 10) Governance & Compliance (EU/Germany aware)
**Learn:** risk management vocabulary; model cards; data lineage; EU AI Act basics.  
**Resources:**
- NIST AI RMF 1.0 (framework) — https://www.nist.gov/itl/ai-risk-management-framework
- EU AI Act (Official Journal) — https://eur-lex.europa.eu/eli/reg/2024/1689/oj

---

## 11) Edge / Embedded (optional but great for automotive)
**Learn:** SLMs on-device; memory/thermal budgets; offline modes; deterministic tests.  
**Resources:**
- Edge AI for Beginners (Microsoft) — https://github.com/microsoft/edgeai-for-beginners
- ONNX Runtime GenAI — https://onnxruntime.ai/docs/genai/

---

## 12) Portfolio Patterns (apply the above)
Build **small, self-contained demos** that prove understanding:
- Agentic **Spec→Code Assistant** (Agents + RAG + tool calling + evals)
- **Traceability Copilot** (RAG + eval gates + simple dashboard)
- **Serving benchmark** (pick one server; report latency/throughput & cost)

**Document each demo** with: problem → design sketch → metrics → lessons learned (use MLflow/Langfuse for traces).
