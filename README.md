# AI Engineering Portfolio — Consolidated Roadmap

A lightweight, topic-first roadmap to build and showcase essential AI engineering skills. The focus is on pragmatic clusters of knowledge rather than strict timelines so you can go deep on what matters for your goals.

## How to Use This Repo
1. Pick one learning cluster that aligns with your immediate goal.
2. Choose **one primary resource** and **one main tool** from the suggestions.
3. Build a tiny demo; capture core metrics (latency, cost, accuracy) and keep notes on trade-offs.
4. Record lessons learned in each demo folder using a short `README.md`.

## Learning Clusters
Each cluster lists what to master and the best-in-class resources/tools to start with. Pick one primary resource before branching out.

### 0. Foundations & Prompting
- **Learn:** Python/Git refreshers, embeddings & similarity, prompt design patterns, typed outputs/function calling basics.
- **Resources:**
  - Microsoft *AI for Beginners* — https://github.com/microsoft/AI-For-Beginners
  - Microsoft *Data Science for Beginners* — https://github.com/microsoft/Data-Science-For-Beginners
  - Microsoft *Generative AI for Beginners* — https://github.com/microsoft/generative-ai-for-beginners
  - deeplearning.ai *Prompt Engineering for Developers* — https://www.deeplearning.ai/short-courses/
  - Google Cloud *Intro to Generative AI* — https://www.cloudskillsboost.google/journeys/118

### 1. Agentic Workflows & Orchestration
- **Learn:** Tool schemas, planning/critique loops, single vs multi-agent patterns, guardrails and human handoffs.
- **Resources & Tools:**
  - Microsoft *AI Agents for Beginners* — https://github.com/microsoft/ai-agents-for-beginners
  - Microsoft *Semantic Kernel* — https://learn.microsoft.com/semantic-kernel/ • https://github.com/microsoft/semantic-kernel
  - AutoGen (multi-agent workflows) — https://microsoft.github.io/autogen/ • https://github.com/microsoft/autogen
  - LangChain Agents & LangGraph — https://python.langchain.com/

### 2. Retrieval & Knowledge Systems
- **Learn:** Chunking strategies, hybrid search (BM25 + embeddings), rerankers, context routing, vector index trade-offs.
- **Resources & Tools:**
  - LangChain RAG guides — https://python.langchain.com/
  - LlamaIndex (indices & query engines) — https://docs.llamaindex.ai/
  - FAISS — https://faiss.ai/ • https://github.com/facebookresearch/faiss
  - pgvector — https://github.com/pgvector/pgvector
  - Milvus — https://milvus.io/ • Docs: https://milvus.io/docs

### 3. Evaluation, Observability & Safety
- **Learn:** Golden datasets, hallucination/faithfulness metrics, retrieval quality (Recall@k, nDCG), traceability, prompt injection risks, governance vocab.
- **Resources & Tools:**
  - Ragas — https://docs.ragas.io/ • https://github.com/explodinggradients/ragas
  - promptfoo (CI for prompts/agents) — https://www.promptfoo.dev/ • https://github.com/promptfoo/promptfoo
  - MLflow for GenAI/Agents — https://mlflow.org/docs/latest/llms/index.html
  - Langfuse (tracing & analytics) — https://docs.langfuse.com/
  - OWASP Top-10 for LLM Applications — https://owasp.org/www-project-top-10-for-large-language-model-applications/
  - NIST AI Risk Management Framework — https://www.nist.gov/itl/ai-risk-management-framework
  - EU AI Act (Official Journal) — https://eur-lex.europa.eu/eli/reg/2024/1689/oj

### 4. Serving, Inference & Optimization
- **Learn:** Gateway vs model server roles, batching/multiplexing, streaming, KV cache management, quantization, speculative decoding.
- **Resources & Tools:**
  - NVIDIA Triton Inference Server — https://docs.nvidia.com/deeplearning/triton/ • https://github.com/triton-inference-server/server
  - TensorRT-LLM — https://nvidia.github.io/TensorRT-LLM/ • https://github.com/NVIDIA/TensorRT-LLM
  - vLLM — https://vllm.ai/ • https://github.com/vllm-project/vllm
  - Hugging Face Text Generation Inference — https://huggingface.co/docs/text-generation-inference/index • https://github.com/huggingface/text-generation-inference
  - ONNX Runtime GenAI — https://onnxruntime.ai/docs/genai/ • https://github.com/microsoft/onnxruntime-genai • https://pypi.org/project/onnxruntime-genai/
  - TensorRT overview — https://developer.nvidia.com/tensorrt

### 5. Edge, Embedded & Domain Specialization
- **Learn:** Deploying SLMs on-device, memory/thermal budgets, deterministic testing, automotive safety standards.
- **Resources & Tools:**
  - Microsoft *Edge AI for Beginners* — https://github.com/microsoft/edgeai-for-beginners
  - ONNX Runtime GenAI (on-device guidance) — https://onnxruntime.ai/docs/genai/
  - ISO 21448 SOTIF overview — https://www.iso.org/standard/70939.html
  - ISO 26262 overview — https://www.synopsys.com/automotive/what-is-iso-26262.html
  - NVIDIA DLI Training Catalog — https://www.nvidia.com/en-us/training/ • https://courses.nvidia.com/
  - NVIDIA NeMo — https://docs.nvidia.com/nemo/ • https://github.com/NVIDIA/NeMo

## Portfolio Patterns to Practice
Build small, auditable demos that combine the above skills:
- **Spec→Code Assistant:** agentic loop + tool calling + focused evals.
- **Traceability Copilot:** retrieval pipeline with evaluation gates and a lightweight dashboard.
- **Serving Benchmark:** pick one inference stack, benchmark latency/throughput/cost, document findings.

Document each project with problem framing → design sketch → metrics → lessons learned. Use tooling like MLflow or Langfuse to capture traces.

— gpt-5-codex
