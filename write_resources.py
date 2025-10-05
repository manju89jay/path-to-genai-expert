import json

resources = {
    "Orientation & Study Plan": [
        {
            "title": "KAUST Generative AI Training Curriculum",
            "url": "https://github.com/kaust-generative-ai/training-curriculum",
            "format": "Curriculum",
            "level": "Beginner",
            "est_time_hrs": 6,
            "why": "Sets expectations for weekly cadence and gives a structured campus-style plan you can adapt to your own schedule."
        },
        {
            "title": "GenAI Curriculum Tracker",
            "url": "https://github.com/nckclrk/generative-ai-curriculum",
            "format": "Workbook",
            "level": "Beginner",
            "est_time_hrs": 3,
            "why": "Provides checklist-driven study logs and note templates for keeping an experiment journal."
        },
        {
            "title": "Microsoft Generative AI for Beginners",
            "url": "https://github.com/microsoft/generative-ai-for-beginners",
            "format": "Course",
            "level": "Beginner",
            "est_time_hrs": 20,
            "why": "Offers a comprehensive syllabus you can pace at 4/8/12/24 week intervals with clear module outcomes."
        }
    ],
    "Module 00 · Introduction": [
        {
            "title": "fastai fastsetup",
            "url": "https://github.com/fastai/fastsetup",
            "format": "Setup guide",
            "level": "Beginner",
            "est_time_hrs": 1,
            "why": "Walks through reliable Python environment bootstrapping on Windows, macOS, and Linux."
        },
        {
            "title": "Google Colab Tools",
            "url": "https://github.com/googlecolab/colabtools",
            "format": "Documentation",
            "level": "Beginner",
            "est_time_hrs": 1,
            "why": "Explains cloud notebook workflows, GPU quotas, and collaboration tips for quick experiments."
        },
        {
            "title": "dair-ai Machine Learning YouTube Courses",
            "url": "https://github.com/dair-ai/ML-YouTube-Courses",
            "format": "Playlist index",
            "level": "Beginner",
            "est_time_hrs": 8,
            "why": "Curates high-quality video series to slot into your personal learning sprint."
        }
    ],
    "Module 01 · GenAI Fundamentals": [
        {
            "title": "OpenAI Cookbook",
            "url": "https://github.com/openai/openai-cookbook",
            "format": "Cookbook",
            "level": "Beginner",
            "est_time_hrs": 6,
            "why": "Covers API basics, embeddings, and prompt iteration with runnable notebooks."
        },
        {
            "title": "Hugging Face Transformers",
            "url": "https://github.com/huggingface/transformers",
            "format": "Library docs",
            "level": "Intermediate",
            "est_time_hrs": 10,
            "why": "Teaches model architectures, tokenizers, and inference utilities with examples."
        },
        {
            "title": "Karpathy LLM101n",
            "url": "https://github.com/karpathy/LLM101n",
            "format": "Course",
            "level": "Intermediate",
            "est_time_hrs": 12,
            "why": "Demystifies transformer math with annotated code and lecture videos."
        }
    ],
    "Math for GenAI": [
        {
            "title": "nn-zero-to-hero",
            "url": "https://github.com/karpathy/nn-zero-to-hero",
            "format": "Course",
            "level": "Beginner",
            "est_time_hrs": 10,
            "why": "Refreshes calculus and optimization intuition with code-first notebooks."
        },
        {
            "title": "Numerical Linear Algebra",
            "url": "https://github.com/fastai/numerical-linear-algebra",
            "format": "Book",
            "level": "Intermediate",
            "est_time_hrs": 12,
            "why": "Focuses on matrix decompositions that power embeddings and attention."
        },
        {
            "title": "Math for ML",
            "url": "https://github.com/cadizm/math-for-ml",
            "format": "Notes",
            "level": "Beginner",
            "est_time_hrs": 5,
            "why": "Summarizes probability and vector math essentials for GenAI prompts and loss functions."
        }
    ],
    "Module 02 · Prompting & Structured Outputs": [
        {
            "title": "Prompt Engineering Guide",
            "url": "https://github.com/dair-ai/Prompt-Engineering-Guide",
            "format": "Guide",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Collects tested prompt patterns with JSON schema examples."
        },
        {
            "title": "Awesome Prompt Engineering",
            "url": "https://github.com/promptslab/Awesome-Prompt-Engineering",
            "format": "Repository",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Provides tooling references and benchmarks for structured outputs."
        },
        {
            "title": "Outlines",
            "url": "https://github.com/outlines-dev/outlines",
            "format": "Library",
            "level": "Intermediate",
            "est_time_hrs": 3,
            "why": "Offers deterministic prompting utilities for JSON, regex, and CFG-constrained outputs."
        }
    ],
    "Prompt Engineering Essentials": [
        {
            "title": "Google Cloud Generative AI Examples",
            "url": "https://github.com/GoogleCloudPlatform/generative-ai",
            "format": "Examples",
            "level": "Beginner",
            "est_time_hrs": 5,
            "why": "Demonstrates role/task/format prompting across Vertex AI and open models."
        },
        {
            "title": "Anthropic Cookbook",
            "url": "https://github.com/anthropics/anthropic-cookbook",
            "format": "Cookbook",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Shows Claude function-calling, safety filters, and evaluation loops."
        },
        {
            "title": "Microsoft Promptflow",
            "url": "https://github.com/microsoft/promptflow",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Adds reproducible prompt pipelines with versioned inputs and outputs."
        }
    ],
    "Module 03 · Agents & Orchestration": [
        {
            "title": "LangGraph",
            "url": "https://github.com/langchain-ai/langgraph",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Implements stateful agent workflows with guardrails and retries."
        },
        {
            "title": "Marvin",
            "url": "https://github.com/PrefectHQ/marvin",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Shows pragmatic agent orchestration with Prefect observability hooks."
        },
        {
            "title": "Semantic Kernel",
            "url": "https://github.com/microsoft/semantic-kernel",
            "format": "SDK",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Blends planners, connectors, and memory for enterprise-grade orchestration."
        }
    ],
    "Agents & Tool-Use": [
        {
            "title": "AutoGen",
            "url": "https://github.com/microsoft/autogen",
            "format": "Framework",
            "level": "Advanced",
            "est_time_hrs": 8,
            "why": "Explores multi-agent collaboration, human-in-the-loop, and tool routing."
        },
        {
            "title": "Langroid",
            "url": "https://github.com/langroid/langroid",
            "format": "Framework",
            "level": "Advanced",
            "est_time_hrs": 6,
            "why": "Focuses on deterministic tool contracts and streaming traces."
        },
        {
            "title": "Tool Calling Guide",
            "url": "https://github.com/ALucek/tool-calling-guide",
            "format": "Guide",
            "level": "Intermediate",
            "est_time_hrs": 2,
            "why": "Summarizes function schema design and safety patterns for external tools."
        }
    ],
    "Module 04 · RAG Essentials": [
        {
            "title": "LlamaIndex",
            "url": "https://github.com/jerryjliu/llama_index",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Covers indices, retrievers, and evaluators for small-to-medium RAG stacks."
        },
        {
            "title": "Pinecone Examples",
            "url": "https://github.com/pinecone-io/examples",
            "format": "Examples",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Demonstrates hybrid search, metadata filters, and streaming inserts."
        },
        {
            "title": "Haystack",
            "url": "https://github.com/deepset-ai/haystack",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Provides modular pipelines for retrievers, rerankers, and evaluators."
        }
    ],
    "Retrieval-Augmented Generation": [
        {
            "title": "GraphRAG",
            "url": "https://github.com/microsoft/graphrag",
            "format": "Toolkit",
            "level": "Advanced",
            "est_time_hrs": 6,
            "why": "Shows how to capture entity graphs for high-context retrieval."
        },
        {
            "title": "fastRAG",
            "url": "https://github.com/IntelLabs/fastRAG",
            "format": "Toolkit",
            "level": "Advanced",
            "est_time_hrs": 5,
            "why": "Benchmarks latency, cost, and accuracy trade-offs across RAG configurations."
        },
        {
            "title": "Awesome GraphRAG",
            "url": "https://github.com/DEEP-PolyU/Awesome-GraphRAG",
            "format": "Curated list",
            "level": "Advanced",
            "est_time_hrs": 3,
            "why": "Aggregates research, frameworks, and datasets for graph-enhanced RAG."
        }
    ],
    "Module 05 · Evaluation & Observability": [
        {
            "title": "Ragas",
            "url": "https://github.com/explodinggradients/ragas",
            "format": "Library",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Implements retrieval and answer quality metrics with dataset scaffolding."
        },
        {
            "title": "promptfoo",
            "url": "https://github.com/promptfoo/promptfoo",
            "format": "CLI",
            "level": "Intermediate",
            "est_time_hrs": 3,
            "why": "Automates prompt regression tests with diff-friendly outputs."
        },
        {
            "title": "Langfuse",
            "url": "https://github.com/langfuse/langfuse",
            "format": "Platform",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Delivers tracing, analytics, and user feedback dashboards for LLM apps."
        },
        {
            "title": "TruLens",
            "url": "https://github.com/truera/trulens",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Adds evaluator suites and guardrail policies with audit trails."
        }
    ],
    "Evaluation & Evals": [
        {
            "title": "InstructEval",
            "url": "https://github.com/declare-lab/instruct-eval",
            "format": "Benchmark",
            "level": "Advanced",
            "est_time_hrs": 4,
            "why": "Provides task-specific evaluation datasets for instruction-tuned models."
        },
        {
            "title": "BigCode Evaluation Harness",
            "url": "https://github.com/bigcode-project/bigcode-evaluation-harness",
            "format": "Benchmark",
            "level": "Advanced",
            "est_time_hrs": 6,
            "why": "Standardizes code-generation metrics, baselines, and prompts."
        },
        {
            "title": "HELM",
            "url": "https://github.com/stanford-crfm/helm",
            "format": "Benchmark",
            "level": "Advanced",
            "est_time_hrs": 8,
            "why": "Offers a broad evaluation framework with safety and efficiency slices."
        }
    ],
    "Module 06 · Serving & Inference": [
        {
            "title": "vLLM",
            "url": "https://github.com/vllm-project/vllm",
            "format": "Inference engine",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Teaches paged KV-cache serving with OpenAI-compatible APIs."
        },
        {
            "title": "Text Generation Inference",
            "url": "https://github.com/huggingface/text-generation-inference",
            "format": "Inference server",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Demonstrates managed deployment patterns and streaming responses."
        },
        {
            "title": "TensorRT-LLM",
            "url": "https://github.com/NVIDIA/TensorRT-LLM",
            "format": "Toolkit",
            "level": "Advanced",
            "est_time_hrs": 6,
            "why": "Covers GPU-optimized inference, quantization, and serving pipelines."
        }
    ],
    "Module 07 · Performance & Cost Optimization": [
        {
            "title": "LLM-AWQ",
            "url": "https://github.com/mit-han-lab/llm-awq",
            "format": "Library",
            "level": "Advanced",
            "est_time_hrs": 4,
            "why": "Explains activation-aware quantization with benchmarks."
        },
        {
            "title": "LLM Perf Bench",
            "url": "https://github.com/mlc-ai/llm-perf-bench",
            "format": "Benchmark",
            "level": "Advanced",
            "est_time_hrs": 5,
            "why": "Provides scripts to profile latency, throughput, and memory under load."
        },
        {
            "title": "lit-gpt",
            "url": "https://github.com/Lightning-AI/litgpt",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Combines quantization, LoRA, and efficient serving recipes."
        }
    ],
    "Module 08 · Vector Search": [
        {
            "title": "FAISS",
            "url": "https://github.com/facebookresearch/faiss",
            "format": "Library",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Covers index types, IVF parameters, and GPU acceleration."
        },
        {
            "title": "pgvector",
            "url": "https://github.com/pgvector/pgvector",
            "format": "Extension",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Shows how to add vector search to PostgreSQL with hybrid queries."
        },
        {
            "title": "Milvus",
            "url": "https://github.com/milvus-io/milvus",
            "format": "Vector database",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Provides production-scale vector indexing with horizontal scaling."
        }
    ],
    "Module 09 · Security & Safety": [
        {
            "title": "OWASP Top 10 for LLM Applications",
            "url": "https://github.com/OWASP/www-project-top-10-for-large-language-model-applications",
            "format": "Guide",
            "level": "Intermediate",
            "est_time_hrs": 3,
            "why": "Defines common GenAI threat vectors and mitigations."
        },
        {
            "title": "NeMo Guardrails",
            "url": "https://github.com/NVIDIA/NeMo-Guardrails",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Implements policy-based guardrails for chat, RAG, and tool agents."
        },
        {
            "title": "Rebuff",
            "url": "https://github.com/protectai/rebuff",
            "format": "Toolkit",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Adds prompt injection detection with sandboxed tool execution."
        }
    ],
    "Security, Privacy & Compliance": [
        {
            "title": "AI Privacy Checklist",
            "url": "https://github.com/JariPesonen/AIPrivacyChecklist",
            "format": "Checklist",
            "level": "Intermediate",
            "est_time_hrs": 2,
            "why": "Outlines privacy impact questions for model development."
        },
        {
            "title": "AI Privacy Toolkit",
            "url": "https://github.com/IBM/ai-privacy-toolkit",
            "format": "Toolkit",
            "level": "Advanced",
            "est_time_hrs": 5,
            "why": "Provides differential privacy utilities and policy templates."
        },
        {
            "title": "AI Compliance Auditor",
            "url": "https://github.com/awsdataarchitect/ai-compliance-auditor",
            "format": "Toolkit",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Demonstrates automated checks for data retention and access controls."
        }
    ],
    "Module 10 · Governance & Compliance": [
        {
            "title": "EU AI Act Notes",
            "url": "https://github.com/daveshap/EU_AI_Act",
            "format": "Notes",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Summarizes obligations by risk class with quick reference tables."
        },
        {
            "title": "AI Governance Playbook",
            "url": "https://github.com/Neetu-kapoor/Ai-governance-playbook",
            "format": "Playbook",
            "level": "Intermediate",
            "est_time_hrs": 3,
            "why": "Provides governance charters, RACI templates, and stakeholder maps."
        },
        {
            "title": "AI Governance Risk & Compliance Docs",
            "url": "https://github.com/ahsan-141117/AI-Governance-Risk-Compliance-Documentation",
            "format": "Templates",
            "level": "Intermediate",
            "est_time_hrs": 3,
            "why": "Includes audit-ready checklists and policy samples for GenAI programs."
        }
    ],
    "Responsible AI & Governance": [
        {
            "title": "Awesome Responsible AI",
            "url": "https://github.com/AthenaCore/AwesomeResponsibleAI",
            "format": "Curated list",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Aggregates research, tooling, and case studies on responsible AI."
        },
        {
            "title": "Responsible AI Toolbox",
            "url": "https://github.com/microsoft/responsible-ai-toolbox",
            "format": "Toolkit",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Provides model interpretability, fairness, and error analysis dashboards."
        },
        {
            "title": "Awesome ML Model Governance",
            "url": "https://github.com/visenger/Awesome-ML-Model-Governance",
            "format": "Curated list",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Offers frameworks for lifecycle governance, approvals, and audits."
        }
    ],
    "Module 11 · Edge & Embedded": [
        {
            "title": "Edge AI for Beginners",
            "url": "https://github.com/microsoft/edgeai-for-beginners",
            "format": "Course",
            "level": "Intermediate",
            "est_time_hrs": 10,
            "why": "Covers hardware-aware deployment and telemetry for small models."
        },
        {
            "title": "ONNX Runtime GenAI",
            "url": "https://github.com/microsoft/onnxruntime-genai",
            "format": "Toolkit",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Shows how to run quantized models on CPU, GPU, and mobile."
        },
        {
            "title": "Phi-3 Mini Samples",
            "url": "https://github.com/Azure-Samples/Phi-3MiniSamples",
            "format": "Examples",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Demonstrates on-device SLM workflows with telemetry capture."
        }
    ],
    "Fine-Tuning & Adapters": [
        {
            "title": "PEFT",
            "url": "https://github.com/huggingface/peft",
            "format": "Library",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Implements LoRA, QLoRA, and other parameter-efficient adapters."
        },
        {
            "title": "QLoRA",
            "url": "https://github.com/artidoro/qlora",
            "format": "Toolkit",
            "level": "Advanced",
            "est_time_hrs": 6,
            "why": "Explains low-bit fine-tuning with reproducible scripts."
        },
        {
            "title": "Open Instruct",
            "url": "https://github.com/allenai/open-instruct",
            "format": "Dataset & recipes",
            "level": "Advanced",
            "est_time_hrs": 8,
            "why": "Provides supervised fine-tuning data pipelines and evaluation baselines."
        }
    ],
    "Multimodal": [
        {
            "title": "LLaVA",
            "url": "https://github.com/haotian-liu/LLaVA",
            "format": "Model",
            "level": "Advanced",
            "est_time_hrs": 6,
            "why": "Demonstrates visual instruction tuning and captioning workflows."
        },
        {
            "title": "Segment Anything",
            "url": "https://github.com/facebookresearch/segment-anything",
            "format": "Model",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Provides image segmentation building blocks for multimodal agents."
        },
        {
            "title": "Whisper",
            "url": "https://github.com/openai/whisper",
            "format": "Model",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Gives ready-to-run speech transcription pipelines for audio augmentation."
        }
    ],
    "LLMOps & MLE for Leaders": [
        {
            "title": "Awesome LLMOps",
            "url": "https://github.com/tensorchord/Awesome-LLMOps",
            "format": "Curated list",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Maps the vendor and open-source landscape for monitoring, deployment, and controls."
        },
        {
            "title": "LLM Engineer's Handbook",
            "url": "https://github.com/PacktPublishing/LLM-Engineers-Handbook",
            "format": "Book code",
            "level": "Intermediate",
            "est_time_hrs": 10,
            "why": "Includes project templates, release checklists, and operational metrics."
        },
        {
            "title": "OpenLLMetry",
            "url": "https://github.com/traceloop/openllmetry",
            "format": "Toolkit",
            "level": "Advanced",
            "est_time_hrs": 5,
            "why": "Instrument LLM apps with OpenTelemetry for observability and governance."
        }
    ],
    "Module 12 · Portfolio Patterns": [
        {
            "title": "GenAI Projects Portfolio",
            "url": "https://github.com/Ashleshk/GenAI-Projects-Portfolio",
            "format": "Examples",
            "level": "Beginner",
            "est_time_hrs": 3,
            "why": "Offers portfolio layout ideas with metrics-driven writeups."
        },
        {
            "title": "SAP BTP GenAI Starter Kit",
            "url": "https://github.com/SAP-samples/btp-genai-starter-kit",
            "format": "Starter kit",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Includes production-ready project scaffolds and deployment scripts."
        },
        {
            "title": "AWS Agentic AI Demos",
            "url": "https://github.com/aws-samples/sample-agentic-ai-demos",
            "format": "Examples",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Showcases multi-agent demos with architecture diagrams and KPIs."
        }
    ],
    "Architecture Patterns": [
        {
            "title": "Modular RAG",
            "url": "https://github.com/ThomasVitale/modular-rag",
            "format": "Guide",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Breaks RAG systems into repeatable modules with architecture diagrams."
        },
        {
            "title": "LLM Multi-Agent Patterns",
            "url": "https://github.com/VinZCodz/llm_multi_agent_patterns",
            "format": "Guide",
            "level": "Advanced",
            "est_time_hrs": 4,
            "why": "Explains when to choose planner/worker, reflection, and critique loops."
        },
        {
            "title": "Kernel Memory",
            "url": "https://github.com/microsoft/kernel-memory",
            "format": "Framework",
            "level": "Intermediate",
            "est_time_hrs": 5,
            "why": "Demonstrates memory-first architectures with ingestion, chunking, and retrieval services."
        }
    ],
    "Portfolio & Interview Prep": [
        {
            "title": "GenAI Interview Questions",
            "url": "https://github.com/a-tabaza/genai_interview_questions",
            "format": "Question bank",
            "level": "Intermediate",
            "est_time_hrs": 4,
            "why": "Curates current GenAI system design and scenario prompts."
        },
        {
            "title": "MLE/DS Interview Prep Guide",
            "url": "https://github.com/whaleonearth/MLE-DS-Interview-Prep-Guide",
            "format": "Guide",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Provides behavioral prompts, take-home tips, and technical refreshers."
        },
        {
            "title": "Andreis Interview Handbook",
            "url": "https://github.com/andreis/interview",
            "format": "Guide",
            "level": "Intermediate",
            "est_time_hrs": 8,
            "why": "Gives structured preparation checklists and storytelling frameworks."
        }
    ],
    "Community & Mentoring": [
        {
            "title": "Mentorship Guide Docs",
            "url": "https://github.com/mentorship-sponsorship/mentorship-guide-docs",
            "format": "Guide",
            "level": "Beginner",
            "est_time_hrs": 2,
            "why": "Explains how to run mentorship agreements and feedback loops."
        },
        {
            "title": "The Open Source Way Guidebook",
            "url": "https://github.com/theopensourceway/guidebook",
            "format": "Guidebook",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Details community-of-practice facilitation and governance tactics."
        },
        {
            "title": "Papers We Love",
            "url": "https://github.com/papers-we-love/papers-we-love",
            "format": "Community",
            "level": "Intermediate",
            "est_time_hrs": 6,
            "why": "Models an open study group with reading checklists and meetup templates."
        }
    ],
    "Tooling & Environment Setup": [
        {
            "title": "Miniforge",
            "url": "https://github.com/conda-forge/miniforge",
            "format": "Installer",
            "level": "Beginner",
            "est_time_hrs": 1,
            "why": "Lightweight conda distribution for reproducible Python environments."
        },
        {
            "title": "VS Code Dev Containers",
            "url": "https://github.com/microsoft/vscode-dev-containers",
            "format": "Templates",
            "level": "Intermediate",
            "est_time_hrs": 3,
            "why": "Provides ready-to-use containerized development environments."
        },
        {
            "title": "runpodctl",
            "url": "https://github.com/runpod/runpodctl",
            "format": "CLI",
            "level": "Intermediate",
            "est_time_hrs": 2,
            "why": "Helps manage on-demand GPU pods for cloud training and inference."
        }
    ]
}

with open("RESOURCES.json", "w") as f:
    json.dump(resources, f, indent=2)
