# ADR-0002: Choose Inference Stack

## Status

Proposed

## Context

Need to standardize on a serving stack for consistency and performance.

## Options

- NVIDIA Triton / TensorRT-LLM
- vLLM
- HF TGI

## Decision (TBD)

Select based on latency, throughput, cost, and ops fit.

## Consequences

- Enables shared tooling and benchmarks
