# 06 - Inference and Serving

## Goal

Understand how trained models become interactive systems with latency,
throughput, memory, scheduling, and cost constraints.

## Build

- A minimal model-serving loop.
- Continuous or dynamic batching experiments.
- KV-cache accounting for decoder inference.
- Token streaming and cancellation behavior.
- A vLLM or TensorRT-LLM comparison when hardware and model support allow it.
- A small load test.

## Measure

- Time to first token.
- Inter-token latency.
- Requests per second.
- Batch size versus latency.
- Memory per active request.
- Prefix length, output length, and KV-cache pressure.

## Teach Back

Explain why inference is a scheduling and memory-management problem, not
only a model-forward-pass problem.

## Stretch

Prototype speculative decoding, paged KV cache behavior, or request
routing across model variants.
