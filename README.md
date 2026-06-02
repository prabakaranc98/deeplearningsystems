# Deep Learning Systems for Modern AI

This repo is a working substrate for learning, teaching, experimenting,
researching, hacking, and building deep learning systems.

The curriculum follows the systems stack from the machine upward:
kernels, runtimes, graphs and compilers, model systems, distributed
training, inference systems, and deployment.

## Curriculum Spine

| Stage | Area | What You Build |
| --- | --- | --- |
| 0 | Orientation and measurement | Reproducible environment, profiling notes, benchmark harness |
| 1 | Kernels | CUDA, Triton, Pallas, CuTe DSL kernels for reductions, softmax, layer norm, matmul, attention |
| 2 | Framework runtime and lower-level development | PyTorch/JAX runtime, custom ops, autograd, memory and stream behavior |
| 3 | Graphs and compilers | `torch.fx`, `torch.compile`, XLA, CUDA Graphs, fusion, shape experiments |
| 4 | Model systems | Transformer blocks, attention variants, quantization and memory tradeoffs |
| 5 | Distributed training | Data loading, DDP, FSDP/ZeRO, tensor/pipeline parallelism, NCCL, fault tolerance |
| 6 | Inference systems | KV cache, batching, paged attention, speculative decoding, serving latency |
| 7 | Deployment and operations | Networking, observability, reliability, cost, packaging, rollout notes |
| 8 | Research and hacking | Paper reproductions, ablations, open-ended system prototypes |

Start at [curriculum/README.md](curriculum/README.md).

Before starting the modules, read [PRIMER.md](PRIMER.md) for a single
map of the deep learning systems stack: frameworks, kernels, compilers,
data loading, distributed training, inference, networking, and fault
tolerance.

## Learning Loop

Every module should produce four artifacts:

1. A minimal implementation.
2. A benchmark or profiling trace.
3. A short explanation in `notes/`.
4. One experiment that changes a meaningful system variable.

This keeps the repo practical: learn the concept, build it, measure it,
teach it back, then push beyond the reference path.

## Repo Layout

```text
curriculum/   staged learning path and module briefs
src/          reusable code, kernels, runtime helpers, and systems components
benchmarks/   reproducible benchmark scripts and measurement notes
experiments/  exploratory work, ablations, paper reproductions, prototypes
notes/        teach-back writeups, diagrams, reading notes, design logs
```

## Current Starting Point

The first concrete thread is Triton/CUDA practice. The existing
`src/triton_intro.py` can become the first scratchpad for Stage 1, but
the long-term goal is to grow from isolated kernels into complete modern
AI training and inference systems.
