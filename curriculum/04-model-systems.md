# 04 - Model Systems

## Goal

Connect model architecture decisions to systems constraints: compute,
memory, communication, cache behavior, and numerical precision.

## Build

- A minimal transformer block.
- Attention variants such as causal, grouped-query, sliding-window, or sparse attention.
- Mixed-precision and quantization experiments.
- Memory accounting for parameters, activations, optimizer state, and KV cache.

## Measure

- FLOPs, memory footprint, and bandwidth pressure.
- Activation memory during training.
- Inference memory as sequence length grows.
- Accuracy or loss changes from precision changes.

## Teach Back

Explain how transformer architecture choices affect both training cost
and inference latency.

## Stretch

Implement a model-system tradeoff, such as lower precision, recomputation,
or attention sparsity, and measure the system impact.
