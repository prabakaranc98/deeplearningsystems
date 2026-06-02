# 02 - Runtime and Autograd

## Goal

Learn how deep learning frameworks execute tensor programs and connect
operators, memory, streams, custom kernels, and gradients.

## Build

- A PyTorch custom op around one kernel.
- A manual autograd function with forward and backward implementations.
- A JAX `jit`/`grad`/`vmap` comparison for the same computation.
- A memory experiment showing allocation, reuse, and peak usage.
- A stream or synchronization experiment that exposes hidden waiting.

## Measure

- Eager op overhead versus fused/custom paths.
- Forward and backward latency.
- Peak memory during training.
- Synchronization points introduced by measurement or data movement.

## Teach Back

Explain the difference between writing a fast kernel and integrating it
into a framework runtime correctly. Include the PyTorch eager/autograd
view and the JAX tracing/transformation view.

## Stretch

Replace one PyTorch operation in a toy model with a custom op and compare
training correctness, speed, and memory.
