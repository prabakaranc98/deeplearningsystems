# 03 - Graphs and Compilers

## Goal

Understand how tensor programs become graphs, how graphs are transformed,
and how compilers use fusion, scheduling, and lowering to improve
execution.

## Build

- A `torch.fx` trace of a small model.
- A graph rewrite pass that fuses or replaces a pattern.
- A `torch.compile` experiment comparing eager and compiled execution.
- A JAX/XLA lowering or sharding experiment.
- A CUDA Graphs experiment for repeated static-shape execution.
- A shape-specialization experiment.

## Measure

- Compile time versus runtime savings.
- Graph breaks and their causes.
- Memory changes from fusion.
- CPU launch overhead with and without CUDA Graphs.
- Sensitivity to static and dynamic shapes.

## Teach Back

Explain what information a compiler needs in order to optimize a tensor
program and why dynamic Python can make that difficult.

## Stretch

Write a pass that recognizes a common model pattern and replaces it with
a faster implementation.
