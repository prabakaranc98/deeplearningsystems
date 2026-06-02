# 01 - Kernels

## Goal

Understand how tensor operations map onto hardware: memory hierarchy,
parallelism, synchronization, occupancy, and numerical behavior.

## Build

- Vector add and reduction kernels in CUDA and Triton.
- Softmax forward pass.
- Layer norm forward and backward.
- Tiled matrix multiplication.
- A simple attention forward pass.
- One Pallas kernel if working in the JAX track.
- One CuTe DSL or CUTLASS inspection if studying NVIDIA tensor-core layouts.

Use Triton first for velocity, then inspect the equivalent CUDA concepts
where useful. Use Pallas when the surrounding framework is JAX. Use
CuTe DSL/CUTLASS when the target is NVIDIA matmul-like performance and
layout control.

## Measure

- Latency and throughput.
- Memory bandwidth versus compute utilization.
- Register pressure, occupancy, and block size effects.
- Numerical error versus PyTorch reference outputs.

## Teach Back

Explain why memory access patterns often dominate simple tensor kernels,
and why matmul-like kernels are shaped by tiling.

## Stretch

Optimize one kernel across at least three versions and keep the benchmark
results in `benchmarks/`.
