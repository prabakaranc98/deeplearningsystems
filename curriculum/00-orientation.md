# 00 - Orientation and Measurement

## Goal

Set up the habits that make systems work useful: reproducibility,
profiling literacy, controlled benchmarks, and clear experiment logs.

## Build

- A local environment note for Python, PyTorch, CUDA/Triton, and hardware.
- A small benchmark harness with warmup, repeated measurements, and summary stats.
- A profiling checklist for CPU time, GPU time, memory, and synchronization.

## Measure

- Wall-clock time versus accelerator time.
- Memory allocated, reserved, and transferred.
- Variance across repeated runs.

## Teach Back

Explain why naive timing is misleading for GPU workloads and what a good
microbenchmark must control.

## Stretch

Compare one operation across eager PyTorch, compiled PyTorch, and a
custom kernel.
