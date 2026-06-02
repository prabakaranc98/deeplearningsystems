# 05 - Distributed Training

## Goal

Learn how training scales beyond one accelerator and why communication,
memory layout, optimizer state, and fault tolerance become first-class
systems problems.

## Build

- Data-parallel training for a small model.
- Rank-aware data loading and sampling.
- Gradient accumulation and effective batch-size experiments.
- A comparison of data, tensor, and pipeline parallel ideas.
- Checkpointing and resume logic.
- A small communication benchmark.
- A fault-tolerance drill that kills and resumes a run.

## Measure

- Throughput per accelerator.
- Scaling efficiency.
- Communication time versus compute time.
- Data loading time per rank.
- Memory saved by sharding or recomputation.
- Checkpoint size and restore time.
- Lost work after failure and resume.

## Teach Back

Explain why faster single-GPU kernels do not automatically imply good
multi-GPU training performance.

## Stretch

Create a scaling report that compares one-GPU, two-GPU, and larger runs
when hardware is available.
