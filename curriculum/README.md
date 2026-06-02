# Curriculum Path

The path is ordered by dependency, not difficulty. Kernel work teaches
the machine. Runtime and graph work teaches how frameworks organize that
machine. Distributed training and inference teach how modern AI systems
turn those pieces into useful infrastructure.

Read the root [primer](../PRIMER.md) first. It gives the framework and
systems map: CUDA, Triton, Pallas, CuTe DSL, PyTorch, JAX, XLA,
`torch.compile`, CUDA Graphs, data loading, distributed training,
networking, inference serving, and fault tolerance.

## Path

| Stage | Module | Primary Question |
| --- | --- | --- |
| 0 | [Orientation and Measurement](00-orientation.md) | How do we make experiments reproducible and measurable? |
| 1 | [Kernels](01-kernels.md) | How does tensor math map to hardware? |
| 2 | [Runtime and Autograd](02-runtime-and-autograd.md) | How do frameworks execute tensor programs? |
| 3 | [Graphs and Compilers](03-graphs-and-compilers.md) | How do tensor programs become optimized execution graphs? |
| 4 | [Model Systems](04-model-systems.md) | How do architecture choices create systems constraints? |
| 5 | [Distributed Training](05-distributed-training.md) | How do we train models that do not fit on one accelerator? |
| 6 | [Inference and Serving](06-inference-serving.md) | How do we serve models under latency, throughput, and cost constraints? |
| 7 | [Deployment and Operations](07-deployment-and-ops.md) | How do we run these systems repeatedly and safely? |
| 8 | [Research and Hacking](08-research-hacking.md) | How do we turn papers and ideas into measured prototypes? |

## Standard Module Contract

Each module should end with:

- `implementation`: code that runs locally or on a target accelerator.
- `benchmark`: latency, throughput, memory, or scaling measurements.
- `explanation`: a short note that teaches the core idea.
- `experiment`: one change that tests a hypothesis.

Use this rhythm for every topic, from a single softmax kernel to a
multi-node training job.
