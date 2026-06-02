# Primer: Deep Learning Systems for Modern AI

This primer gives the full map before starting the curriculum, but it is
best read as a walkthrough, not as a glossary. The field is about making
tensor programs run correctly, fast, cheaply, and reliably across real
hardware.

The useful mental model is not "learn one framework." It is a stack:

```text
hardware and interconnect
  -> kernels and libraries
  -> framework runtime and autograd
  -> graphs and compilers
  -> data pipeline
  -> training loop
  -> distributed training
  -> inference serving
  -> deployment and operations
```

Each layer exists because the layer above eventually hits a bottleneck:
too much memory movement, too many kernel launches, poor fusion, slow
data loading, communication overhead, KV-cache pressure, failures, or
cost.

## The Story: Follow One Batch

Imagine a simple transformer training run. You wrote a PyTorch model,
picked a dataset, and pressed run. At first it feels like "Python trains
a model," but the real system is a chain of smaller systems.

First, examples have to arrive. Files are read from disk or object
storage. Text may be tokenized, images may be decoded, examples are
batched, and CPU workers try to prepare the next batch before the GPU
finishes the current one. If this part is slow, the GPU waits. The fix is
not a better attention kernel; it is a better input pipeline.

Next, the batch moves from host memory to accelerator memory. This is
where pinned memory, non-blocking copies, device placement, and hidden
synchronization start to matter. A single accidental `.item()` or CPU
copy can force the program to wait for GPU work that should have stayed
asynchronous.

Then the framework runtime takes over. PyTorch or JAX does not execute
"a model" as one magical thing. It launches many tensor operations:
matmuls, layer norms, softmaxes, dropout, attention, optimizer updates.
Each operation eventually becomes one or more kernels or calls into a
vendor library. Autograd records enough information to run the backward
pass. The runtime is where correctness, debuggability, memory lifetime,
and scheduling meet.

Now look inside one operation, such as softmax or attention. The kernel
has to decide how data is tiled across blocks, warps, registers, shared
memory, and HBM. If the kernel repeatedly reads and writes large
temporary tensors, performance falls apart. This is why people write
custom CUDA, Triton, Pallas, or CUTLASS/CuTe kernels: not because custom
code is inherently better, but because a particular shape, fusion, or
memory pattern needs control the default framework path does not provide.

Once the eager version is correct, the compiler path becomes interesting.
`torch.compile`, XLA, and related compiler stacks try to see more of the
program at once. If they can capture stable computation, they can fuse
ops, remove temporary tensors, choose better kernels, and reduce CPU
launch overhead. CUDA Graphs attack a related problem: if the same GPU
work repeats with stable shapes, capture it once and replay it cheaply.

The backward pass repeats the same story under tighter memory pressure.
Activations saved from the forward pass compete with parameters,
gradients, optimizer state, and temporary buffers. Activation
checkpointing trades extra compute for lower memory. Mixed precision
trades numeric range and accuracy risk for speed and capacity. The
training loop becomes a resource accounting problem.

On one GPU, the question is usually: "Where is time or memory going?" On
many GPUs, the question changes: "Who needs to talk to whom, how often,
and over what link?" DDP replicates the model and all-reduces gradients.
FSDP/ZeRO shards model or optimizer state so bigger models fit. Tensor
parallelism splits work inside layers. Pipeline parallelism splits model
depth. These are not interchangeable tricks. Each one moves a different
kind of data at a different frequency.

At cluster scale, failures become normal. A training job needs to save
enough state to resume: model, optimizer, scheduler, RNG, dataloader
position, and sharding metadata. Checkpointing is not just storage; it is
part of the performance and reliability system. If checkpointing takes
too long, wastes bandwidth, or cannot restore the exact distributed
layout, the system is fragile.

After training, inference changes the game. There is no backward pass,
but now requests arrive at unpredictable times. The system has to batch
requests, stream tokens, manage KV cache, handle cancellations, and keep
tail latency under control. Serving a transformer is less like "run the
model once" and more like running a scheduler around a memory-hungry
decode loop.

That is the whole map in one story:

```text
data arrives
  -> batch moves to device
  -> framework launches ops
  -> kernels do the math
  -> compiler may fuse and schedule
  -> backward and optimizer consume memory
  -> distributed training adds communication
  -> checkpointing adds recovery
  -> inference turns the model into a live service
```

Use the rest of this primer as the expanded map for that story.

## The Three Questions

When a deep learning system is slow, expensive, or unreliable, start with
three questions:

1. **Where is the bottleneck?**
   CPU input pipeline, GPU compute, GPU memory bandwidth, kernel launch
   overhead, inter-GPU communication, network, storage, or serving
   scheduler?
2. **What is the resource tradeoff?**
   Compute for memory, memory for latency, latency for throughput,
   precision for capacity, communication for model size, checkpoint time
   for fault tolerance?
3. **What is the simplest tool that attacks that bottleneck?**
   DataLoader tuning, a fused kernel, `torch.compile`, CUDA Graphs, DDP,
   FSDP, tensor parallelism, vLLM, or a deployment/observability fix?

This prevents the common mistake: reaching for an advanced tool before
proving the problem it solves.

## Frontier Lab Skill Signals

Public research-engineering and systems roles from OpenAI, Anthropic, and
Google DeepMind point to the same pattern. Frontier labs value people who
can turn ambiguous research ideas into measured, scalable systems. The
job title may say research engineer, performance engineer, data
infrastructure engineer, distributed systems engineer, or inference
engineer, but the underlying skill shape is similar.

This is not a hiring rubric. It is a learning signal for this repo.

### What They Seem To Need

| Skill Area | What It Means | How To Learn It Here |
| --- | --- | --- |
| Research engineering | Bridge theory and implementation; design experiments, build prototypes, scale the promising ones. | Reproduce a paper, implement the core idea, measure it, then write what changed and why. |
| Measurement and evals | Build evals, graders, environments, and methodology that are reliable under noisy model behavior. | Create a small benchmark suite with variance, failure cases, and regression thresholds. |
| Post-training and agents | Understand RL, RLHF/RLAIF, tool use, coding agents, computer-use tasks, and long-horizon behavior. | Build toy RL environments, agent task harnesses, and automatic graders before touching huge models. |
| GPU performance | Understand kernels, memory bandwidth, tensor cores, profiling, fusion, low precision, and attention kernels. | Write CUDA/Triton kernels, profile with Nsight or PyTorch tools, compare against framework baselines. |
| Framework internals | Know PyTorch/JAX runtime behavior, autograd, custom ops, `torch.compile`, XLA, and graph breaks. | Replace a framework op with a custom op, compile the model, and explain where compilation helps or fails. |
| Distributed training | Scale across GPUs with DDP, FSDP/ZeRO, tensor parallelism, pipeline parallelism, collectives, and checkpointing. | Run single-GPU first, then DDP, then sharded training; measure MFU, communication, memory, and resume correctness. |
| Data infrastructure | Build dataset APIs, sharding, validation, reproducibility, inspection tools, and fast loading at GPU scale. | Make a dataloader benchmark, add rank-aware sampling, and test whether data stalls slow global training. |
| Inference systems | Optimize serving across kernels, memory, network, batching, KV cache, and distributed execution. | Compare a simple serving loop with vLLM-like batching concepts; measure time to first token and inter-token latency. |
| Reliability and observability | Detect bad nodes, stragglers, bottlenecks, failed jobs, and correctness drift in large systems. | Add traces, logs, health checks, checkpoint drills, and failure/restart experiments. |
| Safety and responsibility | Treat capability, alignment, misuse, robustness, and evaluation as part of the system, not as afterthoughts. | Add eval notes for failure modes, misuse boundaries, and model behavior that would block deployment. |

### The Pattern

At frontier scale, the important work often sits between categories:

- A new model idea becomes useful only if it can be trained efficiently.
- A faster kernel matters only if it improves end-to-end throughput.
- An eval matters only if it is reliable enough to steer a training run.
- A data pipeline matters only if it prevents GPU-scale stalls and keeps
  experiments reproducible.
- A distributed training system matters only if it survives failures and
  gives researchers usable feedback.
- An inference optimization matters only if it improves latency,
  throughput, quality, or cost in the real serving path.

The lesson for this repo: every project should connect an idea to a
measurement and a system constraint. Do not only implement softmax. Ask:
how fast is it, what memory path limits it, how does it integrate with
autograd, can the compiler fuse around it, does it matter in a model, and
what breaks when it scales?

### Essential Frontier-Lab Learning Projects

Use these as north-star projects after the early curriculum modules:

1. **Microkernel to model path**
   Implement softmax or layer norm in Triton, verify correctness, profile
   it, wrap it as a PyTorch op, and measure whether a tiny transformer
   block improves end-to-end.
2. **Compiler and graph path**
   Take an eager PyTorch model, run `torch.compile`, inspect graph breaks,
   measure compile overhead versus runtime savings, and document when
   CUDA Graphs would help.
3. **Data-at-scale path**
   Build a dataset interface with sharding, validation, deterministic
   resume, and a benchmark that shows whether the GPU is input-starved.
4. **Distributed training path**
   Train a small transformer with DDP, then FSDP/ZeRO-style sharding.
   Track throughput, memory, communication, checkpoint size, and resume
   correctness.
5. **Post-training and eval path**
   Build a toy agent/RL environment with an automatic grader. Measure
   reward variance, failure modes, and whether the eval actually
   distinguishes model behavior.
6. **Inference path**
   Serve a small decoder model. Measure prefill latency, decode latency,
   KV-cache memory, batching effects, and cancellation/streaming behavior.
7. **Reliability path**
   Add observability and failure drills: kill a worker, resume from a
   checkpoint, detect a slow data worker, and report lost work.

If you can complete these projects with clear measurements and readable
writeups, you will have learned the practical core that frontier labs
keep signaling: research ideas, scalable implementation, performance,
evaluation, and reliability are one connected discipline.

## 1. Hardware and Interconnect

What matters:

- GPU/accelerator compute: SMs, warps, tensor cores, vector units.
- Memory hierarchy: registers, shared memory/SRAM, L2, HBM, host memory.
- Interconnect: PCIe, NVLink, InfiniBand/RDMA, Ethernet.
- System shape: single GPU, multi-GPU node, multi-node cluster.

Use this layer when you ask:

- Is this workload compute-bound, memory-bound, or launch-bound?
- Is the bottleneck inside one GPU or between GPUs?
- Is the model too large, the batch too small, or the sequence too long?

Why it is needed:

Deep learning performance is usually constrained by movement: moving
bytes through memory hierarchy, across GPU links, across network, or
between CPU and GPU. Good systems work starts by measuring where movement
and waiting occur.

## 2. Kernel Development

Kernels are the small programs that run on accelerators. Framework ops
are built from kernels.

| Tool | Use It When | Why |
| --- | --- | --- |
| CUDA C++ | You need maximum NVIDIA control, full access to hardware features, or production library work. | Lowest-level mainstream path; exposes threads, blocks, shared memory, streams, events, and advanced GPU features. |
| Triton | You need custom GPU kernels quickly from Python, especially fused elementwise ops, reductions, matmul-like tiling, attention pieces, or research kernels. | Much faster iteration than raw CUDA while still exposing block-level memory and tiling ideas. |
| Pallas | You are in JAX and need custom kernels for GPU or TPU. | Keeps custom kernels inside JAX's tracing, transformation, and XLA-oriented world. |
| CUTLASS / CuTe DSL | You need serious NVIDIA tensor-core performance, layout control, GEMM/conv/attention building blocks, or Blackwell/Hopper-specific features. | More hardware-specific than Triton; useful when matmul-like performance and layout algebra matter. |
| Vendor libraries: cuBLAS, cuDNN, NCCL | The operation is standard and already heavily optimized. | Do not rewrite kernels that vendor libraries already do well unless you need a new fusion, shape, datatype, or memory pattern. |

Preferred learning order:

1. Learn CUDA concepts: threads, blocks, warps, memory hierarchy,
   streams, events, occupancy, bandwidth.
2. Write Triton kernels because iteration is fast.
3. Inspect CUDA/CUTLASS/CuTe when Triton hides too much or when tensor
   core layout matters.
4. Use Pallas when the surrounding system is JAX.

Typical projects:

- vector add;
- reductions;
- softmax;
- layer norm;
- matmul;
- FlashAttention-style tiled attention;
- fused optimizer update;
- quantized matmul or dequantization kernel.

## 3. Framework Runtime: PyTorch and JAX

Frameworks are where models are usually expressed.

| Framework | Use It When | Why |
| --- | --- | --- |
| PyTorch | Default choice for research, teaching, debugging, model hacking, and most ecosystem work. | Eager execution is easy to inspect; ecosystem is broad; distributed training and serving integrations are mature. |
| JAX | You want functional transformations, strong compiler orientation, SPMD-style scaling, TPU work, or clean mathematical programs. | `jit`, `vmap`, autodiff, sharding, and XLA make whole-program optimization and distributed array programming more explicit. |
| TensorFlow | You are maintaining an existing production stack or using infrastructure built around it. | Still important in production, but less likely to be the default for a new systems-learning repo. |

What to learn in PyTorch:

- tensors, autograd, modules, optimizers;
- CUDA device movement and synchronization;
- `torch.utils.data.DataLoader`;
- AMP/mixed precision;
- custom `autograd.Function`;
- C++/CUDA extensions or Python custom ops;
- `torch.compile`;
- `torch.distributed`.

What to learn in JAX:

- pure functions and pytrees;
- `jit`, `grad`, `vmap`;
- sharding and distributed arrays;
- `shard_map`/SPMD ideas;
- XLA lowering and compilation;
- Pallas for custom kernels.

## 4. Graphs, Compilers, and CUDA Graphs

Eager execution is flexible but can leave performance on the table. A
compiler tries to see more of the program at once, then fuse, schedule,
specialize, and lower it.

| Tool | Use It When | Why |
| --- | --- | --- |
| `torch.fx` | You want to inspect or rewrite PyTorch computation graphs. | Good for learning graph capture and transformation. |
| `torch.compile` / TorchDynamo / Inductor | You want PyTorch speedups with minimal code changes. | Captures PyTorch programs, lowers to optimized kernels, often Triton-backed on GPU. |
| XLA | You are in JAX, TensorFlow, or another XLA-backed path and want compiler-optimized execution across accelerators. | Mature compiler path for whole-program tensor optimization and sharding. |
| MLIR | You want compiler infrastructure or research-level lowering work. | Common compiler substrate for building domain-specific compiler pipelines. |
| CUDA Graphs | You have repeated work with stable shapes and want to reduce CPU launch overhead. | Captures a sequence of GPU work and replays it with lower launch overhead. Useful in small-batch inference or repeated training steps. |

Use compilers when:

- Python overhead is visible;
- many small ops should be fused;
- shapes are stable enough;
- launch overhead matters;
- memory traffic can be reduced by fusion;
- you want automatic scheduling or autotuning.

Avoid relying only on compilers when:

- graph breaks dominate;
- shapes are highly dynamic;
- custom Python control flow is central;
- a hand-designed kernel is clearly needed;
- debugging correctness is still the main task.

## 5. Data Loading and Input Pipelines

Training can be GPU-bound or input-bound. If the GPU is waiting for
batches, kernel work will not help.

What matters:

- dataset format and storage locality;
- CPU decode/tokenization/augmentation;
- multiprocessing workers;
- pinned memory for faster host-to-GPU transfer;
- prefetching;
- sharding data across ranks;
- deterministic sampling and resume state;
- streaming large datasets.

Use `DataLoader` first in PyTorch. Tune `num_workers`, `pin_memory`,
`prefetch_factor`, `persistent_workers`, and the collate function. For
large web-scale data, study streaming formats and rank-aware sharding.

Typical projects:

- measure GPU utilization with slow versus fast data loading;
- build a rank-aware distributed sampler;
- compare image decode/tokenization in the training process versus
  preprocessed data;
- checkpoint dataloader position for resume.

## 6. Training Systems

A training system is more than `loss.backward()`.

Core pieces:

- model forward and backward;
- optimizer and scheduler;
- precision policy: fp32, tf32, bf16, fp16, fp8;
- activation checkpointing/recomputation;
- gradient accumulation;
- logging and profiling;
- checkpointing;
- validation and evaluation;
- failure recovery.

Use this layer to answer:

- What consumes memory: parameters, gradients, activations, optimizer
  state, temporary buffers, or dataloader queues?
- What consumes time: forward, backward, optimizer, data, communication,
  checkpointing, or evaluation?
- What can be traded: compute for memory, memory for latency, latency for
  throughput, precision for speed?

## 7. Distributed and Parallel Training

Distributed training starts simple, then becomes a composition problem.

| Technique | Use It When | Main Cost |
| --- | --- | --- |
| DDP / data parallel | Model fits on one GPU, but you want more throughput. | Gradient all-reduce. |
| FSDP / ZeRO | Model, gradients, or optimizer state do not fit comfortably on each GPU. | More communication and more complex checkpointing. |
| Tensor parallelism | Individual layers are too large or you need lower per-GPU memory/latency. | Frequent collectives inside layers; needs fast interconnect. |
| Pipeline parallelism | Model depth is too large for one GPU group. | Pipeline bubbles, scheduling complexity, activation movement. |
| Sequence/context parallelism | Sequence length dominates memory or attention cost. | Communication around sequence-dependent ops. |
| Expert parallelism | Mixture-of-experts routing creates sparse expert computation. | All-to-all routing and load imbalance. |
| 3D parallelism | Large model training needs data + tensor + pipeline parallelism together. | Placement, scheduling, communication overlap, checkpoint complexity. |

The practical default:

1. Single GPU baseline.
2. DDP when the model fits.
3. FSDP/ZeRO when memory is the blocker.
4. Tensor or pipeline parallelism when model dimensions force it.
5. Combine methods only after measuring the bottleneck.

Networking and communication:

- NCCL is the core NVIDIA collective communication library used under
  many distributed training stacks.
- Important collectives: all-reduce, all-gather, reduce-scatter,
  broadcast, all-to-all.
- Fast intra-node links help tensor parallelism.
- Fast inter-node networking matters for large data parallel jobs and
  all-to-all-heavy workloads.

Fault tolerance:

- save model, optimizer, scheduler, RNG, dataloader, and distributed
  sharding state;
- use distributed checkpointing for sharded models;
- test resume before trusting long runs;
- use elastic launch/control-plane tools when jobs can lose workers or
  change membership;
- measure checkpoint time because checkpointing itself can become a
  systems bottleneck.

## 8. Inference and Serving Systems

Training optimizes tokens learned per dollar. Inference optimizes useful
tokens served per dollar under latency constraints.

What matters:

- prefill latency;
- decode latency;
- time to first token;
- inter-token latency;
- throughput;
- batching policy;
- KV-cache memory;
- quantization;
- routing;
- cancellation;
- streaming;
- tail latency;
- autoscaling and cost.

| Tool / Idea | Use It When | Why |
| --- | --- | --- |
| vLLM | You want flexible high-throughput LLM serving with continuous batching and paged KV-cache behavior. | Strong default for open LLM serving experiments. |
| TensorRT-LLM | You want NVIDIA-focused optimized inference engines and production-oriented performance. | Useful when deployment is NVIDIA-specific and engine build/runtime tradeoffs are acceptable. |
| CUDA Graphs | You have stable decode shapes or repeated execution and CPU launch overhead matters. | Reduces launch overhead. |
| Quantization | Memory bandwidth or capacity is limiting serving. | Lower precision can improve throughput/cost if quality holds. |
| Speculative decoding | Decode is the bottleneck and a draft model or draft path is available. | Can reduce latency by verifying multiple proposed tokens per step. |

Training and inference are different systems. A training stack cares
about backward pass, optimizer state, and gradient communication.
Inference cares about scheduling, cache memory, batching, streaming, and
tail latency.

## 9. Deployment and Operations

An operable system needs:

- environment reproducibility;
- versioned models and configs;
- health checks;
- logs and traces;
- latency and memory metrics;
- load tests;
- rollback path;
- capacity planning;
- cost tracking;
- security and data-handling policy.

Use containers, cluster schedulers, and service frameworks only after the
single-process behavior is measured. Production tools hide problems well;
profiling should happen before abstraction.

## 10. What To Use First

For this repo, the preferred default path is:

1. **PyTorch for model and training baselines.**
   It is easiest to debug and is the common practical baseline.
2. **Triton for first custom kernels.**
   It gives fast iteration while teaching memory tiling and fusion.
3. **CUDA concepts alongside Triton.**
   You need CUDA vocabulary to understand what Triton and PyTorch are
   doing.
4. **`torch.compile` after eager code is correct.**
   Compile only after you have correctness tests and benchmarks.
5. **DDP before FSDP/ZeRO.**
   Use the simplest distributed method that solves the actual bottleneck.
6. **FSDP/ZeRO before tensor/pipeline parallelism.**
   Sharding optimizer/model state is usually simpler than redesigning the
   model's parallel layout.
7. **vLLM for first LLM serving experiments.**
   It directly exposes modern inference concerns: batching, KV cache,
   streaming, and throughput.
8. **JAX/Pallas as a second framework track.**
   Learn it when compiler-first programming, sharding, or TPU-style
   systems are part of the goal.
9. **CuTe DSL/CUTLASS after basic kernels.**
   Use it when tensor-core layout and NVIDIA-specific matmul performance
   become the object of study.

## 11. The Curriculum Path After This Primer

```text
00 primer and measurement
01 CUDA concepts and Triton kernels
02 PyTorch runtime, autograd, and custom ops
03 JAX, XLA, Pallas, and SPMD mental models
04 graph capture, torch.compile, CUDA Graphs, and compiler tradeoffs
05 data loading, input pipelines, and training loop performance
06 single-node multi-GPU: DDP, FSDP, NCCL, checkpointing
07 multi-node training: networking, launchers, elasticity, fault tolerance
08 model parallelism: tensor, pipeline, sequence/context, expert parallelism
09 inference systems: KV cache, batching, paged attention, quantization
10 deployment and operations: observability, reliability, cost
11 research hacking: reproduce papers, ablate, benchmark, teach back
```

The sequence is deliberate: measure first, then make one GPU fast, then
make the framework integration correct, then scale, then serve.

## 12. Reference Anchors

These are the public anchors used to shape this map:

- OpenAI Triton introduction: https://openai.com/index/triton/
- OpenAI Frontier Evals & Environments role: https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/
- OpenAI Platform Systems role: https://openai.com/careers/software-engineer-platform-systems-san-francisco/
- OpenAI Training Performance Engineer role: https://openai.com/careers/training-performance-engineer-san-francisco/
- OpenAI Data Infrastructure role: https://openai.com/careers/software-engineer-data-infrastructure-research-san-francisco/
- OpenAI Inference GPU Enablement role: https://openai.com/careers/software-engineer-inference-amd-gpu-enablement-san-francisco/
- Anthropic Research Engineer, Reinforcement Learning role: https://www.anthropic.com/careers/jobs/4613568008
- Anthropic GPU Performance Engineer role: https://www.anthropic.com/careers/jobs/4926227008
- Google DeepMind careers role descriptions: https://deepmind.google/careers/
- Triton documentation: https://triton-lang.org/
- CMU 10-414/714 Deep Learning Systems: https://dlsyscourse.org/lectures/
- Stanford CS149 Parallel Computing: https://gfxcourses.stanford.edu/cs149/
- Stanford CS329S Machine Learning Systems Design: https://stanford-cs329s.github.io/syllabus.html
- Stanford CS349D AI Inference Infrastructure: https://web.stanford.edu/class/cs349d/
- PyTorch Distributed Overview: https://docs.pytorch.org/tutorials/beginner/dist_overview.html
- PyTorch `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch Data Loading: https://docs.pytorch.org/docs/stable/data.html
- PyTorch Elastic Agent: https://docs.pytorch.org/docs/main/elastic/agent.html
- PyTorch Distributed Checkpoint: https://docs.pytorch.org/docs/main/distributed.checkpoint.html
- JAX Pallas: https://docs.jax.dev/en/latest/pallas/index.html
- JAX `pmap` and SPMD notes: https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html
- OpenXLA/XLA: https://github.com/openxla/xla
- NVIDIA CUDA Graphs: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- NVIDIA NCCL: https://docs.nvidia.com/deeplearning/nccl/
- NVIDIA CuTe DSL / CUTLASS: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html
- NVIDIA TensorRT-LLM: https://docs.nvidia.com/tensorrt-llm/
- vLLM: https://docs.vllm.ai/
