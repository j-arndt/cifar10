"""SlapBack message builder — turns profiler output into targeted mutation instructions."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

from cifar10.profiler import ProfilerReport


# --------------------------------------------------------------------------- #
# Strategy hints — deterministic, not LLM-generated
# --------------------------------------------------------------------------- #

def get_strategy_hint(profiler: ProfilerReport) -> str:
    """Select a targeted mutation hint based on the top bottleneck."""
    if not profiler.bottleneck_ops:
        return (
            "No profiler data available. Focus on fusing the most compute-intensive layers "
            "(Conv2d + BatchNorm2d + ReLU) into a single CUDA kernel to eliminate intermediate "
            "memory reads/writes."
        )

    top_op   = profiler.bottleneck_ops[0].op.lower()
    top_pct  = profiler.bottleneck_ops[0].pct_of_total
    bw_util  = profiler.memory_bandwidth_util_pct
    l2_rate  = profiler.l2_cache_hit_rate_pct or 0.0
    conflicts = profiler.shared_mem_bank_conflicts

    if conflicts:
        return (
            f"BOTTLENECK: Shared memory bank conflicts detected in '{top_op}' ({top_pct}% of compute).\n"
            "FIX: Pad shared memory arrays by 1 element (4 bytes) per row to avoid conflicts. "
            "RTX 4060 has 32 banks x 4-byte words. Tile into 16x16 blocks. "
            "Example: `__shared__ float tile[16][16 + 1];` — the +1 eliminates bank conflicts."
        )

    if l2_rate > 0 and l2_rate < 50:
        return (
            f"BOTTLENECK: Low L2 cache hit rate ({l2_rate:.0f}%) in '{top_op}' ({top_pct}% of compute).\n"
            "FIX: Improve data locality by tiling the computation to fit in RTX 4060's 32MB L2 cache. "
            "Access memory in coalesced, sequential patterns. "
            "Use __ldg() for read-only data to route through texture cache."
        )

    if "batch_norm" in top_op or "bn" in top_op:
        return (
            f"BOTTLENECK: BatchNorm is {top_pct}% of compute — this means it's making a separate pass over activations.\n"
            "FIX: Fuse BatchNorm statistics (mean, variance) computation into the Conv kernel itself. "
            "Compute running stats in shared memory during the convolution pass. "
            "This eliminates the second memory read of the activation tensor."
        )

    if "data" in top_op or "copy" in top_op or "memcpy" in top_op or "h2d" in top_op:
        return (
            f"BOTTLENECK: Data transfer is {top_pct}% of compute.\n"
            "FIX: Overlap data transfer with GPU computation using CUDA streams. "
            "Use cudaMemcpyAsync and pin_memory to pipeline the next batch while the current one trains. "
            "Also: pre-normalize the dataset on GPU once at epoch start rather than per-batch."
        )

    if "sgd" in top_op or "adam" in top_op or "optim" in top_op or "grad" in top_op:
        return (
            f"BOTTLENECK: Optimizer step is {top_pct}% of compute.\n"
            "FIX: Fuse the optimizer update (weight -= lr * grad) directly into the backward pass "
            "kernel to avoid a separate read of all parameters. "
            "Use torch.optim.SGD with foreach=True as a first step, then write a custom fused kernel."
        )

    if bw_util > 85:
        return (
            f"BOTTLENECK: Memory bandwidth saturated at {bw_util:.0f}% utilization in '{top_op}' ({top_pct}%).\n"
            "FIX: Reduce memory traffic by operator fusion. "
            "Every separate kernel launch requires a full read+write of activations. "
            "Fusing N sequential pointwise ops into one kernel cuts memory traffic by N-1x."
        )

    # Default: generic kernel fusion
    return (
        f"BOTTLENECK: '{top_op}' is {top_pct}% of compute at {bw_util:.0f}% bandwidth utilization.\n"
        "FIX: Fuse this operator with adjacent elementwise operations (ReLU, scaling, bias add) "
        "into a single CUDA kernel. Each fusion eliminates one full read+write cycle of activations. "
        "Use register-level tiling to keep intermediate results in registers, not global memory."
    )


# --------------------------------------------------------------------------- #
# Slap message builders
# --------------------------------------------------------------------------- #

def build_compilation_slap(stderr: str, attempt_n: int) -> str:
    return f"""\
## COMPILATION REJECTED

ATTEMPT #{attempt_n} — FAILED AT: nvcc compiler

ERROR (verbatim):
```
{stderr[:3000]}
```

MUTATION REQUIRED:
The kernel failed to compile. Read the error above carefully.
Fix the syntax and resubmit a complete, compilable kernel.
Common issues:
- Missing semicolons or braces
- Undeclared variables or functions
- Incorrect template syntax
- sm_89 incompatible intrinsics (use __half for FP16, not half)
"""


def build_fitness_slap(
    accuracy: float,
    wall_time_s: float,
    target_time_s: float,
    profiler: ProfilerReport,
    attempt_n: int,
    champion_time: Optional[float] = None,
) -> str:
    delta        = wall_time_s - target_time_s
    champion_str = f"{champion_time:.3f}s" if champion_time else "none yet"
    hint         = get_strategy_hint(profiler)

    bottleneck_lines = ""
    if profiler.bottleneck_ops:
        bottleneck_lines = "\n".join(
            f"  {i+1}. {op.op:<40} {op.self_cuda_ms:>7.2f}ms  ({op.pct_of_total:.1f}%)"
            for i, op in enumerate(profiler.bottleneck_ops[:5])
        )

    return f"""\
## FITNESS REJECTED

ATTEMPT #{attempt_n}

RESULT:
  accuracy:    {accuracy:.4f}  (need >= 0.940)
  wall_time:   {wall_time_s:.3f}s  (need < {target_time_s:.1f}s)
  delta:       +{delta:.3f}s over target
  champion:    {champion_str}

PROFILER (top 5 CUDA ops):
{bottleneck_lines or '  (no profiler data)'}
  bandwidth:   {profiler.memory_bandwidth_util_pct:.1f}% utilization
  bank conf:   {'YES - CRITICAL' if profiler.shared_mem_bank_conflicts else 'no'}
  L2 hit rate: {f"{profiler.l2_cache_hit_rate_pct:.0f}%" if profiler.l2_cache_hit_rate_pct else 'unavailable'}

STRATEGY:
{hint}

MUTATION REQUIRED:
Submit a new kernel proposal addressing the bottleneck above.
Output ONLY valid JSON matching the schema. No prose before or after.
"""


def build_baseline_slap(baseline_time: float, target_time_s: float, skeleton_code: str) -> str:
    return f"""\
## INITIAL CHALLENGE

You are the MOAB CIFAR-10 Crucible optimization agent.

HARDWARE: RTX 4060 Mobile | sm_89 | 8GB GDDR6 | 272 GB/s bandwidth | 14 TFLOPS FP16
TARGET:   accuracy >= 0.940 AND wall_time < {target_time_s:.1f} second(s)
BASELINE: {baseline_time:.1f}s (naive PyTorch skeleton)

SKELETON CODE (your starting point):
```python
{skeleton_code[:4000]}
```

Your job: propose a CUDA kernel or Triton kernel that replaces one bottleneck in this training loop.
Measure impact. Iterate. Each proposal must be one JSON object matching the schema exactly.

Output ONLY valid JSON. No markdown, no prose, no explanation outside the JSON.
"""


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from cifar10.profiler import ProfilerReport, BottleneckOp

    mock_profiler = ProfilerReport(
        bottleneck_ops=[
            BottleneckOp("aten::conv2d",     4.21, 41.2),
            BottleneckOp("aten::batch_norm", 2.10, 20.5),
            BottleneckOp("aten::relu_",      1.05, 10.3),
            BottleneckOp("aten::add_",       0.80,  7.8),
            BottleneckOp("aten::sgd",        0.60,  5.9),
        ],
        memory_bandwidth_util_pct=78.0,
        l2_cache_hit_rate_pct=62.0,
        shared_mem_bank_conflicts=False,
        total_cuda_ms=10.21,
    )

    print("=" * 60)
    print("SLAP MESSAGE SAMPLES")
    print("=" * 60)

    print("\n--- Sample 1: Compilation failure ---")
    print(build_compilation_slap("error: expected ';' before '}' token\n   __global__ void k() {", attempt_n=1))

    print("\n--- Sample 2: Fitness failure (too slow) ---")
    print(build_fitness_slap(
        accuracy=0.941, wall_time_s=14.8, target_time_s=1.0,
        profiler=mock_profiler, attempt_n=7, champion_time=14.8,
    ))

    print("\n--- Sample 3: Near-miss with bank conflicts ---")
    bank_conflict_profiler = ProfilerReport(
        bottleneck_ops=[BottleneckOp("aten::conv2d", 3.1, 55.0)],
        memory_bandwidth_util_pct=91.0,
        l2_cache_hit_rate_pct=None,
        shared_mem_bank_conflicts=True,
        total_cuda_ms=5.6,
    )
    print(build_fitness_slap(
        accuracy=0.940, wall_time_s=1.42, target_time_s=1.0,
        profiler=bank_conflict_profiler, attempt_n=23, champion_time=1.42,
    ))
    print("=" * 60)
    print("Slap messages look good.")
