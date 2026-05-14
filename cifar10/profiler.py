"""PyTorch Profiler wrapper — identifies training bottlenecks."""
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class BottleneckOp:
    op:            str
    self_cuda_ms:  float
    pct_of_total:  float


@dataclass
class ProfilerReport:
    bottleneck_ops:            list[BottleneckOp] = field(default_factory=list)
    memory_bandwidth_util_pct: float              = 0.0
    l2_cache_hit_rate_pct:     Optional[float]    = None
    shared_mem_bank_conflicts: bool               = False
    total_cuda_ms:             float              = 0.0


def profile_training_loop(model: nn.Module, loader, n_steps: int = 20, device=None) -> ProfilerReport:
    if device is None:
        device = next(model.parameters()).device

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler    = torch.cuda.amp.GradScaler()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        step = 0
        for x, y in loader:
            if step >= n_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out  = model(x)
                loss = nn.functional.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1

    # Extract top ops by self_cuda_time_total
    key_avgs = prof.key_averages()
    ops = sorted(key_avgs, key=lambda e: e.self_cuda_time_total, reverse=True)

    total_cuda_us = sum(e.self_cuda_time_total for e in ops)
    total_cuda_ms = total_cuda_us / 1000.0

    bottlenecks = []
    for e in ops[:10]:
        if e.self_cuda_time_total == 0:
            continue
        bottlenecks.append(BottleneckOp(
            op=e.key,
            self_cuda_ms=round(e.self_cuda_time_total / 1000.0, 3),
            pct_of_total=round(100.0 * e.self_cuda_time_total / max(total_cuda_us, 1), 1),
        ))

    # Heuristic: flag bank conflicts if batch_norm or any elementwise op is unexpectedly high
    bn_pct   = sum(b.pct_of_total for b in bottlenecks if "batch_norm" in b.op.lower())
    top_pct  = bottlenecks[0].pct_of_total if bottlenecks else 0
    bank_conflicts = bn_pct > 15.0  # BN dominating often indicates memory layout issues

    # Memory bandwidth: rough estimate from bytes_self / time
    total_mem_bytes = sum(e.self_cpu_memory_usage for e in ops if e.self_cpu_memory_usage > 0)
    bw_util = min(100.0, (total_mem_bytes / max(total_cuda_ms / 1000.0, 1e-6)) / (272e9) * 100.0)

    return ProfilerReport(
        bottleneck_ops=bottlenecks,
        memory_bandwidth_util_pct=round(bw_util, 1),
        l2_cache_hit_rate_pct=None,  # requires nvml or nsys
        shared_mem_bank_conflicts=bank_conflicts,
        total_cuda_ms=round(total_cuda_ms, 2),
    )


if __name__ == "__main__":
    from cifar10.network import ResNet9
    from cifar10.data import get_cifar10_loaders

    device = torch.device("cuda")
    model  = ResNet9().to(device)
    train_loader, _ = get_cifar10_loaders(batch_size=512)

    print("Profiling 20 training steps...")
    report = profile_training_loop(model, train_loader, n_steps=20, device=device)

    print(f"\nTotal CUDA time (20 steps): {report.total_cuda_ms:.1f}ms")
    print(f"Bandwidth util estimate:     {report.memory_bandwidth_util_pct}%")
    print(f"Shared mem bank conflicts:   {report.shared_mem_bank_conflicts}")
    print(f"\nTop bottleneck ops:")
    for op in report.bottleneck_ops[:5]:
        print(f"  {op.op:<40} {op.self_cuda_ms:>8.2f}ms  ({op.pct_of_total}%)")
