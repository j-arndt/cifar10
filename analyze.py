"""Post-run analysis — load annealing log, print trajectory, write RESULTS.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _sparkline(values: list[float], width: int = 40) -> str:
    """Simple ASCII sparkline from a list of floats."""
    if not values:
        return ""
    bars = " _.-~*#@"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    chars = [bars[min(int((v - mn) / rng * (len(bars) - 1)), len(bars) - 1)] for v in values]
    # Truncate to width
    if len(chars) > width:
        step = len(chars) / width
        chars = [chars[int(i * step)] for i in range(width)]
    return "".join(chars)


def load_log(path: str = "logs/annealing_log.jsonl") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return sorted(records, key=lambda r: r.get("attempt_n", 0))


def analyze(log_path: str = "logs/annealing_log.jsonl") -> None:
    records = load_log(log_path)

    if not records:
        print("No annealing log found. Run the orchestrator first.")
        return

    total  = len(records)
    passed = [r for r in records if r.get("passed")]
    times  = [r["wall_time_s"] for r in records if r.get("wall_time_s", 9999) < 9999]
    accs   = [r["accuracy"] for r in records if r.get("accuracy", 0) > 0]

    print("=" * 65)
    print("MOAB CIFAR-10 Crucible — Post-Run Analysis")
    print("=" * 65)
    print(f"\nTotal attempts:   {total}")
    print(f"Passed (target):  {len(passed)}")
    print(f"Best wall time:   {min(times):.3f}s" if times else "Best wall time:   N/A")
    print(f"Best accuracy:    {max(accs):.4f}" if accs else "Best accuracy:    N/A")

    # Wall-time trajectory sparkline
    if times:
        print(f"\nWall-time trajectory ({len(times)} data points):")
        print(f"  {_sparkline(times)}")
        print(f"  {min(times):.2f}s {'':>38} {max(times):.2f}s")

    # Top 5 improving kernels
    print("\nTop 5 kernel attempts by wall time:")
    top5 = sorted(
        [r for r in records if r.get("wall_time_s", 9999) < 9999],
        key=lambda r: r["wall_time_s"]
    )[:5]
    for i, r in enumerate(top5, 1):
        print(f"  {i}. attempt={r.get('attempt_n','?'):<5} "
              f"time={r.get('wall_time_s', 0):.3f}s  "
              f"acc={r.get('accuracy', 0):.4f}  "
              f"type={r.get('kernel_type','?')}")

    # Gate failure breakdown
    gate_counts: dict[str, int] = {}
    for r in records:
        gf = r.get("gate_failed") or ("fitness" if not r.get("passed") else "PASS")
        gate_counts[gf] = gate_counts.get(gf, 0) + 1
    print("\nRejection breakdown:")
    for gate, count in sorted(gate_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(int(count / max(total, 1) * 40), 40)
        print(f"  {gate:<20} {count:>4}  {bar}")

    # Write RESULTS.md
    _write_results_md(records, top5)


def _write_results_md(records: list[dict], top5: list[dict]) -> None:
    times  = [r["wall_time_s"] for r in records if r.get("wall_time_s", 9999) < 9999]
    accs   = [r["accuracy"] for r in records if r.get("accuracy", 0) > 0]
    passed = [r for r in records if r.get("passed")]

    champion = top5[0] if top5 else {}

    lines = [
        "# MOAB CIFAR-10 Crucible — Results",
        "",
        "## Hardware",
        "- GPU: RTX 4060 Mobile (Ada Lovelace, sm_89)",
        "- VRAM: 8 GB GDDR6 @ 272 GB/s",
        "- Agent: Qwen2.5-Coder-7B-Instruct (GGUF, Q4_K_M)",
        "",
        "## Objective",
        "- Target accuracy: ≥ 94.0%",
        "- Target wall time: < 1.0 second",
        "",
        "## Results",
        f"- Total attempts: {len(records)}",
        f"- Passed (both conditions): {len(passed)}",
        f"- Best wall time: {min(times):.3f}s" if times else "- Best wall time: N/A",
        f"- Best accuracy: {max(accs):.4f}" if accs else "- Best accuracy: N/A",
        "",
        "## Champion Kernel",
        f"- Attempt #{champion.get('attempt_n', '?')}",
        f"- Wall time: {champion.get('wall_time_s', 'N/A'):.3f}s" if champion.get("wall_time_s") else "- Wall time: N/A",
        f"- Accuracy: {champion.get('accuracy', 'N/A'):.4f}" if champion.get("accuracy") else "- Accuracy: N/A",
        f"- Kernel type: {champion.get('kernel_type', 'N/A')}",
        f"- Kernel hash: {champion.get('kernel_hash', 'N/A')[:16]}",
        "",
        "## Top 5 Attempts",
        "| Rank | Attempt | Wall Time | Accuracy | Kernel Type |",
        "|------|---------|-----------|----------|-------------|",
    ]
    for i, r in enumerate(top5, 1):
        lines.append(
            f"| {i} | #{r.get('attempt_n','?')} | {r.get('wall_time_s',0):.3f}s | "
            f"{r.get('accuracy',0):.4f} | {r.get('kernel_type','?')} |"
        )

    lines += [
        "",
        "## Wall-Time Trajectory",
        "```",
        _sparkline(times, width=60) if times else "(no data)",
        "```",
        "",
        "## Methodology",
        "Autonomous CUDA kernel optimization swarm:",
        "1. Qwen2.5-Coder-7B generates kernel proposals as JSON",
        "2. 5-gate firewall validates before compilation",
        "3. Kernels compiled via `torch.utils.cpp_extension` (sm_89)",
        "4. Training run measured wall-clock in subprocess isolation",
        "5. SlapBack engine translates profiler bottlenecks into targeted mutation hints",
        "6. SEED_POLY_CIFAR_v1 compresses context when token budget exceeded",
        "7. REM sleep extracts QA pairs from successful attempts for QLoRA",
    ]

    results_md = Path("RESULTS.md")
    results_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to: {results_md.absolute()}")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "logs/annealing_log.jsonl"
    analyze(log)
