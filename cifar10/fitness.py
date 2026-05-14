"""Fitness function — runs training in an isolated subprocess, returns FitnessResult."""
import hashlib
import json
import multiprocessing as mp
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class FitnessResult:
    attempt_id:      str
    accuracy:        float
    wall_time_s:     float
    passed:          bool
    kernel_hash:     str
    profiler_report: dict
    error:           Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _worker(queue: mp.Queue, binding_path: Optional[str], config: dict):
    """Runs inside subprocess — full VRAM isolation from agent."""
    import time
    import torch
    from cifar10.skeleton import run_skeleton

    try:
        t0 = time.perf_counter()

        # Apply the pytorch_binding if provided
        apply_fn = None
        if binding_path and Path(binding_path).exists():
            try:
                ns: dict = {}
                exec(compile(Path(binding_path).read_text(encoding="utf-8"), binding_path, "exec"), ns)
                apply_fn = ns.get("apply")
            except Exception as e:
                queue.put({"accuracy": 0.0, "wall_time_s": 9999.0,
                           "error": f"binding exec failed in subprocess: {e}"})
                return

        result = run_skeleton(config, apply_fn=apply_fn)
        queue.put({
            "accuracy":    result["accuracy"],
            "wall_time_s": result["wall_time_s"],
            "error":       None,
        })
    except Exception as e:
        queue.put({"accuracy": 0.0, "wall_time_s": 9999.0, "error": str(e)})


def run_cifar(kernel_proposal=None, config: dict | None = None) -> FitnessResult:
    if config is None:
        import yaml
        config = yaml.safe_load(Path("config.yaml").read_text())

    attempt_id  = str(uuid.uuid4())[:8]

    # Extract binding_path from CompiledKernel or None
    binding_path = None
    kernel_hash  = "skeleton"
    if kernel_proposal is not None:
        binding_path = getattr(kernel_proposal, "binding_path", None)
        kernel_hash  = getattr(kernel_proposal, "ptx_hash", "unknown")

    queue = mp.Queue()
    proc  = mp.Process(
        target=_worker,
        args=(queue, binding_path, config),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=600)  # 10-minute hard ceiling

    if proc.is_alive():
        proc.kill()
        return FitnessResult(
            attempt_id=attempt_id, accuracy=0.0, wall_time_s=9999.0,
            passed=False, kernel_hash=kernel_hash, profiler_report={},
            error="TIMEOUT: training exceeded 600 seconds",
        )

    if queue.empty():
        return FitnessResult(
            attempt_id=attempt_id, accuracy=0.0, wall_time_s=9999.0,
            passed=False, kernel_hash=kernel_hash, profiler_report={},
            error="SUBPROCESS: no result returned (crash)",
        )

    result = queue.get()
    target_acc  = config.get("target_accuracy", 0.940)
    target_time = config.get("target_wall_time_s", 1.0)

    return FitnessResult(
        attempt_id=attempt_id,
        accuracy=result["accuracy"],
        wall_time_s=result["wall_time_s"],
        passed=(result["accuracy"] >= target_acc and result["wall_time_s"] < target_time),
        kernel_hash=kernel_hash,
        profiler_report={},
        error=result.get("error"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Run baseline verification")
    args = parser.parse_args()

    if args.verify:
        print("Running baseline fitness check (skeleton, no kernel)...")
        result = run_cifar()
        print(result.to_json())
        print(f"\n{'PASS' if result.accuracy >= 0.90 else 'FAIL'}: "
              f"accuracy={result.accuracy:.4f}  time={result.wall_time_s:.1f}s")
