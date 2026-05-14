"""Fitness function — runs training in an isolated subprocess, returns FitnessResult."""
import hashlib
import json
import multiprocessing as mp
import os
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


def _worker(queue: mp.Queue, kernel_proposal, config: dict):
    """Runs inside subprocess — full VRAM isolation from agent."""
    import time
    import torch
    import yaml
    from pathlib import Path
    from cifar10.skeleton import run_skeleton

    try:
        t0 = time.perf_counter()

        if kernel_proposal is not None:
            # Future: inject compiled kernel into training loop
            # For now skeleton handles everything
            pass

        result = run_skeleton(config)
        wall_time = time.perf_counter() - t0  # subprocess total time

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
    kernel_hash = hashlib.sha256(
        str(kernel_proposal).encode() if kernel_proposal else b"skeleton"
    ).hexdigest()[:16]

    queue = mp.Queue()
    proc  = mp.Process(target=_worker, args=(queue, kernel_proposal, config), daemon=True)
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
        print("Running baseline fitness check...")
        result = run_cifar()
        print(result.to_json())
        print(f"\n{'PASS' if result.accuracy >= 0.90 else 'FAIL'}: accuracy={result.accuracy:.4f}  time={result.wall_time_s:.1f}s")
