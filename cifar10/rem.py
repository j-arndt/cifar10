"""REM Sleep — QA pair extraction + QLoRA fine-tune trigger."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from cifar10.observer import AttemptRecord, load_history

REM_PAIRS_LOG = Path("logs/rem_pairs.jsonl")
REM_STATE_FILE = Path("logs/rem_state.json")


# --------------------------------------------------------------------------- #
# QA pair extraction
# --------------------------------------------------------------------------- #

def _extract_qa_pair(record: AttemptRecord) -> Optional[dict]:
    """
    Extract a question-answer pair from a successful (improving) attempt.
    Returns None if the record doesn't contain enough signal.
    """
    if record.wall_time_s >= 9999.0:
        return None

    pr = record.profiler_report or {}
    bottleneck_ops = pr.get("bottleneck_ops", [])
    bw_util        = pr.get("memory_bandwidth_util_pct", 0.0)
    bank_conflicts = pr.get("shared_mem_bank_conflicts", False)
    l2_rate        = pr.get("l2_cache_hit_rate_pct")

    # Build a rich question from profiler context
    bottleneck_str = ""
    if bottleneck_ops:
        top = bottleneck_ops[0]
        if isinstance(top, dict):
            bottleneck_str = (
                f"The {top.get('op','?')} layer is consuming {top.get('pct_of_total',0):.0f}% "
                f"of compute ({top.get('self_cuda_ms',0):.1f}ms). "
            )

    bc_str = "Shared memory bank conflicts detected. " if bank_conflicts else ""
    l2_str = f"L2 cache hit rate is {l2_rate:.0f}%. " if l2_rate else ""
    bw_str = f"Memory bandwidth utilization: {bw_util:.0f}%. "

    question = (
        f"Profiler context: {bottleneck_str}{bc_str}{l2_str}{bw_str}"
        f"Current wall time: {record.wall_time_s:.3f}s. "
        f"Target: <1.0s. "
        f"Kernel type applied: {record.kernel_type}. "
        f"What mutation was required to improve performance?"
    )

    if not record.slap_received:
        return None

    answer = (
        f"Applied kernel type: {record.kernel_type}. "
        f"Result: {record.wall_time_s:.3f}s wall time, {record.accuracy:.4f} accuracy. "
        f"Context from slap: {record.slap_received[:400]}"
    )

    return {
        "question":   question,
        "answer":     answer,
        "metadata": {
            "attempt_id":    record.attempt_id,
            "attempt_n":     record.attempt_n,
            "wall_time_s":   record.wall_time_s,
            "accuracy":      record.accuracy,
            "kernel_type":   record.kernel_type,
            "kernel_hash":   record.kernel_hash,
            "extracted_at":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    }


def extract_improving_pairs(history: Optional[list[AttemptRecord]] = None) -> int:
    """
    Walk history, extract QA pairs from attempts that improved on the previous
    best wall time. Appends to rem_pairs.jsonl.
    Returns number of new pairs written.
    """
    if history is None:
        history = load_history()

    REM_PAIRS_LOG.parent.mkdir(exist_ok=True)

    best_time  = 9999.0
    n_written  = 0

    for record in sorted(history, key=lambda r: r.attempt_n):
        if record.accuracy < 0.90:
            continue
        if record.wall_time_s < best_time:
            best_time = record.wall_time_s
            pair = _extract_qa_pair(record)
            if pair:
                with REM_PAIRS_LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(pair) + "\n")
                n_written += 1

    return n_written


def load_rem_pairs() -> list[dict]:
    if not REM_PAIRS_LOG.exists():
        return []
    pairs = []
    for line in REM_PAIRS_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                pairs.append(json.loads(line))
            except Exception:
                continue
    return pairs


# --------------------------------------------------------------------------- #
# REM cycle state tracking
# --------------------------------------------------------------------------- #

def _load_rem_state() -> dict:
    if not REM_STATE_FILE.exists():
        return {"wins_since_last_rem": 0, "rem_cycles_run": 0, "last_rem_at": None}
    try:
        return json.loads(REM_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"wins_since_last_rem": 0, "rem_cycles_run": 0, "last_rem_at": None}


def _save_rem_state(state: dict) -> None:
    REM_STATE_FILE.parent.mkdir(exist_ok=True)
    REM_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def record_win() -> None:
    """Call after each attempt that improves champion wall time."""
    state = _load_rem_state()
    state["wins_since_last_rem"] = state.get("wins_since_last_rem", 0) + 1
    _save_rem_state(state)


def trigger_rem_cycle(config: Optional[dict] = None) -> bool:
    """Return True if REM cycle should fire now."""
    state  = _load_rem_state()
    wins   = state.get("wins_since_last_rem", 0)
    thresh = 10
    if config:
        thresh = config.get("rem_trigger_every_n_wins", 10)
    return wins >= thresh


def run_rem_cycle(config: Optional[dict] = None) -> None:
    """
    Run a REM sleep cycle:
    1. Extract improving QA pairs from history
    2. If enough pairs, launch QLoRA (via WSL2) or log intent
    3. Reset win counter
    """
    history  = load_history()
    n_new    = extract_improving_pairs(history)
    all_pairs = load_rem_pairs()

    print(f"[rem] Extracted {n_new} new QA pairs. Total: {len(all_pairs)}")

    state = _load_rem_state()
    state["wins_since_last_rem"] = 0
    state["rem_cycles_run"]      = state.get("rem_cycles_run", 0) + 1
    state["last_rem_at"]         = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _save_rem_state(state)

    # If enough pairs exist, attempt QLoRA launch
    min_pairs_for_qlora = 20
    if len(all_pairs) >= min_pairs_for_qlora:
        _launch_qlora(config, all_pairs)
    else:
        print(f"[rem] Skipping QLoRA — need {min_pairs_for_qlora} pairs, have {len(all_pairs)}")


def _launch_qlora(config: Optional[dict], pairs: list[dict]) -> None:
    """
    Write dataset and launch QLoRA via WSL2 Unsloth training script.
    Non-blocking — logs intent if WSL2/Unsloth not available.
    """
    dataset_path = Path("logs/rem_dataset.jsonl")
    dataset_path.write_text(
        "\n".join(json.dumps(p) for p in pairs), encoding="utf-8"
    )
    print(f"[rem] Dataset written: {dataset_path} ({len(pairs)} pairs)")

    # Try WSL2 launch
    wsl_train_script = Path("wsl_train.sh")
    if wsl_train_script.exists():
        import subprocess
        try:
            subprocess.Popen(
                ["wsl", "bash", str(wsl_train_script)],
                stdout=open("logs/rem_train.log", "a"),
                stderr=subprocess.STDOUT,
            )
            print("[rem] QLoRA launched via WSL2 (non-blocking)")
        except Exception as e:
            print(f"[rem] WSL2 launch failed: {e}. Dataset saved for manual run.")
    else:
        print("[rem] wsl_train.sh not found — dataset saved. Run QLoRA manually.")
        print(f"[rem] Command: wsl python train.py --dataset {dataset_path}")


# --------------------------------------------------------------------------- #
# Test suite
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import datetime
    from cifar10.observer import AttemptRecord

    print("=" * 60)
    print("REM SLEEP TEST SUITE")
    print("=" * 60)

    mock_history = [
        AttemptRecord(
            attempt_id="r001", attempt_n=1,
            accuracy=0.941, wall_time_s=14.8, passed=False,
            kernel_hash="abc123", kernel_type="conv_bn_relu_fusion",
            gate_failed=None, error=None,
            slap_received="Bottleneck: conv2d is 41% of compute. Fuse BN into conv kernel.",
            profiler_report={
                "bottleneck_ops": [{"op": "aten::conv2d", "self_cuda_ms": 4.2, "pct_of_total": 41}],
                "memory_bandwidth_util_pct": 78.0, "shared_mem_bank_conflicts": False,
                "l2_cache_hit_rate_pct": 62, "total_cuda_ms": 10.2,
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
        AttemptRecord(
            attempt_id="r002", attempt_n=2,
            accuracy=0.942, wall_time_s=12.1, passed=False,
            kernel_hash="def456", kernel_type="conv_bn_relu_fusion",
            gate_failed=None, error=None,
            slap_received="Bottleneck: batch_norm is 20% of compute. Still not fused.",
            profiler_report={
                "bottleneck_ops": [{"op": "aten::batch_norm", "self_cuda_ms": 2.1, "pct_of_total": 20}],
                "memory_bandwidth_util_pct": 85.0, "shared_mem_bank_conflicts": True,
                "l2_cache_hit_rate_pct": 55, "total_cuda_ms": 8.5,
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
        AttemptRecord(
            attempt_id="r003", attempt_n=3,
            accuracy=0.943, wall_time_s=9.5, passed=False,
            kernel_hash="ghi789", kernel_type="optimizer_fusion",
            gate_failed=None, error=None,
            slap_received="SGD step is 10% of compute. Fuse into backward pass.",
            profiler_report={
                "bottleneck_ops": [{"op": "aten::sgd", "self_cuda_ms": 0.9, "pct_of_total": 10}],
                "memory_bandwidth_util_pct": 91.0, "shared_mem_bank_conflicts": False,
                "l2_cache_hit_rate_pct": 70, "total_cuda_ms": 7.1,
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    ]

    n_pairs = extract_improving_pairs(mock_history)
    print(f"\nExtracted {n_pairs} QA pairs from {len(mock_history)} attempts")

    pairs = load_rem_pairs()
    if pairs:
        print(f"\nSample QA pair:")
        print(f"  Q: {pairs[0]['question'][:120]}...")
        print(f"  A: {pairs[0]['answer'][:120]}...")

    record_win()
    record_win()
    state = _load_rem_state()
    print(f"\nREM state: wins_since_last_rem={state['wins_since_last_rem']}")

    trigger = trigger_rem_cycle({"rem_trigger_every_n_wins": 2})
    print(f"Trigger fires (thresh=2): {trigger}")

    print("\n" + "=" * 60)
    print("REM TEST COMPLETE")
