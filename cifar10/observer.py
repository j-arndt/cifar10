"""Attempt logger — appends to JSONL, maintains champion.json."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


ANNEALING_LOG = Path("logs/annealing_log.jsonl")
CHAMPION_FILE = Path("logs/champion.json")


@dataclass
class AttemptRecord:
    attempt_id:       str
    attempt_n:        int
    accuracy:         float
    wall_time_s:      float
    passed:           bool
    kernel_hash:      str
    kernel_type:      str
    gate_failed:      Optional[str]  = None
    error:            Optional[str]  = None
    slap_received:    str            = ""
    profiler_report:  dict           = field(default_factory=dict)
    timestamp:        str            = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def log_attempt(record: AttemptRecord) -> None:
    """Append record to annealing log. Update champion if best or passed."""
    ANNEALING_LOG.parent.mkdir(exist_ok=True)
    with ANNEALING_LOG.open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")

    # Update champion if this is a passing run or the fastest time seen
    champion = _load_champion_dict()
    is_better = (
        record.passed
        or champion is None
        or (record.accuracy >= 0.90 and record.wall_time_s < champion.get("wall_time_s", 9999))
    )
    if is_better:
        CHAMPION_FILE.write_text(record.to_json(), encoding="utf-8")


def load_history() -> list[AttemptRecord]:
    """Load all attempts from log, sorted by wall_time_s ascending."""
    if not ANNEALING_LOG.exists():
        return []
    records = []
    for line in ANNEALING_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(AttemptRecord(**d))
        except Exception:
            continue
    return sorted(records, key=lambda r: r.wall_time_s)


def get_champion() -> Optional[AttemptRecord]:
    """Return champion record, or None if no champion yet."""
    d = _load_champion_dict()
    if d is None:
        return None
    try:
        return AttemptRecord(**d)
    except Exception:
        return None


def _load_champion_dict() -> Optional[dict]:
    if not CHAMPION_FILE.exists():
        return None
    try:
        return json.loads(CHAMPION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def attempt_count() -> int:
    """Fast line count of annealing log (no full parse)."""
    if not ANNEALING_LOG.exists():
        return 0
    with ANNEALING_LOG.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


if __name__ == "__main__":
    import datetime

    # Quick functional test
    print("Testing observer...")

    r = AttemptRecord(
        attempt_id="test0001", attempt_n=1,
        accuracy=0.941, wall_time_s=14.8, passed=False,
        kernel_hash="abc12345", kernel_type="conv_bn_relu_fusion",
        gate_failed=None, error=None,
        slap_received="baseline slap here",
        profiler_report={"total_cuda_ms": 10.2},
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    log_attempt(r)
    history = load_history()
    champion = get_champion()

    print(f"  Logged 1 attempt. History length: {len(history)}")
    print(f"  Champion: {champion.wall_time_s}s  accuracy={champion.accuracy}")
    print(f"  Total count: {attempt_count()}")
    print("  Observer OK")
