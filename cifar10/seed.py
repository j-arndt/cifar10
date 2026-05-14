"""SEED_POLY_CIFAR_v1 — state compression + 6-probe DEPA verification."""
from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

from cifar10.observer import AttemptRecord


# --------------------------------------------------------------------------- #
# Bloom filter (minimal, no external deps)
# --------------------------------------------------------------------------- #

class BloomFilter:
    """Simple Bloom filter using SHA-256 seeded with strategy strings."""
    def __init__(self, size_bits: int = 2048, k_hashes: int = 4):
        self.size_bits = size_bits
        self.k         = k_hashes
        self.bits      = bytearray(size_bits // 8)

    def _positions(self, item: str) -> list[int]:
        positions = []
        for seed in range(self.k):
            h = hashlib.sha256(f"{seed}:{item}".encode()).digest()
            positions.append(int.from_bytes(h[:4], "big") % self.size_bits)
        return positions

    def add(self, item: str) -> None:
        for pos in self._positions(item):
            self.bits[pos // 8] |= (1 << (pos % 8))

    def contains(self, item: str) -> bool:
        return all(
            self.bits[pos // 8] & (1 << (pos % 8))
            for pos in self._positions(item)
        )

    def to_bytes(self) -> bytes:
        return bytes(self.bits)

    @classmethod
    def from_bytes(cls, data: bytes, k_hashes: int = 4) -> "BloomFilter":
        bf = cls(size_bits=len(data) * 8, k_hashes=k_hashes)
        bf.bits = bytearray(data)
        return bf


# --------------------------------------------------------------------------- #
# Merkle root (for PTX hash chain)
# --------------------------------------------------------------------------- #

def _merkle_root(items: list[str]) -> str:
    if not items:
        return "0" * 16
    leaves = [hashlib.sha256(s.encode()).digest() for s in items]
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])  # pad odd
        leaves = [
            hashlib.sha256(leaves[i] + leaves[i + 1]).digest()
            for i in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()[:16]


# --------------------------------------------------------------------------- #
# SEED_POLY_CIFAR_v1 core
# --------------------------------------------------------------------------- #

@dataclass
class SeedPayload:
    # Kernel lineage
    kernel_merkle:   str            # Merkle root of compiled PTX hashes
    # Memory profile timeline (packed floats: attempt_n, wall_time_s, bw_util)
    mem_profiles_b64: str           # base64-encoded packed floats
    # Strategy bloom (failed approaches)
    slap_bloom_hex:  str            # hex of 256-byte Bloom filter
    # Champion snapshot
    champion_hash:   str            # kernel_hash of best run
    champion_time:   float          # best wall_time_s
    champion_acc:    float          # best accuracy
    # Context summary (compressed insights)
    context_gz_hex:  str            # hex of zlib-compressed insight string
    # Metadata
    attempt_count:   int
    seed_version:    str = "SEED_POLY_CIFAR_v1"

    def to_compact_str(self) -> str:
        """Compact <2048-byte representation for LLM context injection."""
        return json.dumps({
            "v":   self.seed_version,
            "km":  self.kernel_merkle,
            "mp":  self.mem_profiles_b64[:256],  # truncate if needed
            "sb":  self.slap_bloom_hex[:256],
            "ch":  self.champion_hash[:16],
            "ct":  round(self.champion_time, 3),
            "ca":  round(self.champion_acc, 4),
            "cx":  self.context_gz_hex[:512],
            "n":   self.attempt_count,
        }, separators=(",", ":"))


def compress_context(history: list[AttemptRecord]) -> str:
    """Compress attempt history into a SEED_POLY_CIFAR_v1 string."""
    import base64

    if not history:
        payload = SeedPayload(
            kernel_merkle="0" * 16, mem_profiles_b64="", slap_bloom_hex="00" * 256,
            champion_hash="none", champion_time=9999.0, champion_acc=0.0,
            context_gz_hex="", attempt_count=0,
        )
        return payload.to_compact_str()

    # 1. Kernel Merkle
    ptx_hashes = [r.kernel_hash for r in history if r.kernel_hash]
    kernel_merkle = _merkle_root(ptx_hashes)

    # 2. Memory profile (packed floats: attempt_n, wall_time_s, bw_util)
    profile_floats = []
    for r in history:
        bw = r.profiler_report.get("memory_bandwidth_util_pct", 0.0) if r.profiler_report else 0.0
        profile_floats.extend([float(r.attempt_n), r.wall_time_s, bw])
    packed = struct.pack(f"{len(profile_floats)}f", *profile_floats)
    mem_profiles_b64 = base64.b64encode(packed).decode()

    # 3. Slap Bloom (failed strategies — derived from gate_failed + kernel_type)
    bloom = BloomFilter(size_bits=2048)
    for r in history:
        if not r.passed:
            strategy_key = f"{r.kernel_type}:{r.gate_failed or 'fitness'}"
            bloom.add(strategy_key)
    slap_bloom_hex = bloom.to_bytes().hex()

    # 4. Champion
    sorted_by_time = sorted(
        [r for r in history if r.accuracy >= 0.90],
        key=lambda r: r.wall_time_s,
    )
    if sorted_by_time:
        champ = sorted_by_time[0]
        champion_hash = champ.kernel_hash
        champion_time = champ.wall_time_s
        champion_acc  = champ.accuracy
    else:
        champion_hash = "none"
        champion_time = 9999.0
        champion_acc  = 0.0

    # 5. Context summary — key insights from history
    n_total  = len(history)
    n_passed = sum(1 for r in history if r.passed)
    n_compile_errors = sum(1 for r in history if r.gate_failed == "SCHEMA" or (r.error and "error" in (r.error or "").lower()))
    n_safety = sum(1 for r in history if r.gate_failed in ("SAFETY", "PATTERN", "SYNTAX", "SIZE"))
    top_types = {}
    for r in history:
        top_types[r.kernel_type] = top_types.get(r.kernel_type, 0) + 1
    best_type = max(top_types, key=top_types.get) if top_types else "unknown"

    insight = (
        f"total={n_total} passed={n_passed} compile_errs={n_compile_errors} blocked={n_safety} "
        f"best_type={best_type} champion={champion_time:.3f}s/{champion_acc:.4f}"
    )
    context_gz = zlib.compress(insight.encode(), level=9)
    context_gz_hex = context_gz.hex()

    payload = SeedPayload(
        kernel_merkle=kernel_merkle,
        mem_profiles_b64=mem_profiles_b64,
        slap_bloom_hex=slap_bloom_hex,
        champion_hash=champion_hash,
        champion_time=champion_time,
        champion_acc=champion_acc,
        context_gz_hex=context_gz_hex,
        attempt_count=n_total,
    )
    return payload.to_compact_str()


def decompress_context(seed_str: str) -> dict:
    """Decode a SEED_POLY_CIFAR_v1 string back to human-readable summary."""
    d = json.loads(seed_str)
    insight = ""
    if d.get("cx"):
        try:
            insight = zlib.decompress(bytes.fromhex(d["cx"])).decode()
        except Exception:
            insight = "(decompression failed)"
    return {
        "version":         d.get("v"),
        "kernel_merkle":   d.get("km"),
        "champion_time_s": d.get("ct"),
        "champion_acc":    d.get("ca"),
        "attempts":        d.get("n"),
        "insight":         insight,
    }


# --------------------------------------------------------------------------- #
# 6-probe DEPA verification
# --------------------------------------------------------------------------- #

@dataclass
class ProbeResult:
    name:    str
    passed:  bool
    value:   float
    message: str


def run_depa_probes(seed_str: str, history: list[AttemptRecord]) -> list[ProbeResult]:
    """Run all 6 DEPA probes against the compressed seed."""
    import base64
    results = []

    try:
        d = json.loads(seed_str)
    except Exception as e:
        return [ProbeResult("PARSE", False, 0.0, f"Seed JSON invalid: {e}")]

    # Probe 1: BYTE_BUDGET — seed <= 2048 bytes
    seed_bytes = len(seed_str.encode())
    results.append(ProbeResult(
        "BYTE_BUDGET",
        passed=seed_bytes <= 2048,
        value=seed_bytes,
        message=f"Seed size: {seed_bytes} bytes (max 2048)",
    ))

    # Probe 2: KERNEL_CONTINUITY — Merkle root matches history
    ptx_hashes = [r.kernel_hash for r in history if r.kernel_hash]
    expected_merkle = _merkle_root(ptx_hashes)
    actual_merkle = d.get("km", "")
    results.append(ProbeResult(
        "KERNEL_CONTINUITY",
        passed=(actual_merkle == expected_merkle),
        value=1.0 if actual_merkle == expected_merkle else 0.0,
        message=f"Merkle: expected={expected_merkle[:8]}... actual={actual_merkle[:8]}...",
    ))

    # Probe 3: PROFILE_FIDELITY — reconstruct first attempt wall_time within 1%
    fidelity_ok = True
    if history and d.get("mp"):
        try:
            packed = base64.b64decode(d["mp"])
            n_floats = len(packed) // 4
            floats = list(struct.unpack(f"{n_floats}f", packed))
            # First triple: attempt_n, wall_time_s, bw_util
            reconstructed_time = floats[1]
            original_time = history[0].wall_time_s
            pct_error = abs(reconstructed_time - original_time) / max(original_time, 0.001)
            fidelity_ok = pct_error < 0.01
            results.append(ProbeResult(
                "PROFILE_FIDELITY",
                passed=fidelity_ok,
                value=pct_error,
                message=f"Time reconstruction error: {pct_error*100:.3f}% (max 1%)",
            ))
        except Exception as e:
            results.append(ProbeResult("PROFILE_FIDELITY", False, 1.0, f"Decode failed: {e}"))
    else:
        results.append(ProbeResult("PROFILE_FIDELITY", True, 0.0, "No history to verify"))

    # Probe 4: STRATEGY_INTEGRITY — Bloom false positive estimate < 5%
    # Test 100 random strings that were NOT added
    bloom_data = bytes.fromhex(d.get("sb", "00" * 256))
    bloom = BloomFilter.from_bytes(bloom_data)
    failed_strats = {
        f"{r.kernel_type}:{r.gate_failed or 'fitness'}"
        for r in history if not r.passed
    }
    fp_count = 0
    for i in range(100):
        test_key = f"random_noise_{i}_xyz_abc"
        if test_key not in failed_strats and bloom.contains(test_key):
            fp_count += 1
    fp_rate = fp_count / 100
    results.append(ProbeResult(
        "STRATEGY_INTEGRITY",
        passed=fp_rate < 0.05,
        value=fp_rate,
        message=f"Bloom FP rate: {fp_rate*100:.1f}% (max 5%)",
    ))

    # Probe 5: CHAMPION_CONTEXT — champion hash + time preserved
    sorted_by_time = sorted(
        [r for r in history if r.accuracy >= 0.90],
        key=lambda r: r.wall_time_s,
    )
    if sorted_by_time:
        expected_hash = sorted_by_time[0].kernel_hash[:16]
        expected_time = sorted_by_time[0].wall_time_s
        actual_hash   = d.get("ch", "")
        actual_time   = d.get("ct", 9999.0)
        time_ok   = abs(actual_time - expected_time) < 0.001
        hash_ok   = actual_hash == expected_hash
        results.append(ProbeResult(
            "CHAMPION_CONTEXT",
            passed=(time_ok and hash_ok),
            value=1.0 if (time_ok and hash_ok) else 0.0,
            message=f"Champion hash match={hash_ok}, time match={time_ok}",
        ))
    else:
        results.append(ProbeResult("CHAMPION_CONTEXT", True, 1.0, "No champion to verify"))

    # Probe 6: DRIFT_THRESHOLD — composite anomaly < 0.10
    drift_score = 0.0
    if not results[0].passed:  drift_score += 0.3  # byte budget failure is severe
    if not results[1].passed:  drift_score += 0.3  # Merkle mismatch is severe
    if not results[2].passed:  drift_score += 0.15
    if not results[3].passed:  drift_score += 0.1
    if not results[4].passed:  drift_score += 0.15
    results.append(ProbeResult(
        "DRIFT_THRESHOLD",
        passed=drift_score < 0.10,
        value=drift_score,
        message=f"Composite drift: {drift_score:.3f} (max 0.10)",
    ))

    return results


# --------------------------------------------------------------------------- #
# Test suite
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import datetime
    from cifar10.observer import AttemptRecord

    print("=" * 60)
    print("SEED_POLY_CIFAR_v1 TEST SUITE")
    print("=" * 60)

    # Mock history
    mock_history = [
        AttemptRecord(
            attempt_id="a001", attempt_n=1,
            accuracy=0.941, wall_time_s=14.8, passed=False,
            kernel_hash="abc12345678901234", kernel_type="conv_bn_relu_fusion",
            gate_failed=None, error=None, slap_received="",
            profiler_report={"memory_bandwidth_util_pct": 78.0, "total_cuda_ms": 10.2},
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
        AttemptRecord(
            attempt_id="a002", attempt_n=2,
            accuracy=0.942, wall_time_s=12.1, passed=False,
            kernel_hash="def45678901234567", kernel_type="depthwise_conv",
            gate_failed=None, error=None, slap_received="",
            profiler_report={"memory_bandwidth_util_pct": 85.0, "total_cuda_ms": 8.5},
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
        AttemptRecord(
            attempt_id="a003", attempt_n=3,
            accuracy=0.943, wall_time_s=9.5, passed=False,
            kernel_hash="ghi78901234567890", kernel_type="conv_bn_relu_fusion",
            gate_failed="SAFETY", error="blocked", slap_received="",
            profiler_report={},
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    ]

    # Compress
    print("\nCompressing 3-attempt history...")
    seed_str = compress_context(mock_history)
    print(f"  Seed size:    {len(seed_str.encode())} bytes (max 2048)")
    print(f"  Seed preview: {seed_str[:120]}...")

    # Decompress
    print("\nDecompressing...")
    summary = decompress_context(seed_str)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # DEPA probes
    print("\nRunning 6 DEPA probes...")
    probes = run_depa_probes(seed_str, mock_history)
    all_ok = True
    for p in probes:
        status = "PASS" if p.passed else "FAIL"
        if not p.passed:
            all_ok = False
        print(f"  [{status}] {p.name:<25} {p.message}")

    print("\n" + "=" * 60)
    print(f"Result: {'ALL PROBES PASSED' if all_ok else 'PROBE FAILURES DETECTED'}")
