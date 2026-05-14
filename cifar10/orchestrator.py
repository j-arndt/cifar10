"""Main orchestration loop — hot-swap, firewall, arena, fitness, slap, repeat."""
from __future__ import annotations

import datetime
import gc
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

from cifar10.agent     import CifarAgent, build_system_prompt
from cifar10.arena     import compile_kernel, CompilationError
from cifar10.firewall  import validate
from cifar10.fitness   import run_cifar, FitnessResult
from cifar10.observer  import AttemptRecord, log_attempt, get_champion, load_history, attempt_count
from cifar10.slap      import (
    build_baseline_slap, build_compilation_slap, build_fitness_slap,
)


def _load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _token_count(text: str) -> int:
    """Approximate token count (chars / 4)."""
    return len(text) // 4


class OrchestrationLoop:
    def __init__(self, config: dict):
        self.config        = config
        self.agent         = CifarAgent(config)
        self._shutdown     = False
        self._attempt_n    = 0
        self._slap_history: list[str] = []   # rolling context
        self._seeds_used   = 0
        self._rem_cycles   = 0

        # Register shutdown handlers
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        # NOTE: no print() here — stdout may be mid-write (reentrant crash on Windows)
        self._shutdown = True

    def _reload_config(self) -> None:
        """Reload config.yaml every cycle — picks up max_hours changes live."""
        try:
            self.config = _load_config()
        except Exception:
            pass  # keep running with old config if file is broken

    def _build_slap(self) -> str:
        """Route to the correct slap builder based on last attempt."""
        history = load_history()
        champion = get_champion()

        if not history:
            # First ever attempt — send the baseline challenge
            skeleton_code = Path("cifar10/skeleton.py").read_text(encoding="utf-8")
            baseline_time = 18.0  # estimated; will be updated once fitness runs
            return build_baseline_slap(
                baseline_time=baseline_time,
                target_time_s=self.config.get("target_wall_time_s", 1.0),
                skeleton_code=skeleton_code,
            )

        last = sorted(history, key=lambda r: r.attempt_n)[-1]

        if last.gate_failed:
            # Last attempt was blocked by firewall — schema/safety slap
            return build_compilation_slap(
                stderr=last.error or f"Gate {last.gate_failed} blocked this kernel.",
                attempt_n=self._attempt_n,
            )

        if last.error and "TIMEOUT" not in last.error and "SUBPROCESS" not in last.error:
            # Compilation error
            return build_compilation_slap(stderr=last.error, attempt_n=self._attempt_n)

        # Normal fitness slap
        from cifar10.profiler import ProfilerReport, BottleneckOp
        pr_dict = last.profiler_report or {}
        profiler = ProfilerReport(
            bottleneck_ops=[
                BottleneckOp(**op) for op in pr_dict.get("bottleneck_ops", [])
            ],
            memory_bandwidth_util_pct=pr_dict.get("memory_bandwidth_util_pct", 0.0),
            l2_cache_hit_rate_pct=pr_dict.get("l2_cache_hit_rate_pct"),
            shared_mem_bank_conflicts=pr_dict.get("shared_mem_bank_conflicts", False),
            total_cuda_ms=pr_dict.get("total_cuda_ms", 0.0),
        )
        return build_fitness_slap(
            accuracy=last.accuracy,
            wall_time_s=last.wall_time_s,
            target_time_s=self.config.get("target_wall_time_s", 1.0),
            profiler=profiler,
            attempt_n=self._attempt_n,
            champion_time=champion.wall_time_s if champion else None,
        )

    def _compress_context(self) -> None:
        """Trigger seed compression when context token budget is exceeded."""
        try:
            from cifar10.seed import compress_context
            history = load_history()
            seed_str = compress_context(history)
            self._slap_history = [f"[SEED COMPRESSED]\nState: {seed_str[:500]}"]
            self._seeds_used += 1
            print(f"[orchestrator] Seed compression #{self._seeds_used} complete.")
        except Exception as e:
            print(f"[orchestrator] Seed compression failed (non-fatal): {e}")
            self._slap_history = self._slap_history[-5:]  # keep last 5

    def _check_rem(self) -> None:
        """Trigger REM sleep if enough wins accumulated."""
        try:
            from cifar10.rem import trigger_rem_cycle, run_rem_cycle
            if trigger_rem_cycle():
                print("[orchestrator] REM cycle triggered...")
                run_rem_cycle(self.config)
                self._rem_cycles += 1
                print(f"[orchestrator] REM cycle #{self._rem_cycles} complete.")
        except Exception as e:
            print(f"[orchestrator] REM cycle skipped: {e}")

    def _champion_info(self) -> str:
        champ = get_champion()
        if champ is None:
            return f"No champion yet. Total attempts: {self._attempt_n}"
        return (
            f"Champion: {champ.wall_time_s:.3f}s | {champ.accuracy:.4f} accuracy | "
            f"attempt #{champ.attempt_n} | kernel={champ.kernel_type}"
        )

    def run(self, max_hours: Optional[float] = None, dry_run: bool = False) -> Optional[AttemptRecord]:
        """Main loop. Returns champion AttemptRecord or None."""
        if max_hours is None:
            max_hours = self.config.get("max_hours", 4)

        deadline = time.time() + max_hours * 3600
        dry_run_cycles = 3
        skeleton_code  = Path("cifar10/skeleton.py").read_text(encoding="utf-8")

        print(f"\n{'='*60}")
        print(f"MOAB CIFAR-10 Crucible — RTX 4060 Mobile")
        print(f"Target: <{self.config.get('target_wall_time_s', 1.0)}s | "
              f">={self.config.get('target_accuracy', 0.940)} accuracy")
        print(f"Run duration: {max_hours}h | {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"{'='*60}\n")

        while not self._shutdown:
            # ── Live config reload ──────────────────────────────────
            self._reload_config()
            max_hours = self.config.get("max_hours", max_hours)
            deadline  = time.time() + max_hours * 3600  # recalculate on each cycle

            # ── Time check ──────────────────────────────────────────
            elapsed_h = (time.time() - (deadline - max_hours * 3600)) / 3600
            if time.time() >= deadline:
                print(f"\n[orchestrator] Time limit reached ({max_hours}h). Exiting.")
                break

            self._attempt_n += 1
            cycle_start = time.time()

            # ── Dry-run exit check (all cycles, not just successful) ─
            if dry_run and self._attempt_n > dry_run_cycles:
                print(f"\n[dry-run] {dry_run_cycles} cycles complete. System validated.")
                return None

            print(f"\n[attempt {self._attempt_n}] {datetime.datetime.now().strftime('%H:%M:%S')} | "
                  f"elapsed: {elapsed_h:.1f}h | {self._champion_info()}")

            # ── Step 1: Build slap ──────────────────────────────────
            slap = self._build_slap()

            # ── Step 2: Context compression check ──────────────────
            context_tokens = sum(_token_count(s) for s in self._slap_history) + _token_count(slap)
            compress_thresh = self.config.get("compress_threshold_tokens", 6000)
            if context_tokens > compress_thresh:
                print(f"[orchestrator] Context {context_tokens} > {compress_thresh} tokens → compressing")
                self._compress_context()

            # ── Step 3: Agent load + generate ──────────────────────
            full_slap = "\n\n---\n\n".join(self._slap_history[-3:] + [slap])
            system_prompt = build_system_prompt(self.config, skeleton_code, self._champion_info())

            raw_proposal = None
            try:
                self.agent.load()
                raw_proposal = self.agent.generate(system_prompt, full_slap)
            except Exception as e:
                print(f"[orchestrator] Agent error: {e}")
            finally:
                self.agent.unload()  # CRITICAL: free VRAM before training

            if raw_proposal is None:
                print("[orchestrator] No proposal generated. Skipping cycle.")
                if dry_run and self._attempt_n >= dry_run_cycles:
                    print(f"\n[dry-run] {dry_run_cycles} cycles attempted (model missing or error).")
                    return None
                continue

            # ── Step 4: Extract JSON from raw LLM output ────────────
            proposal_json = self.agent.parse_proposal(raw_proposal)
            if proposal_json is None:
                # Model didn't produce valid JSON — log snippet and slap with schema error
                snippet = raw_proposal[:300].replace("\n", " ")
                print(f"[orchestrator] JSON parse failed. Raw output: {snippet!r}")
                err_msg = (
                    "Your output could not be parsed as JSON. "
                    "You MUST output ONLY a JSON object starting with {{ and ending with }}. "
                    "No prose, no markdown, no code fences. Raw output snippet:\n"
                    + raw_proposal[:400]
                )
                record = AttemptRecord(
                    attempt_id=f"jp{self._attempt_n:04d}",
                    attempt_n=self._attempt_n,
                    accuracy=0.0, wall_time_s=9999.0, passed=False,
                    kernel_hash="", kernel_type="parse_fail",
                    gate_failed="SCHEMA",
                    error=err_msg[:2000],
                    slap_received=slap[:500],
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                log_attempt(record)
                self._slap_history.append(slap)
                continue

            # ── Step 5: Firewall (receives clean Pydantic model) ────
            import json as _json
            fw_result = validate(proposal_json.model_dump_json())
            if not fw_result.passed:
                print(f"[orchestrator] Firewall blocked: gate={fw_result.gate_failed} | {fw_result.error_message[:120]}")
                record = AttemptRecord(
                    attempt_id=f"fw{self._attempt_n:04d}",
                    attempt_n=self._attempt_n,
                    accuracy=0.0, wall_time_s=9999.0, passed=False,
                    kernel_hash="", kernel_type="firewall_block",
                    gate_failed=fw_result.gate_failed,
                    error=fw_result.error_message,
                    slap_received=slap[:500],
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                log_attempt(record)
                self._slap_history.append(slap)
                continue

            # ── Step 5: Compile ─────────────────────────────────────
            compiled = None
            try:
                compiled = compile_kernel(fw_result.proposal, timeout_s=30)
                print(f"[orchestrator] Compiled in {compiled.elapsed_s:.1f}s | PTX: {compiled.ptx_hash}")
            except CompilationError as e:
                print(f"[orchestrator] Compilation failed: {str(e)[:200]}")
                record = AttemptRecord(
                    attempt_id=f"ce{self._attempt_n:04d}",
                    attempt_n=self._attempt_n,
                    accuracy=0.0, wall_time_s=9999.0, passed=False,
                    kernel_hash="", kernel_type=fw_result.proposal.kernel_type.value,
                    gate_failed=None, error=str(e)[:2000],
                    slap_received=slap[:500],
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                log_attempt(record)
                self._slap_history.append(slap)
                continue

            # ── Step 6: Fitness ─────────────────────────────────────
            print(f"[orchestrator] Running fitness...")
            fitness_result = run_cifar(kernel_proposal=compiled, config=self.config)
            print(f"[orchestrator] Result: acc={fitness_result.accuracy:.4f} "
                  f"time={fitness_result.wall_time_s:.3f}s "
                  f"{'PASS' if fitness_result.passed else 'fail'}")
            if fitness_result.error:
                print(f"[orchestrator] Fitness error: {fitness_result.error[:300]}")

            # ── Step 7: Log ─────────────────────────────────────────
            record = AttemptRecord(
                attempt_id=fitness_result.attempt_id,
                attempt_n=self._attempt_n,
                accuracy=fitness_result.accuracy,
                wall_time_s=fitness_result.wall_time_s,
                passed=fitness_result.passed,
                kernel_hash=fitness_result.kernel_hash,
                kernel_type=fw_result.proposal.kernel_type.value,
                gate_failed=None,
                error=fitness_result.error,
                slap_received=slap[:500],
                profiler_report=fitness_result.profiler_report,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            log_attempt(record)
            self._slap_history.append(slap)

            # ── Step 8: Victory check ────────────────────────────────
            if fitness_result.passed:
                print(f"\n{'='*60}")
                print(f"VICTORY! Target achieved on attempt #{self._attempt_n}")
                print(f"accuracy={fitness_result.accuracy:.4f}  wall_time={fitness_result.wall_time_s:.3f}s")
                print(f"{'='*60}")
                return record

            # ── Step 9: REM sleep check ──────────────────────────────
            if not dry_run:
                self._check_rem()

            # ── Dry run exit ─────────────────────────────────────────
            if dry_run and self._attempt_n >= dry_run_cycles:
                print(f"\n[dry-run] {dry_run_cycles} cycles complete. System validated.")
                return None

            cycle_elapsed = time.time() - cycle_start
            print(f"[orchestrator] Cycle time: {cycle_elapsed:.1f}s")

        # Summary on exit
        champ = get_champion()
        total = attempt_count()
        print(f"\n[orchestrator] Run complete. {total} attempts. "
              f"Best: {champ.wall_time_s:.3f}s / {champ.accuracy:.4f}" if champ
              else f"\n[orchestrator] Run complete. {total} attempts. No champion.")
        return get_champion()
