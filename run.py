"""Entry point — python run.py [--hours N] [--agent 7b|1.5b] [--dry-run]"""
import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="MOAB CIFAR-10 Crucible — Self-improving CUDA kernel swarm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --dry-run                    # validate system (3 cycles)
  python run.py --hours 4                    # 4-hour run (default)
  python run.py --hours 4 --agent 1.5b       # faster cycles, weaker kernels
  python run.py --hours 48 --agent 7b        # full overnight run
        """,
    )
    parser.add_argument("--hours",           type=float, default=None,
                        help="Max run duration in hours (default: from config.yaml)")
    parser.add_argument("--agent",           type=str,   default=None,
                        choices=["7b", "1.5b"],
                        help="Agent model size (default: from config.yaml)")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 3 cycles and exit — validates the full pipeline")
    parser.add_argument("--target-time",     type=float, default=None,
                        help="Override target wall time (seconds)")
    parser.add_argument("--target-accuracy", type=float, default=None,
                        help="Override target accuracy (0-1)")
    parser.add_argument("--skip-rem",        action="store_true",
                        help="Skip REM sleep / QLoRA cycles")
    parser.add_argument("--config",          type=str,   default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # ── CLI overrides ──────────────────────────────────────────────
    if args.hours is not None:
        config["max_hours"] = args.hours
    if args.target_time is not None:
        config["target_wall_time_s"] = args.target_time
    if args.target_accuracy is not None:
        config["target_accuracy"] = args.target_accuracy
    if args.agent == "1.5b":
        config["agent_model"] = "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
    elif args.agent == "7b":
        config["agent_model"] = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"

    max_hours = config.get("max_hours", 4)

    # ── Print run config ───────────────────────────────────────────
    print(f"Config: {config_path}")
    print(f"  target_wall_time_s:  {config.get('target_wall_time_s', 1.0)}s")
    print(f"  target_accuracy:     {config.get('target_accuracy', 0.940)}")
    print(f"  agent_model:         {config.get('agent_model', 'not set')}")
    print(f"  max_hours:           {max_hours}h")
    print(f"  dry_run:             {args.dry_run}")
    print(f"  skip_rem:            {args.skip_rem}")
    print()

    # ── Import here so errors surface early ───────────────────────
    try:
        import torch
        print(f"PyTorch:  {torch.__version__}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU:      {gpu} ({vram:.1f}GB)")
            cap = torch.cuda.get_device_capability()
            print(f"CUDA cap: sm_{cap[0]}{cap[1]}")
        else:
            print("WARNING: No CUDA GPU detected — training will be very slow", file=sys.stderr)
        print()
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    from cifar10.orchestrator import OrchestrationLoop

    loop = OrchestrationLoop(config)

    try:
        result = loop.run(max_hours=max_hours, dry_run=args.dry_run)
        if args.dry_run:
            print("\nDry run complete. System validated. Ready for full run.")
            sys.exit(0)
        if result and result.passed:
            print(f"\nSUCCESS: {result.wall_time_s:.3f}s | {result.accuracy:.4f} accuracy")
            sys.exit(0)
        else:
            print("\nRun complete. Target not yet achieved.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
