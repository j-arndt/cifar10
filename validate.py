"""Pre-launch system validation — run before starting the full run."""
import subprocess
import sys
import os
from pathlib import Path

import torch
import yaml


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def validate_system() -> bool:
    print("=" * 55)
    print("Pre-Launch Validation")
    print("=" * 55)

    all_ok = True

    # 1. Config
    config_ok = Path("config.yaml").exists()
    config = {}
    if config_ok:
        config = yaml.safe_load(Path("config.yaml").read_text())
    all_ok &= check("config.yaml exists", config_ok)

    # 2. GPU
    cuda_ok = torch.cuda.is_available()
    all_ok &= check("CUDA available", cuda_ok)
    if cuda_ok:
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        all_ok &= check("CUDA arch sm_89", cap >= (8, 9), f"detected sm_{cap[0]}{cap[1]}")
        all_ok &= check(f"VRAM >= 7.5GB", vram >= 7.5, f"{vram:.1f}GB on {name}")

    # 3. Required files
    for fname in ["cifar10/network.py", "cifar10/skeleton.py", "cifar10/fitness.py",
                  "cifar10/firewall.py", "cifar10/arena.py", "cifar10/schemas.py",
                  "cifar10/slap.py", "cifar10/observer.py", "cifar10/seed.py",
                  "cifar10/rem.py", "cifar10/agent.py", "cifar10/orchestrator.py",
                  "run.py"]:
        all_ok &= check(f"{fname}", Path(fname).exists())

    # 4. Directories writable
    for d in ["logs", "kernels/build"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        test_file = Path(d) / ".write_test"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            all_ok &= check(f"{d}/ writable", True)
        except Exception as e:
            all_ok &= check(f"{d}/ writable", False, str(e))

    # 5. Firewall test
    try:
        result = subprocess.run(
            [sys.executable, "-m", "cifar10.firewall"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        fw_ok = result.returncode == 0
        all_ok &= check("Firewall 10/10 tests", fw_ok,
                        "FAILED" if not fw_ok else "pass")
    except Exception as e:
        all_ok &= check("Firewall 10/10 tests", False, str(e))

    # 6. Seed probes
    try:
        result = subprocess.run(
            [sys.executable, "-m", "cifar10.seed"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        seed_ok = result.returncode == 0 and "ALL PROBES PASSED" in result.stdout
        all_ok &= check("Seed 6/6 DEPA probes", seed_ok)
    except Exception as e:
        all_ok &= check("Seed 6/6 DEPA probes", False, str(e))

    # 7. Agent model present (warn only)
    model_name = config.get("agent_model", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
    model_found = any(
        (Path(d) / model_name).exists()
        for d in [".", str(Path.home() / "models"), "C:/models", "D:/models"]
    )
    if not model_found:
        print(f"  [WARN] Model not found: {model_name}")
        print(f"         Place GGUF in ./models/ or ~/models/ before running")
    else:
        check(f"Agent model found", True, model_name[:40])

    # 8. Windows sleep disabled (advisory)
    print(f"  [INFO] Disable sleep before long run:")
    print(f"         powercfg /change standby-timeout-ac 0")

    print("=" * 55)
    print(f"Result: {'SYSTEM VALIDATED' if all_ok else 'VALIDATION FAILED — fix issues above'}")
    print("=" * 55)

    if all_ok:
        print("\nReady to launch:")
        print(f"  python run.py --hours 4 --agent 7b")
        print(f"  python run.py --hours 4 --dry-run   # test 3 cycles first")

    return all_ok


if __name__ == "__main__":
    ok = validate_system()
    sys.exit(0 if ok else 1)
