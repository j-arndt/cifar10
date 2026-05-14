"""LLM agent wrapper — llama-cpp-python with VRAM-aware hot-swap."""
from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

from cifar10.schemas import KernelProposal, FirewallResult
from cifar10.firewall import validate


SCHEMA_JSON = json.dumps({
    "kernel_type":          "beam_search | hyp_tuning | wino_fusion | step_reduction | arch_change",
    "layer_target":         "SpeedyResNet or ConvGroup or whitening or all (max 50 chars)",
    "cuda_kernel_code":     "// tinygrad handles GPU kernels via UOp graph. Leave as comment.",
    "pytorch_binding":      "def apply(hyp, env):\n    env['BEAM'] = '4'\n    env['BS'] = '1024'\n    return hyp, env",
    "integration_patch":    "hyp, env = apply(hyp, env)",
    "rationale":            "Why this tinygrad optimization reduces wall time toward <1s (max 1000 chars)",
    "expected_speedup_pct": 25,
}, indent=2)


EXAMPLE_BINDING = '''
# EXAMPLE 1 - BEAM kernel search (finds optimal GPU kernels, trades search time for faster training):
def apply(hyp, env):
    env['BEAM'] = '4'       # search 4 kernel candidates per op
    env['WINO'] = '1'       # enable Winograd convolutions
    return hyp, env

# EXAMPLE 2 - fewer steps with larger batch to hit accuracy in less wall time:
def apply(hyp, env):
    env['BS']    = '2048'   # larger batch saturates tensor cores
    env['STEPS'] = '600'    # fewer steps, LR must compensate
    hyp['opt']['non_bias_lr'] *= 2.0
    hyp['opt']['bias_lr']     *= 2.0
    return hyp, env

# EXAMPLE 3 - aggressive: max BEAM + Winograd + EMA off (save EMA compute):
def apply(hyp, env):
    env['BEAM']  = '8'
    env['WINO']  = '1'
    env['EMA']   = '0'      # skip EMA, use raw model for eval
    env['STEPS'] = '800'
    return hyp, env
'''


def build_system_prompt(config: dict, skeleton_code: str, champion_info: str) -> str:
    arch       = config.get("cuda_arch", "sm_89")
    target_t   = config.get("target_wall_time_s", 1.0)
    target_acc = config.get("target_accuracy", 0.940)

    return f"""\
You are the MOAB hlb-cifar10/tinygrad Crucible optimization agent.

## HARDWARE
- GPU: RTX 4060 Mobile (Ada Lovelace, sm_89)
- VRAM: 8 GB, 272 GB/s bandwidth, 33 TFLOPS FP16
- tinygrad installed with CUDA backend + kernel JIT

## MISSION
Optimize tinygrad's hlb-cifar10 (SpeedyResNet) on CIFAR-10:
- Target: >= {target_acc} accuracy AND < {target_t}s TOTAL wall time
- The baseline runs 1000 steps. YOU must cut wall time to < {target_t}s.

## CURRENT STATUS
{champion_info}

## HOW TINYGRAD WORKS
- tinygrad builds a lazy UOp graph. Kernels are compiled/fused at `.realize()` or step execution.
- BEAM=N: searches N kernel candidates per op. Higher = better kernels, one-time search cost.
- WINO=1: Winograd convolutions — faster 3x3 convs.
- TinyJit captures the full training step as a single compiled graph after 2 warmup steps.
- After warmup, each step runs the compiled graph: pure GPU, minimal Python overhead.

## YOUR BINDING CONTRACT
Your `pytorch_binding` MUST define:
```python
def apply(hyp, env):
    # hyp: dict with opt/net/ema sub-dicts (learning rates, steps, etc)
    # env: dict with BEAM, WINO, BS, STEPS, EMA, CUDA, CUTMIX etc
    # Return BOTH modified
    return hyp, env
```

## WORKING EXAMPLES
{EXAMPLE_BINDING}

## OUTPUT FORMAT — ONLY valid JSON, start with {{, end with }}
{SCHEMA_JSON}

## OPTIMIZATION LEVERS (tinygrad-specific)
1. env['BEAM'] = '4' or '8' — searches for optimal GPU kernels. One-time cost, permanent speedup.
2. env['WINO'] = '1' — Winograd convolutions, faster 3x3 ops
3. env['STEPS'] = '500' — fewer steps. Risk: lower accuracy unless LR is scaled up.
4. env['BS'] = '2048' — larger batch fills tensor cores. Scale LR proportionally.
5. env['EMA'] = '0' — skip EMA, use raw model weights for eval (saves ~5% compute)
6. hyp['opt']['non_bias_lr'] / hyp['opt']['bias_lr'] — tune learning rate schedules
7. hyp['net']['cutmix_steps'] — delay cutmix for faster early convergence

## RULES
- apply() MUST return (hyp, env) tuple
- env values must be strings (they become os.environ)
- Do NOT import torch — this is tinygrad, not PyTorch
- cuda_kernel_code: always set to '// tinygrad handles kernels via UOp graph'
- Output ONLY the JSON. No prose. No markdown fences.
"""


class CifarAgent:
    def __init__(self, config: dict):
        self.config = config
        self.llm    = None
        self._model_path = self._find_model()

    def _find_model(self) -> Optional[str]:
        """Locate GGUF model. Check common locations."""
        model_name = self.config.get("agent_model", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
        search_dirs = [
            Path("models"),                                     # ./models/ — PRIMARY
            Path("."),
            Path.home() / "models",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("C:/models"),
            Path("D:/models"),
        ]
        for d in search_dirs:
            candidate = d / model_name
            if candidate.exists():
                return str(candidate)
        # Return None — caller will fail fast with a clear error
        return None

    def load(self) -> None:
        """Load LLM. Assert VRAM is mostly free first."""
        if self._model_path is None:
            model_name = self.config.get("agent_model", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
            raise FileNotFoundError(
                f"Model not found: {model_name}\n"
                f"Place the GGUF file in: {Path('models').absolute()}\\"
            )

        if torch.cuda.is_available():
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            if reserved_gb > 1.5:
                raise RuntimeError(
                    f"VRAM not free before agent load: {reserved_gb:.1f}GB reserved. "
                    "Ensure fitness function subprocess has fully exited."
                )

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        n_ctx = self.config.get("agent_context_tokens", 8192)
        print(f"[agent] Loading model: {self._model_path}")
        self.llm = Llama(
            model_path=self._model_path,
            n_gpu_layers=-1,     # full GPU offload
            n_ctx=n_ctx,
            verbose=False,
        )
        print(f"[agent] Model loaded. Context: {n_ctx} tokens")

    def unload(self) -> None:
        """Free VRAM completely before fitness function runs."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            freed_gb = (torch.cuda.get_device_properties(0).total_memory
                        - torch.cuda.memory_reserved()) / 1e9
            print(f"[agent] Unloaded. Free VRAM: {freed_gb:.1f}GB")

    def generate(self, system_prompt: str, slap_message: str) -> str:
        """Generate a kernel proposal in response to the slap message."""
        if self.llm is None:
            raise RuntimeError("Agent not loaded. Call load() first.")

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": slap_message},
            ],
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"},  # grammar-constrained: always returns complete JSON
        )
        return response["choices"][0]["message"]["content"]

    def parse_proposal(self, text: str) -> Optional[KernelProposal]:
        """Extract and parse JSON from agent output. Raises ValueError with reason on failure."""
        import json as _json

        # With json_object mode the entire response IS the JSON — try it directly first
        try:
            return KernelProposal.model_validate_json(text)
        except Exception as direct_err:
            pass  # fall through to regex extraction

        # Fallback: extract outermost {...} (handles models that add preamble text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in output. First 200 chars: {text[:200]!r}")

        json_str = match.group()
        try:
            KernelProposal.model_validate_json(json_str)
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}") from e

        return KernelProposal.model_validate_json(json_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        config = yaml.safe_load(Path("config.yaml").read_text())
        skeleton_code = Path("cifar10/skeleton.py").read_text()

        agent = CifarAgent(config)
        print(f"[dry-run] Model path: {agent._model_path}")

        champion_info = "No champion yet. Baseline: ~18s."
        system_prompt = build_system_prompt(config, skeleton_code, champion_info)

        test_slap = """\
## INITIAL CHALLENGE
Baseline wall time: 18.0s. Target: <1.0s.
Propose one kernel optimization. Output ONLY the JSON object."""

        print("[dry-run] Loading agent...")
        try:
            agent.load()
            print("[dry-run] Generating proposal...")
            raw = agent.generate(system_prompt, test_slap)
            print(f"[dry-run] Raw output ({len(raw)} chars):")
            print(raw[:500])
            proposal = agent.parse_proposal(raw)
            if proposal:
                print(f"\n[dry-run] Parsed proposal: kernel_type={proposal.kernel_type}")
            else:
                print("\n[dry-run] Could not parse proposal (expected during dry-run if model not present)")
        except Exception as e:
            print(f"[dry-run] Agent load/generate skipped: {e}")
            print("[dry-run] (Install llama-cpp-python and place model GGUF to test agent)")
        finally:
            agent.unload()

        print("\n[dry-run] Agent module OK.")
