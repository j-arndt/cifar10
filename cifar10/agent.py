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
    "kernel_type":          "conv_bn_relu_fusion | depthwise_conv | optimizer_fusion | data_pipeline",
    "layer_target":         "layer1.0  (max 50 chars)",
    "cuda_kernel_code":     "__global__ void ...  (CUDA C++ only, sm_89, max 64KB)",
    "pytorch_binding":      "def forward(...): ...  (valid Python, no system calls)",
    "integration_patch":    "out = forward(x)  (how to replace existing call)",
    "rationale":            "Why this kernel reduces wall time  (max 1000 chars)",
    "expected_speedup_pct": 25,
}, indent=2)


def build_system_prompt(config: dict, skeleton_code: str, champion_info: str) -> str:
    arch       = config.get("cuda_arch", "sm_89")
    target_t   = config.get("target_wall_time_s", 1.0)
    target_acc = config.get("target_accuracy", 0.940)

    return f"""\
You are the MOAB CIFAR-10 Crucible kernel optimization agent.

## HARDWARE
- GPU: RTX 4060 Mobile (Ada Lovelace)
- CUDA arch: {arch}
- VRAM: 8 GB GDDR6
- Memory bandwidth: 272 GB/s
- Peak FP16: 33 TFLOPS
- L2 cache: 32 MB
- Shared memory per SM: 100 KB
- Warp size: 32
- Max threads per block: 1024

## MISSION
Train ResNet-9 on CIFAR-10 to >= {target_acc} accuracy in < {target_t} second(s) wall clock.
The wall clock starts at dataset load and ends after final evaluation.

## CURRENT STATUS
{champion_info}

## SKELETON CODE (your starting point — this is what runs without any kernel)
```python
{skeleton_code[:5000]}
```

## YOUR OUTPUT FORMAT
You must output EXACTLY ONE JSON object, nothing else. No prose before or after.
Schema:
```json
{SCHEMA_JSON}
```

## CONSTRAINTS
- `cuda_kernel_code`: Must contain `__global__` (CUDA) or be valid Triton Python
- `pytorch_binding`: Must be valid Python, importable, no system calls
- No `os.`, `subprocess.`, `socket.`, `open(`, `eval(`, `exec(`, `__import__`
- Kernel size: < 64KB
- Target arch: {arch} — use `__half` for FP16, not `half`
- Build dir is managed — do NOT hardcode paths

## STRATEGY PRINCIPLES
1. Every separate kernel launch reads and writes all activations — fusion eliminates N-1 of these
2. Shared memory bank conflicts kill throughput — pad by +1 element per row
3. Coalesced global memory access is critical — threads in a warp must access consecutive addresses
4. BatchNorm can be fused into the preceding Conv kernel — eliminate the second pass
5. torch.compile() with mode='max-autotune' is always a valid first attempt

Output ONLY the JSON object. Start your response with `{{`.
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
            stop=["```\n\n", "Human:", "User:"],
        )
        return response["choices"][0]["message"]["content"]

    def parse_proposal(self, text: str) -> Optional[KernelProposal]:
        """Extract and parse JSON from agent output."""
        # Try to find outermost JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return KernelProposal.model_validate_json(match.group())
        except Exception:
            return None


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
