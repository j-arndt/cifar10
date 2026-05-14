"""Compilation sandbox — Python-exec mode (primary) + CUDA load_inline (optional).

On Windows without MSVC (cl.exe), CUDA kernels cannot be compiled via load_inline.
Primary path: exec the pytorch_binding in a controlled namespace — covers all
torch.compile(), Triton, fused optimizer, and data pipeline optimizations.
CUDA path: attempted only when cl.exe is present; skipped gracefully if not.
"""
from __future__ import annotations

import ast
import hashlib
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from cifar10.schemas import KernelProposal


class CompilationError(Exception):
    def __init__(self, stderr: str, elapsed_s: float = 0.0):
        super().__init__(stderr)
        self.stderr    = stderr
        self.elapsed_s = elapsed_s


@dataclass
class CompiledKernel:
    """Result of arena compilation — passed to fitness subprocess via binding_path."""
    binding_path: str          # path to .py file the subprocess will exec
    ptx_hash:     str          # SHA256 of binding code (identity for Python-only)
    kernel_type:  str          # free string metadata
    build_dir:    str
    elapsed_s:    float
    cuda_compiled: bool = False  # True only if CUDA load_inline succeeded


# --------------------------------------------------------------------------- #
# Compiler availability
# --------------------------------------------------------------------------- #

def _cl_available() -> bool:
    """Return True if MSVC cl.exe is on PATH."""
    return shutil.which("cl") is not None


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None


# --------------------------------------------------------------------------- #
# Python-exec sandbox (primary path — no cl.exe needed)
# --------------------------------------------------------------------------- #

def _write_binding_file(proposal: KernelProposal, build_dir: str) -> str:
    """Write the pytorch_binding as an importable .py file. Returns path."""
    binding_code = proposal.pytorch_binding.strip()

    # Wrap bare expressions into a proper module with apply() entrypoint
    # The binding should define: apply(model, config) -> model
    # If it doesn't define apply(), we wrap it
    try:
        tree = ast.parse(binding_code)
        defines_apply = any(
            isinstance(node, ast.FunctionDef) and node.name == "apply"
            for node in ast.walk(tree)
        )
    except SyntaxError:
        defines_apply = False

    if not defines_apply:
        # Wrap the binding in an apply() function
        indented = "\n".join("    " + line for line in binding_code.splitlines())
        binding_code = (
            "import torch\n"
            "def apply(model, config):\n"
            f"{indented}\n"
            "    return model\n"
        )

    # Always ensure torch is imported
    if "import torch" not in binding_code:
        binding_code = "import torch\n" + binding_code

    path = str(Path(build_dir) / "binding.py")
    Path(path).write_text(binding_code, encoding="utf-8")
    return path


def _compile_python_only(proposal: KernelProposal, timeout_s: int) -> CompiledKernel:
    """Exec the pytorch_binding as Python — no CUDA compiler needed."""
    t0 = time.perf_counter()

    code_hash = hashlib.sha256(proposal.pytorch_binding.encode()).hexdigest()[:8]
    build_dir  = str(Path("kernels") / "build" / code_hash)
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # Write and syntax-validate the binding
    binding_path = _write_binding_file(proposal, build_dir)

    # Validate it actually parses and exec's cleanly in a sandboxed namespace
    source = Path(binding_path).read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise CompilationError(f"Python syntax error: {e}", elapsed_s=time.perf_counter() - t0)

    # Light exec test — just ensure it runs without import errors
    try:
        ns: dict = {}
        exec(compile(source, binding_path, "exec"), ns)
        if "apply" not in ns:
            raise CompilationError(
                "pytorch_binding must define an apply(model, config) function. "
                "Example:\n"
                "  def apply(model, config):\n"
                "      import torch\n"
                "      return torch.compile(model, mode='max-autotune')\n",
                elapsed_s=time.perf_counter() - t0,
            )
    except CompilationError:
        raise
    except Exception as e:
        raise CompilationError(
            f"pytorch_binding exec failed: {type(e).__name__}: {e}",
            elapsed_s=time.perf_counter() - t0,
        )

    ptx_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
    elapsed = time.perf_counter() - t0

    return CompiledKernel(
        binding_path=binding_path,
        ptx_hash=ptx_hash,
        kernel_type=proposal.kernel_type,
        build_dir=build_dir,
        elapsed_s=elapsed,
        cuda_compiled=False,
    )


# --------------------------------------------------------------------------- #
# CUDA compilation (optional — only when cl.exe present)
# --------------------------------------------------------------------------- #

def _compile_cuda(proposal: KernelProposal, build_dir: str, timeout_s: int) -> Optional[object]:
    """Attempt CUDA load_inline. Returns module or None on failure."""
    from torch.utils.cpp_extension import load_inline

    extra_cuda_cflags = ["-O3", "-arch=sm_89", "-use_fast_math", "--expt-relaxed-constexpr"]
    code_hash = hashlib.sha256(proposal.cuda_kernel_code.encode()).hexdigest()[:8]
    result: dict = {}
    error:  dict = {}

    def _compile():
        try:
            mod = load_inline(
                name=f"kernel_{code_hash}",
                cpp_sources=[""],
                cuda_sources=[proposal.cuda_kernel_code],
                extra_cuda_cflags=extra_cuda_cflags,
                build_directory=build_dir,
                verbose=False,
            )
            result["module"] = mod
        except Exception as e:
            error["msg"] = str(e)

    t = threading.Thread(target=_compile, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive() or error:
        return None
    return result.get("module")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def compile_kernel(proposal: KernelProposal, timeout_s: int = 60) -> CompiledKernel:
    """
    Primary: exec pytorch_binding as Python (always works, no cl.exe needed).
    Optional: also attempt CUDA load_inline if cl.exe is present.

    Raises CompilationError if Python-exec fails (syntax error, bad binding, etc.)
    """
    t0 = time.perf_counter()

    # Always do Python-exec first — this is the primary path
    compiled = _compile_python_only(proposal, timeout_s=timeout_s)

    # Optionally also compile CUDA kernel if environment supports it
    has_meaningful_cuda = (
        proposal.cuda_kernel_code.strip()
        and "__global__" in proposal.cuda_kernel_code
        and _cl_available()
        and _nvcc_available()
    )
    if has_meaningful_cuda:
        cuda_mod = _compile_cuda(proposal, compiled.build_dir, timeout_s=max(10, timeout_s - 5))
        if cuda_mod is not None:
            compiled.cuda_compiled = True
            print(f"[arena] CUDA kernel also compiled (sm_89). PTX: {compiled.ptx_hash}")

    return compiled


# --------------------------------------------------------------------------- #
# Test suite
# --------------------------------------------------------------------------- #

_GOOD_BINDING = """\
def apply(model, config):
    import torch
    return torch.compile(model, mode='reduce-overhead')
"""

_BAD_BINDING = """\
def apply(model, config):
    import os; os.system('rm -rf /')  # should be blocked by firewall before here
    return model
"""

_SYNTAX_BROKEN = """\
def apply(model, config)
    return model
"""

_NO_APPLY = """\
x = 1 + 1
"""

if __name__ == "__main__":
    from cifar10.schemas import KernelProposal

    print("=" * 55)
    print("ARENA TEST SUITE (Python-exec mode)")
    print(f"cl.exe available: {_cl_available()}")
    print(f"nvcc available:   {_nvcc_available()}")
    print("=" * 55)

    all_ok = True

    def _make(binding: str) -> KernelProposal:
        return KernelProposal.model_validate({
            "kernel_type":          "conv_bn_relu_fusion",
            "layer_target":         "test",
            "cuda_kernel_code":     "// Python-only mode",
            "pytorch_binding":      binding,
            "integration_patch":    "model = apply(model, config)",
            "rationale":            "test",
            "expected_speedup_pct": 1,
        })

    print("\nTest 1: Valid torch.compile() binding...")
    try:
        k = compile_kernel(_make(_GOOD_BINDING), timeout_s=10)
        print(f"  PASS  elapsed={k.elapsed_s:.2f}s  ptx_hash={k.ptx_hash}")
    except CompilationError as e:
        print(f"  FAIL  {e.stderr[:100]}")
        all_ok = False

    print("\nTest 2: Syntax error binding raises CompilationError or ValidationError...")
    try:
        compile_kernel(_make(_SYNTAX_BROKEN), timeout_s=10)
        print("  FAIL  should have raised an error")
        all_ok = False
    except (CompilationError, Exception) as e:
        print(f"  PASS  {type(e).__name__}: {str(e)[:80]}")

    print("\nTest 3: No apply() function raises CompilationError...")
    try:
        compile_kernel(_make(_NO_APPLY), timeout_s=10)
        print("  FAIL  should have raised CompilationError")
        all_ok = False
    except (CompilationError, Exception) as e:
        print(f"  PASS  {type(e).__name__}: {str(e)[:80]}")

    print("\n" + "=" * 55)
    print(f"Result: {'ALL TESTS PASSED' if all_ok else 'FAILURES DETECTED'}")
    sys.exit(0 if all_ok else 1)
