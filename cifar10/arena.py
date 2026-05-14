"""Compilation sandbox — load_inline wrapper with timeout, PTX hash capture."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from cifar10.schemas import KernelProposal, KernelType


class CompilationError(Exception):
    def __init__(self, stderr: str, elapsed_s: float = 0.0):
        super().__init__(stderr)
        self.stderr    = stderr
        self.elapsed_s = elapsed_s


@dataclass
class CompiledKernel:
    module:      object       # the loaded torch extension
    ptx_hash:    str          # SHA256 of compiled PTX bytes
    kernel_type: KernelType
    build_dir:   str
    elapsed_s:   float        # compile wall time


def _compute_ptx_hash(build_dir: str) -> str:
    """Walk the build dir, find .ptx or .cubin files, hash them all."""
    build_path = Path(build_dir)
    ptx_files  = sorted(build_path.rglob("*.ptx")) + sorted(build_path.rglob("*.cubin"))
    if not ptx_files:
        # Fall back to hashing all .so files
        ptx_files = sorted(build_path.rglob("*.so")) + sorted(build_path.rglob("*.pyd"))
    if not ptx_files:
        return hashlib.sha256(b"no-ptx").hexdigest()[:16]

    h = hashlib.sha256()
    for f in ptx_files:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def compile_kernel(proposal: KernelProposal, timeout_s: int = 30) -> CompiledKernel:
    """
    Compile a KernelProposal via torch.utils.cpp_extension.load_inline.

    Raises CompilationError on nvcc failure or timeout.
    Returns CompiledKernel on success.
    """
    from torch.utils.cpp_extension import load_inline

    code_hash = hashlib.sha256(proposal.cuda_kernel_code.encode()).hexdigest()[:8]
    build_dir  = str(Path("kernels") / "build" / code_hash)
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # CUDA arch for RTX 4060 Mobile (Ada Lovelace = sm_89)
    extra_cuda_cflags = ["-O3", "-arch=sm_89", "-use_fast_math", "--expt-relaxed-constexpr"]

    result: dict = {}
    error:  dict = {}

    def _compile():
        try:
            t0  = time.perf_counter()
            mod = load_inline(
                name=f"kernel_{code_hash}",
                cpp_sources=[""],          # empty C++ stub
                cuda_sources=[proposal.cuda_kernel_code],
                extra_cuda_cflags=extra_cuda_cflags,
                build_directory=build_dir,
                verbose=False,
            )
            result["module"]    = mod
            result["elapsed_s"] = time.perf_counter() - t0
        except Exception as e:
            error["msg"]       = str(e)
            error["elapsed_s"] = time.perf_counter() - time.perf_counter()

    t = threading.Thread(target=_compile, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout=timeout_s)

    elapsed = time.perf_counter() - t0

    if t.is_alive():
        raise CompilationError(
            f"TIMEOUT: compilation exceeded {timeout_s}s. "
            f"Simplify the kernel or reduce template instantiations.",
            elapsed_s=elapsed,
        )

    if error:
        raise CompilationError(error.get("msg", "Unknown compilation error"), elapsed_s=elapsed)

    ptx_hash = _compute_ptx_hash(build_dir)

    return CompiledKernel(
        module=result["module"],
        ptx_hash=ptx_hash,
        kernel_type=proposal.kernel_type,
        build_dir=build_dir,
        elapsed_s=result.get("elapsed_s", elapsed),
    )


# --------------------------------------------------------------------------- #
# Test suite
# --------------------------------------------------------------------------- #

_TRIVIAL_KERNEL = """
#include <cuda_runtime.h>
#include <math.h>

__global__ void identity_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}
"""

_BROKEN_KERNEL = """
__global__ void broken(float* x {{{ SYNTAX ERROR HERE
"""

def _make_trivial_proposal():
    from cifar10.schemas import KernelProposal, KernelType
    import json
    return KernelProposal.model_validate({
        "kernel_type":          "conv_bn_relu_fusion",
        "layer_target":         "test",
        "cuda_kernel_code":     _TRIVIAL_KERNEL,
        "pytorch_binding":      "def forward(x): return x",
        "integration_patch":    "out = forward(x)",
        "rationale":            "identity kernel for testing",
        "expected_speedup_pct": 1,
    })


if __name__ == "__main__":
    print("=" * 55)
    print("ARENA TEST SUITE")
    print("=" * 55)

    all_ok = True

    # Test 1: valid kernel compiles
    print("\nTest 1: Compile valid identity kernel...")
    try:
        proposal = _make_trivial_proposal()
        kernel   = compile_kernel(proposal, timeout_s=60)
        print(f"  ✓ Compiled in {kernel.elapsed_s:.1f}s  PTX hash: {kernel.ptx_hash}")
    except CompilationError as e:
        print(f"  ✗ FAILED: {e.stderr[:200]}")
        all_ok = False

    # Test 2: broken kernel raises CompilationError
    print("\nTest 2: Broken kernel raises CompilationError...")
    try:
        bad_proposal = _make_trivial_proposal()
        bad_proposal = bad_proposal.model_copy(update={"cuda_kernel_code": _BROKEN_KERNEL})
        compile_kernel(bad_proposal, timeout_s=30)
        print("  ✗ FAILED: should have raised CompilationError")
        all_ok = False
    except CompilationError as e:
        print(f"  ✓ CompilationError raised correctly")
        print(f"    stderr[:100]: {str(e)[:100]}")
    except Exception as e:
        print(f"  ✗ UNEXPECTED exception: {type(e).__name__}: {e}")
        all_ok = False

    print("\n" + "=" * 55)
    print(f"Result: {'ALL TESTS PASSED ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    sys.exit(0 if all_ok else 1)
