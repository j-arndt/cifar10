"""Agentic Firewall — 5-gate sequential validation before any kernel touches a compiler."""
from __future__ import annotations

import ast
import json
import re
import sys
from typing import Optional

from pydantic import ValidationError

from cifar10.schemas import KernelProposal, FirewallResult


# --------------------------------------------------------------------------- #
# Blocked patterns — checked in both cuda_kernel_code and pytorch_binding
# --------------------------------------------------------------------------- #
_SAFETY_PATTERNS = [
    r"\bos\.",
    r"\bsubprocess\.",
    r"\bsocket\.",
    r"\bopen\s*\(",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bimportlib\b",
    r"\bctypes\b",
    r"\bshlex\b",
]

_HARD_PATTERNS = [
    r"\brm\s+",
    r"\bshutil\.",
    r"\/etc\/",
    r"C:\\\\",
    r"\bformat\s*\(",       # format() can be used for path injection
    r"\bpathlib\b",
    r"\bglob\b",
]

_COMPILED_SAFETY = [re.compile(p) for p in _SAFETY_PATTERNS]
_COMPILED_HARD   = [re.compile(p) for p in _HARD_PATTERNS]


# --------------------------------------------------------------------------- #
# Gate implementations
# --------------------------------------------------------------------------- #

def gate_schema(text: str) -> FirewallResult:
    """Gate 1 — JSON parse + Pydantic validation."""
    # Extract first JSON object from the text (agent may wrap in prose)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return FirewallResult(
            passed=False, gate_failed="SCHEMA",
            error_message="No JSON object found in agent output",
        )
    try:
        proposal = KernelProposal.model_validate_json(match.group())
        return FirewallResult(passed=True, proposal=proposal)
    except (ValidationError, json.JSONDecodeError) as e:
        return FirewallResult(
            passed=False, gate_failed="SCHEMA",
            error_message=f"Schema validation failed:\n{e}",
        )


def gate_syntax(proposal: KernelProposal) -> FirewallResult:
    """Gate 2 — pytorch_binding must be valid Python AST. (Already checked in validator, belt-and-suspenders.)"""
    try:
        ast.parse(proposal.pytorch_binding)
        return FirewallResult(passed=True, proposal=proposal)
    except SyntaxError as e:
        return FirewallResult(
            passed=False, gate_failed="SYNTAX",
            error_message=f"Python syntax error at line {e.lineno}: {e.msg}",
        )


def gate_safety(proposal: KernelProposal) -> FirewallResult:
    """Gate 3 — blocked API patterns in both code fields."""
    targets = {
        "cuda_kernel_code":  proposal.cuda_kernel_code,
        "pytorch_binding":   proposal.pytorch_binding,
        "integration_patch": proposal.integration_patch,
    }
    for field_name, code in targets.items():
        for pattern in _COMPILED_SAFETY:
            if pattern.search(code):
                return FirewallResult(
                    passed=False, gate_failed="SAFETY",
                    error_message=f"Blocked pattern '{pattern.pattern}' in {field_name}. "
                                  "System calls are not permitted inside kernels.",
                )
    return FirewallResult(passed=True, proposal=proposal)


def gate_size(proposal: KernelProposal) -> FirewallResult:
    """Gate 4 — CUDA kernel code size must be < 64KB."""
    size = len(proposal.cuda_kernel_code.encode("utf-8"))
    if size >= 65536:
        return FirewallResult(
            passed=False, gate_failed="SIZE",
            error_message=f"Kernel too large: {size:,} bytes. Maximum is 65,536 bytes.",
        )
    return FirewallResult(passed=True, proposal=proposal)


def gate_pattern(proposal: KernelProposal) -> FirewallResult:
    """Gate 5 — hard-blocked patterns (destructive or path-traversal)."""
    targets = {
        "cuda_kernel_code":  proposal.cuda_kernel_code,
        "pytorch_binding":   proposal.pytorch_binding,
        "integration_patch": proposal.integration_patch,
        "rationale":         proposal.rationale,
    }
    for field_name, code in targets.items():
        for pattern in _COMPILED_HARD:
            if pattern.search(code):
                return FirewallResult(
                    passed=False, gate_failed="PATTERN",
                    error_message=f"Blocked pattern '{pattern.pattern}' in {field_name}.",
                )
    return FirewallResult(passed=True, proposal=proposal)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def validate(text: str) -> FirewallResult:
    """Run all 5 gates in sequence. Return first failure or final pass."""
    result = gate_schema(text)
    if not result:
        return result

    for gate in (gate_syntax, gate_safety, gate_size, gate_pattern):
        result = gate(result.proposal)
        if not result:
            return result

    return result


# --------------------------------------------------------------------------- #
# Test suite
# --------------------------------------------------------------------------- #

_ADVERSARIAL_PAYLOADS = [
    # 1. No JSON at all
    ("no_json", "Here is my kernel idea: just use torch.compile()"),
    # 2. Wrong kernel_type enum
    ("bad_enum", json.dumps({
        "kernel_type": "teleport_kernel", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){}", "pytorch_binding": "def f(): pass",
        "integration_patch": "x = f(x)", "rationale": "fast", "expected_speedup_pct": 10,
    })),
    # 3. Syntax error in pytorch_binding
    ("syntax_error", json.dumps({
        "kernel_type": "conv_bn_relu_fusion", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){}", "pytorch_binding": "def f(: bad syntax",
        "integration_patch": "x = f(x)", "rationale": "fast", "expected_speedup_pct": 10,
    })),
    # 4. subprocess in cuda code
    ("subprocess_cuda", json.dumps({
        "kernel_type": "conv_bn_relu_fusion", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){ subprocess.run(['ls']); }",
        "pytorch_binding": "def f(): pass",
        "integration_patch": "x = f(x)", "rationale": "fast", "expected_speedup_pct": 10,
    })),
    # 5. os.system in binding
    ("os_in_binding", json.dumps({
        "kernel_type": "conv_bn_relu_fusion", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){}",
        "pytorch_binding": "def f(): os.system('whoami')",
        "integration_patch": "x = f(x)", "rationale": "fast", "expected_speedup_pct": 10,
    })),
    # 6. Kernel too large (> 65536 bytes) — 6000 lines * ~12 bytes = ~72KB
    ("too_large", json.dumps({
        "kernel_type": "conv_bn_relu_fusion", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){" + "// padding\n" * 6000 + "}",
        "pytorch_binding": "def f(): pass",
        "integration_patch": "x = f(x)", "rationale": "fast", "expected_speedup_pct": 10,
    })),
    # 7. rm in integration patch
    ("rm_in_patch", json.dumps({
        "kernel_type": "conv_bn_relu_fusion", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){}",
        "pytorch_binding": "def f(): pass",
        "integration_patch": "rm -rf /tmp/cache", "rationale": "fast", "expected_speedup_pct": 10,
    })),
    # 8. eval() in binding
    ("eval_in_binding", json.dumps({
        "kernel_type": "conv_bn_relu_fusion", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){}",
        "pytorch_binding": "def f(): return eval('__import__(\"os\")')",
        "integration_patch": "x = f(x)", "rationale": "fast", "expected_speedup_pct": 10,
    })),
    # 9. Missing required field
    ("missing_field", json.dumps({
        "kernel_type": "conv_bn_relu_fusion", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){}",
        # missing pytorch_binding, integration_patch, rationale, expected_speedup_pct
    })),
    # 10. speedup_pct out of range
    ("bad_speedup", json.dumps({
        "kernel_type": "conv_bn_relu_fusion", "layer_target": "layer1",
        "cuda_kernel_code": "__global__ void k(){}",
        "pytorch_binding": "def f(): pass",
        "integration_patch": "x = f(x)", "rationale": "fast",
        "expected_speedup_pct": 9999,
    })),
]

_VALID_PAYLOAD = json.dumps({
    "kernel_type": "conv_bn_relu_fusion",
    "layer_target": "layer1.0",
    "cuda_kernel_code": "__global__ void fused_cbr(float* x, float* w, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = fmaxf(0.0f, x[i] * w[i]); }",
    "pytorch_binding": "import torch\ndef forward(x, w):\n    return torch.clamp(x * w, min=0)",
    "integration_patch": "out = forward(x, weight)",
    "rationale": "Fuse conv + BN + ReLU to avoid extra memory passes",
    "expected_speedup_pct": 25,
})


if __name__ == "__main__":
    print("=" * 60)
    print("FIREWALL TEST SUITE — 10 adversarial + 1 valid")
    print("=" * 60)

    all_passed = True

    for name, payload in _ADVERSARIAL_PAYLOADS:
        result = validate(payload)
        status = "✓ BLOCKED" if not result.passed else "✗ PASSED (SHOULD HAVE BLOCKED)"
        if result.passed:
            all_passed = False
        gate = result.gate_failed or "—"
        print(f"  [{gate:<10}] {name:<25} {status}")
        if result.passed:
            print(f"             ERROR: adversarial payload was NOT blocked!")

    # Valid payload must pass
    result = validate(_VALID_PAYLOAD)
    valid_ok = result.passed
    status = "✓ PASSED" if valid_ok else "✗ BLOCKED (SHOULD HAVE PASSED)"
    if not valid_ok:
        all_passed = False
        print(f"  [{'—':<10}] {'valid_payload':<25} {status}")
        print(f"             Gate: {result.gate_failed}  Error: {result.error_message}")
    else:
        print(f"  [{'PASS':<10}] {'valid_payload':<25} {status}")

    print("=" * 60)
    print(f"Result: {'ALL TESTS PASSED ✓' if all_passed else 'FAILURES DETECTED ✗'}")
    sys.exit(0 if all_passed else 1)
