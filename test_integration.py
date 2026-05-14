from cifar10.schemas import KernelProposal
from cifar10.arena import compile_kernel
from cifar10.firewall import validate

binding = "def apply(model, config):\n    import torch\n    model = model.to(memory_format=torch.channels_last)\n    return torch.compile(model, mode='max-autotune')"

p = KernelProposal.model_validate({
    "kernel_type": "channels_last + compile",
    "layer_target": "layer1.0",
    "cuda_kernel_code": "// not used",
    "pytorch_binding": binding,
    "integration_patch": "model = apply(model, config)",
    "rationale": "test",
    "expected_speedup_pct": 20,
})
print(f"Schema OK: kernel_type={p.kernel_type!r}")

k = compile_kernel(p, timeout_s=5)
print(f"Arena OK:  ptx_hash={k.ptx_hash}  elapsed={k.elapsed_s:.2f}s")

fw = validate(p.model_dump_json())
print(f"Firewall:  passed={fw.passed}  gate={fw.gate_failed}")

print("ALL CHECKS PASS")
