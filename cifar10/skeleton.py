"""Baseline skeleton training loop — the starting point handed to the agent."""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from cifar10.data import get_cifar10_loaders, CutoutAugmentation
from cifar10.network import ResNet9


def train_one_epoch(model, loader, optimizer, scaler, scheduler, cutout, device):
    model.train()
    correct = total = 0
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = cutout(x)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out  = model(x)
            loss = nn.functional.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        correct += (out.argmax(1) == y).sum().item()
        total   += y.size(0)
        total_loss += loss.item()
    return {"loss": total_loss / len(loader), "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total   += y.size(0)
    return {"accuracy": correct / total}


def run_skeleton(config: dict | None = None, apply_fn=None) -> dict:
    if config is None:
        config = yaml.safe_load(Path("config.yaml").read_text())

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs     = config.get("epochs", 30)
    batch_size = config.get("batch_size", 512)

    model  = ResNet9().to(device)
    scaler = torch.amp.GradScaler('cuda')
    cutout = CutoutAugmentation(size=8)

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    # ── Apply agent's pytorch_binding (torch.compile, fused ops, etc.) ──
    if apply_fn is not None:
        try:
            # Suppress Triton/dynamo errors — falls back to eager if compile fails
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = apply_fn(model, config)
            print("[skeleton] pytorch_binding applied successfully")
        except Exception as e:
            print(f"[skeleton] pytorch_binding apply failed (falling back to baseline): {e}")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.5,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )

    t0 = time.perf_counter()
    try:
        for epoch in range(epochs):
            train_stats = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, cutout, device)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                test_stats = evaluate(model, test_loader, device)
                print(f"Epoch {epoch+1:2d}/{epochs}  loss={train_stats['loss']:.4f}  "
                      f"train_acc={train_stats['accuracy']:.4f}  test_acc={test_stats['accuracy']:.4f}")
    except Exception as train_err:
        if apply_fn is not None:
            # Compiled/patched model crashed during training — retry with baseline
            print(f"[skeleton] Training crashed ({train_err}). Retrying with baseline model...")
            model  = ResNet9().to(device)
            scaler = torch.amp.GradScaler('cuda')
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4, nesterov=True
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.5, steps_per_epoch=len(train_loader),
                epochs=epochs, pct_start=0.2, anneal_strategy="cos",
                div_factor=25, final_div_factor=1e4,
            )
            t0 = time.perf_counter()  # restart timer
            for epoch in range(epochs):
                train_stats = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, cutout, device)
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    test_stats = evaluate(model, test_loader, device)
        else:
            raise  # no fallback available

    wall_time = time.perf_counter() - t0
    final_acc  = evaluate(model, test_loader, device)["accuracy"]

    result = {
        "accuracy":    round(final_acc, 4),
        "wall_time_s": round(wall_time, 3),
        "passed":      final_acc >= 0.940 and wall_time < 1.0,
        "epochs":      epochs,
        "batch_size":  batch_size,
    }
    print(f"\nBaseline result: accuracy={result['accuracy']}  wall_time={result['wall_time_s']}s  passed={result['passed']}")
    return result


if __name__ == "__main__":
    result = run_skeleton()
    log_path = Path("logs/baseline.json")
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(result, indent=2))
    print(f"Saved to {log_path}")
