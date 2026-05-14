"""CIFAR-10 data pipeline — GPU-native augmentation, pinned memory."""
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size: int = 512, num_workers: int = 4, data_dir: str = "./data"):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader


class CutoutAugmentation:
    """Zero-mask a random square on GPU tensor (N, C, H, W)."""
    def __init__(self, size: int = 8):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        y0 = torch.randint(0, h, (n,), device=x.device)
        x0 = torch.randint(0, w, (n,), device=x.device)
        y1 = (y0 + self.size).clamp(max=h)
        x1 = (x0 + self.size).clamp(max=w)
        out = x.clone()
        for i in range(n):
            out[i, :, y0[i]:y1[i], x0[i]:x1[i]] = 0.0
        return out


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    train_loader, test_loader = get_cifar10_loaders()
    # burn one batch to trigger worker startup
    for x, y in train_loader:
        break
    elapsed = time.perf_counter() - t0
    print(f"First batch ready in {elapsed:.2f}s  (target: <2s)")
    print(f"Train batches: {len(train_loader)}  Test batches: {len(test_loader)}")
