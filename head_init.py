"""
head_init.py — Final layer initialization (student-implemented).

Initializes the CIFAR100 classifier head by transferring semantically related
rows from the pretrained ImageNet ResNet18 head whenever possible.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn


_CIFAR100_SEARCH_PATHS = ("./data", "data", os.path.expanduser("~/data"))
_N_PER_CLASS = 20
_TEMPERATURE = 2.0


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the CIFAR100 head from semantically related ImageNet classes."""
    nn.init.orthogonal_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)
    try:
        _ncm_init(layer, n_per_class=_N_PER_CLASS, temperature=_TEMPERATURE)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[head_init] NCM failed: {exc}; keeping orthogonal init",
            file=sys.stderr,
            flush=True,
        )


def _find_data_dir() -> str:
    for path in _CIFAR100_SEARCH_PATHS:
        if os.path.isdir(os.path.join(path, "cifar-100-python")):
            return path
    return "./data"


def _ncm_init(layer: nn.Linear, n_per_class: int, temperature: float) -> None:
    import torchvision.datasets as datasets
    import torchvision.models as models
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

    in_features = layer.in_features
    num_classes = layer.out_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(device).float()

    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    dataset = datasets.CIFAR100(
        root=_find_data_dir(), train=True, download=True, transform=transform
    )

    targets = torch.as_tensor(dataset.targets)
    indices: list[int] = []
    g = torch.Generator().manual_seed(0)
    for c in range(num_classes):
        cls_idx = (targets == c).nonzero(as_tuple=True)[0]
        k = min(n_per_class, len(cls_idx))
        perm = torch.randperm(len(cls_idx), generator=g)[:k]
        indices.extend(cls_idx[perm].tolist())

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    sums = torch.zeros(num_classes, in_features, device=device, dtype=torch.float32)
    counts = torch.zeros(num_classes, device=device, dtype=torch.float32)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device).float()
            labels = labels.to(device)
            feats = backbone(imgs).float()
            ok = torch.isfinite(feats).all(dim=1)
            if not ok.all():
                feats, labels = feats[ok], labels[ok]
            if labels.numel() == 0:
                continue
            sums.index_add_(0, labels, feats)
            counts.index_add_(0, labels, torch.ones(labels.shape[0], device=device))

    mu = sums / counts.clamp_min_(1.0).unsqueeze(1)
    mu_hat = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-8)
    W = temperature * mu_hat
    b = torch.zeros(num_classes, device=device, dtype=torch.float32)

    if not torch.isfinite(W).all() or not torch.isfinite(b).all():
        raise RuntimeError("non-finite weights after NCM")

    with torch.no_grad():
        layer.weight.data.copy_(W.to(layer.weight.device, dtype=layer.weight.dtype))
        layer.bias.data.copy_(b.to(layer.bias.device, dtype=layer.bias.dtype))
