"""
head_init.py — Final layer initialization (student-implemented).

Builds a stronger CIFAR100 linear head from frozen ImageNet ResNet18 features.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


_CIFAR100_SEARCH_PATHS = ("./data", "data", os.path.expanduser("~/data"))
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

_FEATURE_BATCH_SIZE = 64
_LINEAR_PROBE_BATCH_SIZE = 4096
_LINEAR_PROBE_EPOCHS = 80
_LINEAR_PROBE_LR = 1e-2
_LINEAR_PROBE_WEIGHT_DECAY = 1e-6
_LINEAR_PROBE_LABEL_SMOOTHING = 0.05
_LDA_REG = 1e-4
_FEATURE_CACHE_VERSION = "fulltrain_raw_features_v1"
_HEAD_CACHE_VERSION = "fulltrain_softmax_p05_v1"
_SURROGATE_NORM_POWER = 0.5


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the CIFAR100 head from frozen backbone features."""
    nn.init.orthogonal_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)

    data_dir = _find_data_dir()
    try:
        weight, bias = _load_or_fit_head(
            data_dir=data_dir,
            in_features=layer.in_features,
            num_classes=layer.out_features,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[head_init] linear probe failed: {exc}; falling back to LDA",
            file=sys.stderr,
            flush=True,
        )
        try:
            weight, bias = _build_lda_head(
                data_dir=data_dir,
                in_features=layer.in_features,
                num_classes=layer.out_features,
                reg=_LDA_REG,
            )
        except Exception as exc_lda:  # noqa: BLE001
            print(
                f"[head_init] LDA fallback failed: {exc_lda}; keeping orthogonal init",
                file=sys.stderr,
                flush=True,
            )
            return

    with torch.no_grad():
        layer.weight.copy_(weight.to(layer.weight.device, dtype=layer.weight.dtype))
        layer.bias.copy_(bias.to(layer.bias.device, dtype=layer.bias.dtype))


def _find_data_dir() -> str:
    for path in _CIFAR100_SEARCH_PATHS:
        if os.path.isdir(os.path.join(path, "cifar-100-python")):
            return path
    return "./data"


def _cache_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, ".head_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _feature_cache_path(data_dir: str) -> str:
    return os.path.join(_cache_dir(data_dir), f"train_features_{_FEATURE_CACHE_VERSION}.pt")


def _head_cache_path(data_dir: str, in_features: int, num_classes: int) -> str:
    name = f"head_{_HEAD_CACHE_VERSION}_{in_features}x{num_classes}.pt"
    return os.path.join(_cache_dir(data_dir), name)


def _load_or_fit_head(
    data_dir: str,
    in_features: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = _head_cache_path(data_dir, in_features, num_classes)
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        weight = blob["weight"].float()
        bias = blob["bias"].float()
        if weight.shape == (num_classes, in_features) and bias.shape == (num_classes,):
            return weight, bias

    features, labels = _load_or_extract_train_features(
        data_dir=data_dir,
        in_features=in_features,
        num_classes=num_classes,
    )
    weight, bias = _fit_linear_probe(
        features=features,
        labels=labels,
        in_features=in_features,
        num_classes=num_classes,
    )

    torch.save({"weight": weight.cpu(), "bias": bias.cpu()}, cache_path)
    return weight, bias


def _build_lda_head(
    data_dir: str,
    in_features: int,
    num_classes: int,
    reg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = _load_or_extract_train_features(
        data_dir=data_dir,
        in_features=in_features,
        num_classes=num_classes,
    )

    x = features.to(dtype=torch.float64)
    y = labels.long()

    counts = torch.bincount(y, minlength=num_classes).to(dtype=torch.float64)
    if (counts == 0).any():
        raise RuntimeError("at least one class has no usable features")

    mu = torch.zeros(num_classes, in_features, dtype=torch.float64)
    mu.index_add_(0, y, x)
    mu = mu / counts.unsqueeze(1)

    centered = x - mu[y]
    sigma = centered.T @ centered / float(x.shape[0])
    eye = torch.eye(in_features, dtype=torch.float64)

    weight = torch.linalg.solve(sigma + reg * eye, mu.T).T.float()
    bias = (-0.5 * (mu * weight.double()).sum(dim=1)).float()

    return weight, bias


def _extract_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _linear_probe_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_or_extract_train_features(
    data_dir: str,
    in_features: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = _feature_cache_path(data_dir)
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        features = blob.get("features", blob.get("train_x"))
        labels = blob.get("labels", blob.get("train_y"))
        if features is None or labels is None:
            raise KeyError("feature cache must contain features/labels or train_x/train_y")
        features = features.float()
        labels = labels.long()
        if features.ndim == 2 and features.shape[1] == in_features:
            return features, labels

    import torchvision.datasets as datasets
    import torchvision.models as models
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )
    dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=_FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = _extract_device()
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.eval().to(device).float()

    features = []
    labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            feats = backbone(images.to(device).float()).float().cpu()
            ok = torch.isfinite(feats).all(dim=1)
            if not ok.all():
                feats = feats[ok]
                labels = labels[ok]
            if labels.numel() == 0:
                continue
            features.append(feats)
            labels_all.append(labels.clone())

    if not features:
        raise RuntimeError("no finite features extracted from the train split")

    x = torch.cat(features, dim=0).float()
    y = torch.cat(labels_all, dim=0).long()
    if x.shape[1] != in_features:
        raise RuntimeError(f"expected {in_features} features, got {x.shape[1]}")
    if torch.bincount(y, minlength=num_classes).eq(0).any():
        raise RuntimeError("at least one class is missing from cached features")

    torch.save({"features": x, "labels": y}, cache_path)
    return x, y


def _surrogate_feature_map(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8).pow(_SURROGATE_NORM_POWER)


def _fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    in_features: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.utils.data import DataLoader, TensorDataset

    train_device = _linear_probe_device()
    x = _surrogate_feature_map(features).to(train_device)
    y = labels.to(train_device)

    probe = nn.Linear(in_features, num_classes).to(train_device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=_LINEAR_PROBE_LR,
        weight_decay=_LINEAR_PROBE_WEIGHT_DECAY,
    )
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=_LINEAR_PROBE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(0),
    )

    probe.train()
    for _ in range(_LINEAR_PROBE_EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                probe(batch_x),
                batch_y,
                label_smoothing=_LINEAR_PROBE_LABEL_SMOOTHING,
            )
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        return probe.weight.detach().cpu(), probe.bias.detach().cpu()
