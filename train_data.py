import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True
_SUBSET_SIZE = 8192


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY,
        download=True,
        transform=get_transforms(train=True),
    )

    subset_indices = torch.randperm(len(train_dataset), generator=generator_train)[:_SUBSET_SIZE]
    train_subset = Subset(train_dataset, subset_indices.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    return train_subset, train_loader
