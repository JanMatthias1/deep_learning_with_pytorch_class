"""Data loading and augmentation for CIFAR-100 with Swin Transformer."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Data augmentation on train set, and normalisation for SWin Architecture

def get_train_transform():
    """
    Training transform:
      - Resize 32x32 → 224x224 (Swin expects this)
      - RandomHorizontalFlip: standard for CIFAR
      - RandomCrop with padding: spatial jitter
      - RandAugment: automates augmentation policy search;
        cleaner than hand-picking color jitter / rotation combos
      - Normalize to ImageNet stats (standard for models operating at 224x224)
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 channel means
            std=[0.2675, 0.2565, 0.2761],   # CIFAR-100 channel stds
        ),
    ])


def get_eval_transform():
    """
    Validation / test transform: resize + normalize only. No augmentation needed here.
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])


def get_dataloaders(
    data_root: str = "data",
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
):
    """
    Returns (train_loader, val_loader, test_loader).

    The raw 50k training set is split 45k/5k for train/val using a fixed seed
    so the split is reproducible across runs.

    Note: torchvision applies the transform at __getitem__ time, but
    random_split works on indices, so we need both subsets to share the
    underlying dataset object. We handle the different transforms by
    wrapping each subset. See _TransformSubset below.
    """

    # Download once with no transform — we'll apply transforms per-subset
    full_train = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=None
    )
    test_data = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=get_eval_transform()
    )

    # 90/10 split
    n_val = int(len(full_train) * val_fraction)
    train_subset, val_subset = random_split(
        full_train,
        [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Wrap subsets with appropriate transforms
    train_set = _TransformSubset(train_subset, get_train_transform())
    val_set = _TransformSubset(val_subset, get_eval_transform())

    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True,
        batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, shuffle=False,
        batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_data, shuffle=False,
        batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_set)}  |  Val: {len(val_set)}  |  Test: {len(test_data)}")
    return train_loader, val_loader, test_loader

class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a torch Subset and applies a transform at access time."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    