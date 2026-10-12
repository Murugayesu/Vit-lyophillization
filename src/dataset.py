"""
dataset.py — Shared PyTorch Dataset for all models.

Used by train_resnet.py, train_vit.py, and evaluate.py.
Never instantiate this directly — use get_loaders() instead.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE_VIT    = 224   # ViT-B/16 — 224px fits in 6 GB with batch 8
IMG_SIZE_RESNET = 224   # Keep identical for fair comparison

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class LyoDefectDataset(Dataset):
    """
    Dataset for lyophilized vial defect images using torchvision pipeline.
    """

    def __init__(self, df: pd.DataFrame, img_size: int = 224, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.tf = _build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            img_rgb = img.convert("RGB")
            img_tensor = self.tf(img_rgb)
        label = int(row["class_idx"])
        return img_tensor, label


def get_loaders(
    metadata_csv: str = "results/metrics/metadata.csv",
    img_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Returns (train_loader, val_loader, test_loader, class_names, num_classes).

    Reads the frozen metadata CSV produced by prepare_data.py.

    Pass pin_memory=False for CPU-only evaluation: pinning allocates CUDA host
    memory even when the model runs on CPU, which fails on memory-constrained
    machines with "CUDA error: out of memory" at the first batch.
    """
    meta = pd.read_csv(metadata_csv)

    train_df = meta[meta["split"] == "train"]
    val_df   = meta[meta["split"] == "val"]
    test_df  = meta[meta["split"] == "test"]

    train_ds = LyoDefectDataset(train_df, img_size=img_size, train=True)
    val_ds   = LyoDefectDataset(val_df,   img_size=img_size, train=False)
    test_ds  = LyoDefectDataset(test_df,  img_size=img_size, train=False)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # Recover class names from metadata
    idx_to_class = (
        meta[["class_idx", "defect_class"]]
        .drop_duplicates()
        .sort_values("class_idx")
        .set_index("class_idx")["defect_class"]
        .to_dict()
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    return train_loader, val_loader, test_loader, class_names, len(class_names)
