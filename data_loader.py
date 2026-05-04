"""Dataset pipeline: fix the broken val split, compute class weights, build loaders."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

import config


# ImageNet normalization — required for torchvision pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ChestXRayDataset(Dataset):
    """Loads images from a DataFrame of (filepath, label) rows."""
    def __init__(self, df, img_size, train=False):
        self.df = df.reset_index(drop=True)
        self.label_map = {"NORMAL": 0, "PNEUMONIA": 1}

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transform(img)
        label = float(self.label_map[row["label"]])
        return img, torch.tensor(label, dtype=torch.float32)


def _list_images(folder):
    rows = []
    for label in config.CLASSES:
        class_dir = os.path.join(folder, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append((os.path.join(class_dir, fname), label))
    return rows


def fix_validation_split(val_size=0.2):
    """Merge train+val (the val folder is broken — only 16 images) and re-split 80/20 stratified."""
    train_rows = _list_images(config.TRAIN_DIR)
    val_rows   = _list_images(config.VAL_DIR)
    test_rows  = _list_images(config.TEST_DIR)

    df       = pd.DataFrame(train_rows + val_rows, columns=["filepath", "label"])
    test_df  = pd.DataFrame(test_rows, columns=["filepath", "label"])

    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["label"], random_state=config.RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Train class distribution:")
    print(train_df["label"].value_counts())
    return train_df, val_df, test_df


def compute_class_weights(train_df):
    """Balanced class weights for the ~1:3 NORMAL:PNEUMONIA imbalance."""
    y = train_df["label"].map({"NORMAL": 0, "PNEUMONIA": 1}).values
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.array([0, 1]), y=y)
    print(f"Class weights -> NORMAL: {weights[0]:.3f}, PNEUMONIA: {weights[1]:.3f}")
    # For BCEWithLogitsLoss we need a single pos_weight (weight for the positive class)
    pos_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32)
    return pos_weight


def get_data_loaders(train_df, val_df, test_df, img_size, batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_ds = ChestXRayDataset(train_df, img_size, train=True)
    val_ds   = ChestXRayDataset(val_df,   img_size, train=False)
    test_ds  = ChestXRayDataset(test_df,  img_size, train=False)

    common = dict(batch_size=batch_size, num_workers=config.NUM_WORKERS,
                  pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)
    return train_loader, val_loader, test_loader
