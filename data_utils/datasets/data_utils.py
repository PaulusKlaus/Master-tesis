import pandas as pd

from datasets_aug.sequence_dataset import *  
from datasets_aug.sequence_aug import *

def cap_per_class(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """
    Keep up to n_per_class per label (no oversampling).
    If a class has fewer than n_per_class, keep all of them.
    """
    if n_per_class is None:
        return df

    parts = []
    for label, group in df.groupby("label", sort=False):
        k = min(len(group), n_per_class)
        parts.append(group.sample(n=k, random_state=seed, replace=False))

    return pd.concat(parts, ignore_index=True)



def data_transforms(transform="fft", normlize_type="minus_one_one"):
    "A composed transform pipeline for dataset preprocessing / augmentation depending on whether it's training or validation."
    transforms = {
        'gaussian': Compose([
            Reshape(),
            Normalize(normlize_type),
            AddGaussian(),
            Retype()
        ]),
        'normal': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ]),
        'scale': Compose([
            Reshape(),
            Normalize(normlize_type),
            Scale(),
            Retype()
        ]),
        'randomstrech': Compose([
            Reshape(),
            Normalize(normlize_type),
            RandomStretch(),
            Retype()
        ]),
        'randomcrop': Compose([
            Reshape(),
            Normalize(normlize_type),
            RandomCrop(),
            Retype()
        ]),
        'fft': Compose([
            Reshape(),
            Normalize(normlize_type),
            FFT(),
            Retype()
        ]),
    }
    return transforms[transform]


def train_test_split_order(data_pd, test_size=0.3, labels=None):
    """
    Ordered split within each class:
    - first (1 - val_size) fraction -> train
    - last val_size fraction -> val
    Preserves the existing order in data_pd for each class.
    """
    if labels is None:
        labels = sorted(data_pd["label"].unique())

    train_parts = []
    val_parts = []

    for lbl in labels:
        cls = data_pd[data_pd["label"] == lbl].reset_index(drop=True)
        n = len(cls)
        if n == 0:
            continue

        cut = int((1 - test_size) * n)  # number of train samples
        train_parts.append(cls.iloc[:cut][["data", "label"]])
        val_parts.append(cls.iloc[cut:][["data", "label"]])

    train_pd = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=["data","label"])
    test_pd   = pd.concat(val_parts,   ignore_index=True) if val_parts else pd.DataFrame(columns=["data","label"])
    return train_pd, test_pd
