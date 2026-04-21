import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse

import data_utils.datasets as datasets
import datasets_aug.sequence_dataset as views
from datasets_aug.sequence_aug import *


DATA_DIRS = {
    "CWRU": [r"raw_data/CWRU", 4],
    "JNU": [r"raw_data/JNU/JNU-Bearing-Dataset-main", 4], # {"healthy": 0, "inner_race": 1, "outer_race": 2, "ball" : 3}
    "PU": [r"raw_data/PU", 6], # {"healthy": 0, "inner_race": 1, "combined": 2, "outer_race": 3}
    "SEU": [r"raw_data/SEU/gearbox", 5], # {"healthy": 0, "inner_race": 1, "combined": 2, "outer_race": 3, "ball" : 4}
    "XJTU": [r"raw_data/XJTU/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets", 4]  # TODO: Check this 
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
            "--data_name", # SEU, JNU ,PU , CWRU
            type=str,
            choices=DATA_DIRS.keys(),
            default="CWRU",  # SEU, JNU ,PU , CWRU
            help="The name of the dataset",
        )

    parser.add_argument(
        "--aug_1",
        type=str,
        choices=["gaussian", "normal", "scale", "randomstrech", "randomcrop", "fft"],
        default="normal",
        help="Augmentation type on the online pipeline",
    )
    parser.add_argument(
            "--normlizetype",
            type=str,
            choices=["zero_one", "minus_one_one", "mean_std", "mean"],
            default="mean_std",
            help="Data normalization method",
    )

    return parser.parse_args()

args = parse_args()
args.data_dir = DATA_DIRS[args.data_name][0]


def data_loading(args):
    Dataset = getattr(datasets, args.data_name)
    dataset_view = views.OneViewDataset

    train_ds, val_ds, test_ds, classifier_ds, classifier_val_ds = Dataset(
        data_dir=args.data_dir,
        normlizetype=args.normlizetype,
        augmentype_1=args.aug_1,
        rand=42,
    ).data_prepare(
        split="O_N",
        view=dataset_view,
    )

    return train_ds, test_ds

def plot_samples(dataset, num_samples=5, class_names=None, save_path=None, random_samples=False):
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    if random_samples:
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
    else:
        indices = list(range(min(num_samples, len(dataset))))

    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 3 * len(indices)), squeeze=False)

    for row, idx in enumerate(indices):
        x, y = dataset[idx]

        if torch.is_tensor(x):
            x = x.detach().cpu().squeeze().numpy()

        label_value = int(y) if torch.is_tensor(y) else y

        if class_names is None:
            label_str = str(label_value)
        elif isinstance(class_names, dict):
            label_str = class_names.get(label_value, str(label_value))
        else:
            if 0 <= label_value < len(class_names):
                label_str = class_names[label_value]
            else:
                label_str = str(label_value)

        ax = axes[row, 0]
        ax.plot(x)
        ax.set_title(f"Label: {label_str}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_each_class(dataset, save_path):
    class_samples = {}

    # collect first occurrence of each class
    for i in range(len(dataset)):
        x, y = dataset[i]
        label = int(y) if torch.is_tensor(y) else y

        if label not in class_samples:
            if torch.is_tensor(x):
                x = x.detach().cpu().squeeze().numpy()
            class_samples[label] = x

        # stop early if all classes are found
        if len(class_samples) == len(set(class_samples.keys())):
            continue

    n_classes = len(class_samples)

    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 3 * n_classes))

    # handle case when only 1 class
    if n_classes == 1:
        axes = [axes]

    for ax, (label, signal) in zip(axes, sorted(class_samples.items())):
        ax.plot(signal)
        ax.set_title(f"Class {label}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


args = parse_args()
args.data_dir = DATA_DIRS[args.data_name][0]

train, test = data_loading(args)

plot_each_class(test, save_path="figures/data_vis/all_samples")


