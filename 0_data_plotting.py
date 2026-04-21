import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse

import data_utils.datasets as datasets
import datasets_aug.sequence_dataset as views
from datasets_aug.sequence_aug import *

plt.rcParams.update({
    "font.size": 12,          # base font size
    "axes.titlesize": 16,     # subplot title
    "axes.labelsize": 15,     # x and y labels
    "xtick.labelsize": 12,    # x-axis numbers
    "ytick.labelsize": 12,    # y-axis numbers
    "legend.fontsize": 15,    # legend
    "figure.titlesize": 16    # overall figure title
})


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
            "--normlizetype",
            type=str,
            choices=["zero_one", "minus_one_one", "mean_std", "mean"],
            default=None,
            help="Data normalization method",
    )

    return parser.parse_args()

args = parse_args()
args.data_dir = DATA_DIRS[args.data_name][0]

AUGMENTATIONS = [ "gaussian", "scale", "randomstrech", "randomcrop"]



def build_dataset(data_name,  data_dir, augmentation = "normal"):
    Dataset = getattr(datasets, data_name)
    dataset_view = views.OneViewDataset

    train_ds, val_ds, test_ds, classifier_ds, classifier_val_ds = Dataset(
         data_dir=data_dir,
        normlizetype=None,
        augmentype_1=augmentation,
        rand=1,
    ).data_prepare(
        split="RA",
        view=dataset_view,
    )

    return train_ds, test_ds


def normalize_signal(x, method):
    if method == "zero_one":
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    elif method == "minus_one_one":
        return 2 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1

    elif method == "mean_std":
        return (x - x.mean()) / (x.std() + 1e-8)

    elif method == "mean":
        return (x - x.mean())/(x.max() -x.min() + 1e-8)

    else:
        raise ValueError(f"Unknown normalization: {method}")


def find_sample_index_by_class(dataset, target_class):
    for i in range(len(dataset)):
        x, y = dataset[i]
        label = int(y) if torch.is_tensor(y) else y
        if label == target_class:
            return i
    raise ValueError(f"Class {target_class} not found in dataset.")


def plot_all_augmentations(args, target_class=0, save_path=None):
    datasets_by_aug = {}

    # build one dataset for each augmentation
    for aug in AUGMENTATIONS:
        train, test = build_dataset(
            data_name=args.data_name,
            data_dir=args.data_dir,
            augmentation=aug,
        )
        datasets_by_aug[aug] = train
    # also build "normal" (baseline / no augmentation)
    train_normal, _ = build_dataset(
        data_name=args.data_name,
        data_dir=args.data_dir,
        augmentation="normal",
    )


    sample_idx = find_sample_index_by_class(train_normal, target_class)

    # get reference signal once
    x_ref, y_ref = train_normal[sample_idx]
    if torch.is_tensor(x_ref):
        x_ref = x_ref.detach().cpu().squeeze().numpy()

   # x_ref = x_ref[:512]

    fig, axes = plt.subplots(
        len(AUGMENTATIONS), 1,
        figsize=(12, 3 * len(AUGMENTATIONS)),
        squeeze=False
    )

    for row, aug in enumerate(AUGMENTATIONS):
        x_aug, y = datasets_by_aug[aug][sample_idx]

        if torch.is_tensor(x_aug):
            x_aug = x_aug.detach().cpu().squeeze().numpy()

        #x_aug = x_aug[:512]

        ax = axes[row, 0]

        # plot original
        ax.plot(x_ref, label="original", color="royalblue", alpha = 0.6)

        # plot augmented
        ax.plot(x_aug, label=aug, color="royalblue", alpha = 1)

      #  ax.set_xlim(0, 256)
        ax.set_title(f"{aug} vs original | Class: {target_class}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()


def plot_all_augmentations_overlay(args, target_class=0, save_path=None):
    datasets_by_aug = {}

    # build one dataset per augmentation
    for aug in AUGMENTATIONS:
        train, test = build_dataset(
            data_name=args.data_name,
            data_dir=args.data_dir,
            augmentation=aug,
        )
        datasets_by_aug[aug] = train

    # find one sample index
    ref_dataset = datasets_by_aug[AUGMENTATIONS[0]]
    sample_idx = find_sample_index_by_class(ref_dataset, target_class)

    plt.figure(figsize=(12, 5))
    i = 1
    for aug in AUGMENTATIONS:
        x, y = datasets_by_aug[aug][sample_idx]

        if torch.is_tensor(x):
            x = x.detach().cpu().squeeze().numpy()

        # plot only part of the signal (same as your subplot version)
        x = x[:512] + i
        i+=1

        plt.plot(x, label=aug, alpha=0.8)

    plt.title(f"All Augmentations (Overlay) | Class: {target_class} | Sample idx: {sample_idx}")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude + Offset")
    #plt.xlim(0, 512)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

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
        ax.plot(signal, color= "royalblue")
        ax.set_title(f"Class {label}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()



def plot_normalizations(dataset, save_path, target_class=0):
    # --- find one sample from the target class ---
    for i in range(len(dataset)):
        x, y = dataset[i]
        label = int(y) if torch.is_tensor(y) else y

        if label == target_class:
            if torch.is_tensor(x):
                x = x.detach().cpu().squeeze().numpy()
            break
    else:
        raise ValueError(f"Class {target_class} not found in dataset")

    methods = ["zero_one", "minus_one_one", "mean_std", "mean"]

    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(12, 3 * (len(methods) + 1)))

    # original
    axes[0].plot(x)
    axes[0].set_title("Original")
    axes[0].grid(True)

    # normalized versions
    for i, method in enumerate(methods, start=1):
        x_norm = normalize_signal(x, method)
        axes[i].plot(x_norm, color= "royalblue")
        axes[i].set_title(method)
        axes[i].grid(True)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


args = parse_args()
args.data_dir = DATA_DIRS[args.data_name][0]

train, test = build_dataset(data_name = args.data_name, data_dir=args.data_dir, augmentation="normal")



plot_each_class(train, save_path="figures/data_vis/CWRU_all_samples.pdf")

plot_all_augmentations( args,  target_class=0, save_path="figures/data_vis/CWRU_augmentations.pdf")

#plot_normalizations(train, save_path ="figures/data_vis/PU_all_normalizations.pdf",  target_class=0)
#plot_all_augmentations_overlay(args, target_class=0, save_path="figures/data_vis/PU_augmentations_overlay.pdf")