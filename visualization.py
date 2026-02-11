import random

import matplotlib.pyplot as plt
import torch

# Local imports
from data_utils.datasets import *
from datasets_aug import *

import torch
from torch.utils.data import Dataset

def _unpack_item(item):
    """
    Handles common dataset item formats:
      - (x, y)
      - (x, y, meta...)
      - {"x":..., "y":...} or {"data":..., "label":...}
    Returns: (x, y)
    """
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return item[0], item[1]

    if isinstance(item, dict):
        # try a few common keys
        x = None
        y = None
        for xk in ("x", "X", "data", "input", "inputs"):
            if xk in item:
                x = item[xk]
                break
        for yk in ("y", "Y", "label", "labels", "target", "targets", "lab"):
            if yk in item:
                y = item[yk]
                break
        if x is not None and y is not None:
            return x, y

    raise TypeError(f"Don't know how to unpack dataset item type: {type(item)} (value={item})")


def get_Xy(ds, max_items=None, device="cpu"):
    """
    Adds support for torch Dataset-like objects (e.g., OneViewDataset).
    If ds is a Dataset, we iterate and stack.
    max_items: if set, only loads that many samples (fast for visualization).
    """
    # (X, y)
    if isinstance(ds, (tuple, list)) and len(ds) == 2:
        X, y = ds
        return torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)

    # dict container
    if isinstance(ds, dict):
        for xk in ("X", "x", "data", "inputs"):
            for yk in ("y", "labels", "lab", "target", "targets"):
                if xk in ds and yk in ds:
                    return torch.as_tensor(ds[xk], device=device), torch.as_tensor(ds[yk], device=device)

    # TensorDataset-like
    if hasattr(ds, "tensors"):
        X, y = ds.tensors[0], ds.tensors[1]
        return X.to(device), y.to(device)

    # your custom datasets
    if hasattr(ds, "_get_files"):
        data, lab = ds._get_files()
        return torch.as_tensor(data, device=device), torch.as_tensor(lab, device=device)

    # Generic PyTorch Dataset (this is your OneViewDataset case)
    if isinstance(ds, Dataset) or (hasattr(ds, "__len__") and hasattr(ds, "__getitem__")):
        n = len(ds)
        if max_items is None:
            max_items = n
        max_items = min(int(max_items), n)

        xs, ys = [], []
        for i in range(max_items):
            x_i, y_i = _unpack_item(ds[i])
            xs.append(torch.as_tensor(x_i))
            ys.append(torch.as_tensor(y_i))

        X = torch.stack(xs, dim=0).to(device)
        y = torch.stack(ys, dim=0).view(-1).to(device)
        return X, y

    raise TypeError(f"Unsupported dataset type: {type(ds)}")



def visualize_sample(dataset_split, label, idx=None, seed=0, title=None):
    """
    dataset_split: train_dataset / val_dataset / test_dataset OR anything supported by get_Xy()
    label: int label to filter
    idx: optional explicit index within the filtered set (not global index)
    """

    X, y = get_Xy(dataset_split)

    # ensure y is 1D
    y = y.view(-1)

    # find indices for chosen label
    label = int(label)
    indices = (y == label).nonzero(as_tuple=True)[0].cpu().numpy()

    if len(indices) == 0:
        unique = torch.unique(y).cpu().tolist()
        raise ValueError(f"No samples with label={label}. Available labels: {unique}")

    # choose one sample
    if idx is None:
        random.seed(seed)
        chosen_global = int(random.choice(indices))
    else:
        idx = int(idx)
        if idx < 0 or idx >= len(indices):
            raise IndexError(f"idx out of range. There are {len(indices)} samples with label={label}")
        chosen_global = int(indices[idx])

    sample = X[chosen_global]

    # move to cpu for plotting
    sample_cpu = sample.detach().cpu()

    # --- Plot depending on shape ---
    plt.figure()
    if sample_cpu.ndim == 1:
        plt.plot(sample_cpu.numpy())
        plt.xlabel("Time / index")
        plt.ylabel("Amplitude")
    elif sample_cpu.ndim == 2:
        h, w = sample_cpu.shape

        # Treat (T,1) or (1,T) as 1D signal
        if h == 1 or w == 1:
            signal = sample_cpu.flatten().numpy()
            plt.plot(signal)
            plt.xlabel("Time / index")
            plt.ylabel("Amplitude")

        # Small number of channels → multichannel signal
        elif h <= 8:
            for c in range(h):
                plt.plot(sample_cpu[c].numpy(), label=f"ch{c}")
            plt.legend()
            plt.xlabel("Time / index")
            plt.ylabel("Amplitude")

        # Otherwise → image-like (spectrogram / feature map)
        else:
            plt.imshow(sample_cpu.numpy(), aspect="auto", origin="lower")
            plt.colorbar()
            plt.xlabel("Time")
            plt.ylabel("Freq / feature")
    elif sample_cpu.ndim == 3:
        # often (C, H, W) for spectrogram-like
        # show first channel by default
        img = sample_cpu[0].numpy()
        plt.imshow(img, aspect="auto", origin="lower")
        plt.colorbar()
    else:
        raise ValueError(f"Don't know how to plot sample with shape {tuple(sample_cpu.shape)}")

    ttl = title or f"Label={label} | global_idx={chosen_global} | shape={tuple(sample_cpu.shape)}"
    plt.title(ttl)
    plt.tight_layout()
    plt.show()

    return chosen_global



if __name__ == "__main__":
    ROOT_CWRU = r"raw_data\CWRU"
    ROOT_PU = r"raw_data\PU"
    ROOT_XJTU = r"raw_data\XJTU"
    ROOT_SEU = r"raw_data\SEU\gearbox"
    ROOT_JNU = r"raw_data\JNU\JNU-Bearing-Dataset-main"

    ds = PU(data_dir=ROOT_PU, normlizetype="minus_one_one", rand=42, augmentype_1 = "randomcrop", augmentype_2 = "gaussian")
    # Augmentetion methods:
    # randomstrech(does not work), gaussian, normal, scale, randomcrop, fft
    train_ds, val_ds, test_ds = ds.data_prepare()

    visualize_sample(train_ds, label=0)
    visualize_sample(train_ds, label=1)
    #visualize_sample(train_ds, label=2)
    #visualize_sample(train_ds, label=3)
