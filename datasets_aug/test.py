# test_datasets.py
import numpy as np
import pandas as pd
import torch
import pytest

# --- Import your classes here ---
from sequence_dataset import OneViewDataset, TwoViewDataset, MultiViewDataset

# --- Minimal stubs so tests don't depend on datasets_aug.sequence_aug ---
class Reshape:
    def __call__(self, x):
        # Ensure a predictable 2D output for assertions
        arr = np.asarray(x)
        return arr.reshape(1, -1)

class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


@pytest.fixture
def data_list():
    # "data_list" behaves like your code expects: data_list['data'].tolist()
    # Each sample is a 1D array of length 3.
    return pd.DataFrame(
        {
            "data": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            "label": [7, 8],
        }
    )


def test_one_view_train_len_and_types(data_list):
    ds = OneViewDataset(data_list, test=False, transform=Compose([Reshape()]))

    assert len(ds) == 2

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert x.dtype == torch.float32
    assert y.dtype == torch.long

    # Our stub Reshape makes it (1, 3)
    assert x.shape == (1, 3)
    assert y.item() == 7


def test_one_view_test_label_is_index(data_list):
    ds = OneViewDataset(data_list, test=True, transform=Compose([Reshape()]))

    x, y = ds[1]
    assert x.shape == (1, 3)
    assert y.dtype == torch.long
    assert y.item() == 1  # label == idx in test mode


def test_two_view_returns_two_views_and_label(data_list):
    t1 = Compose([Reshape()])
    # a different transform to ensure the two views can differ
    class Scale2:
        def __call__(self, x):
            return np.asarray(x) * 2

    t2 = Compose([Scale2(), Reshape()])

    ds = TwoViewDataset(data_list, test=False, transform_1=t1, transform_2=t2)

    x1, x2, y = ds[0]

    assert x1.dtype == torch.float32
    assert x2.dtype == torch.float32
    assert y.dtype == torch.long

    assert x1.shape == (1, 3)
    assert x2.shape == (1, 3)

    # x2 should be scaled by 2 compared to x1
    assert torch.allclose(x2, x1 * 2)
    assert y.item() == 7


def test_two_view_test_label_is_index(data_list):
    ds = TwoViewDataset(data_list, test=True, transform_1=Compose([Reshape()]), transform_2=Compose([Reshape()]))

    x1, x2, y = ds[0]
    assert y.item() == 0


def test_multi_view_n_views_and_label(data_list):
    t1 = Compose([Reshape()])
    class Add1:
        def __call__(self, x):
            return np.asarray(x) + 1

    t2 = Compose([Add1(), Reshape()])

    ds = MultiViewDataset(data_list, transforms=[t1, t2], test=False, return_index=False)

    v1, v2, y = ds[0]
    assert v1.shape == (1, 3)
    assert v2.shape == (1, 3)
    assert y.item() == 7

    # v2 = v1 + 1 (elementwise), given Add1
    assert torch.allclose(v2, v1 + 1)


def test_multi_view_return_index(data_list):
    t1 = Compose([Reshape()])
    ds = MultiViewDataset(data_list, transforms=[t1], test=True, return_index=True)

    v1, y, idx = ds[1]
    assert v1.shape == (1, 3)
    assert y == 1
    assert idx == 1
