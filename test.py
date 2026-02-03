# Global imports
import os
from collections import Counter

# Local imports
from data_utils import *
from datasets_aug import *


# ---------- Shared helpers ----------

def count_dataset_labels(ds):
    return Counter(int(ds[i][1]) for i in range(len(ds)))


def run_common_dataset_tests(ds, dataset_name="Dataset"):
    # 1) raw loading
    data, labels = ds._get_files()

    print(f"\n[{dataset_name}] Loaded samples:", len(data))
    print(f"[{dataset_name}] Loaded labels :", len(labels))

    assert len(data) == len(labels), "data/labels length mismatch"
    assert len(data) > 0, f"no samples loaded from {dataset_name}"

    print("\nRaw label counts:")
    print(Counter(labels))

    # 2) check one sample
    print("All Labels:", sorted(set(labels)))
    x0, y0 = data[0], labels[0]

    print(
        "First sample type:", type(x0),
        "shape:", getattr(x0, "shape", None),
        "label:", y0
    )

    assert x0.shape[0] == 1024, f"expected 1024 rows, got {x0.shape}"
    print(f"✅ Basic _get_files() test passed for {dataset_name}")

    # 3) public API
    train_ds, val_ds, test_ds = ds.data_prepare()

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    print("✅ train/val/test dataset objects created")
    print("Lengths:", len(train_ds), len(val_ds), len(test_ds))

    train_data, train_label = train_ds[-1]
    print("Example train sample:", train_data)
    print("Example train label :", train_label)

    print("\nTrain label counts:", count_dataset_labels(train_ds))
    print("Val label counts  :", count_dataset_labels(val_ds))
    print("Test label counts :", count_dataset_labels(test_ds))


# ---------- Dataset-specific tests ----------

def test_CWRU_dataset():
    ROOT = r"raw_data\CWRU"

    normal_path = os.path.join(
        ROOT, "Normal Baseline Data", "0_N_0.mat"
    )
    fault_path = os.path.join(
        ROOT, "12k Drive End Bearing Fault Data", "1_IR007_0.mat"
    )

    assert os.path.exists(normal_path), f"Missing: {normal_path}"
    assert os.path.exists(fault_path), f"Missing: {fault_path}"
    print("✅ CWRU files found")

    ds = CWRU(data_dir=ROOT, normlizetype="minus_one_one", rand=42)
    run_common_dataset_tests(ds, dataset_name="CWRU")


def test_PU_dataset():
    ROOT = r"raw_data\PU"

    first_bearing = RDBdata[0]
    name3 = f"{state}_{first_bearing}_1"
    sample_path = os.path.join(ROOT, first_bearing, name3 + ".mat")

    assert os.path.exists(sample_path), f"Missing example file: {sample_path}"
    print("✅ PU example file found:", sample_path)

    ds = PU(data_dir=ROOT, normlizetype="minus_one_one", rand=42)
    run_common_dataset_tests(ds, dataset_name="PU")


def test_XJTU_dataset():
    ROOT = r"raw_data\XJTU"

    ds = XJTU(data_dir=ROOT, normlizetype="minus_one_one", rand=42)
    run_common_dataset_tests(ds, dataset_name="XJTU")


def test_SEU_dataset():
    ROOT = r"raw_data\SEU\gearbox"

    # Quick sanity check: make sure at least one expected CSV exists somewhere under ROOT
    found_sample = None
    if os.path.exists(ROOT):
        for dirpath, dirnames, filenames in os.walk(ROOT):
            for fname in Bdata + Gdata:
                if fname in filenames:
                    found_sample = os.path.join(dirpath, fname)
                    break
            if found_sample:
                break

    assert found_sample is not None, f"No SEU sample files found under {ROOT}. Checked names: {Bdata + Gdata}"
    print("✅ Found SEU sample:", found_sample)
    ds = SEU(data_dir=ROOT, rand=42, normlizetype="minus_one_one")

    run_common_dataset_tests(ds, dataset_name="SEU")

def test_JNU_dataset():
    pass 

# ---------- Run tests ----------
if __name__ == "__main__":
    #test_CWRU_dataset()
    #test_PU_dataset()
    #test_XJTU_dataset()
    test_SEU_dataset()