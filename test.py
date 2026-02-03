# Global imports 
import os
from collections import Counter


# Local imports 
from data_utils import *
from datasets_aug import *


# ---- Testing of the separate components of the project ----
def test_CWRU_dataset():
    """
    Test Function for the Preprocessing of the CWRU
    """
    ROOT = "raw_data\\CWRU"  
    # 1) check files exist
    normal_path = os.path.join(ROOT, "Normal Baseline Data", "0_N_0.mat")
    fault_path  = os.path.join(ROOT, "12k Drive End Bearing Fault Data", "1_IR007_0.mat")

    assert os.path.exists(normal_path), f"Missing: {normal_path}"
    assert os.path.exists(fault_path), f"Missing: {fault_path}"
    print("✅ Files found")

    # 2) create dataset object
    ds = CWRU(data_dir=ROOT, normlizetype="minus_one_one", rand=42)

    # 3) test "raw loading" via the class 
    data, labels = ds._get_files()

    print("\nLoaded samples:", len(data))
    print("Loaded labels :", len(labels))
    assert len(data) == len(labels), "data/labels length mismatch"
    assert len(data) > 0, "no samples loaded"
    print("\nRaw label counts:")
    print(Counter(labels))

    # 4) check one sample shape
    print("All Labels:",  sorted(set(labels)))
    x0 = data[0]
    y0 = labels[0]
    print("First sample type:", type(x0), "shape:", getattr(x0, "shape", None), "label:", y0)

    # sanity checks: each window should have 1024 points (or (1024,1) depending on mat format)
    assert x0.shape[0] == 1024, f"expected 1024 rows, got {x0.shape}"
    print("✅ Basic get_files() test passed")

    # 5) test the public API: train/val datasets
    train_ds, val_ds, test_ds = ds.data_prepare()
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    print("✅ train/val and test dataset objects created")
    print("Length of train, val, test", len(train_ds), len(val_ds), len(test_ds))
    train_data, train_l = train_ds[-1]
    print(train_data)
    print(train_l)
    
    def count_dataset_labels(ds):
        labels = [int(ds[i][1]) for i in range(len(ds))]
        return Counter(labels)

    print("\nTrain label counts:", count_dataset_labels(train_ds))
    print("Val label counts  :", count_dataset_labels(val_ds))
    print("Test label counts :", count_dataset_labels(test_ds))

def test_PU_dataset():
    """
    Test Function for the Preprocessing of the PU Dataset
    """
    # 1) ROOT folder for your PU data
    #    Adjust this to match your actual folder structure
    ROOT = r"raw_data\PU"

    # 1) check that at least one known file exists
    first_bearing = RDBdata[0]
    name3 = f"{state}_{first_bearing}_1"
    sample_path = os.path.join(ROOT, first_bearing, name3 + ".mat")

    assert os.path.exists(sample_path), f"Missing example .mat file: {sample_path}"
    print("✅ Example PU .mat file found:", sample_path)

    # 2) create dataset object
    ds = PU(data_dir=ROOT, normlizetype="minus_one_one", rand=42)

    # 3) test "raw loading" via the class
    data, labels = ds._get_files()

    print("\nLoaded samples:", len(data))
    print("Loaded labels :", len(labels))
    assert len(data) == len(labels), "data/labels length mismatch"
    assert len(data) > 0, "no samples loaded from PU dataset"
    print("\nRaw label counts:")
    print(Counter(labels))

    # 4) check one sample shape
    print("All Labels in PU dataset:", sorted(set(labels)))
    x0 = data[0]
    y0 = labels[0]
    print("First sample type:", type(x0),
          "shape:", getattr(x0, "shape", None),
          "label:", y0)

    # sanity checks: each window should have 1024 points
    # (or (1024, 1) depending on how you keep fl)
    assert x0.shape[0] == 1024, f"expected 1024 rows, got {x0.shape}"
    print("✅ Basic _get_files() test passed for PU")

    # 5) test the public API: train/val datasets
    train_ds, val_ds, test_ds = ds.data_prepare()
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    print("✅ train/val and test dataset objects created")
    print("Length of train, val, test", len(train_ds), len(val_ds), len(test_ds))
    train_data, train_l = train_ds[-1]
    print(train_data)
    print(train_l)
    
    def count_dataset_labels(ds):
        labels = [int(ds[i][1]) for i in range(len(ds))]
        return Counter(labels)

    print("\nTrain label counts:", count_dataset_labels(train_ds))
    print("Val label counts  :", count_dataset_labels(val_ds))
    print("Test label counts :", count_dataset_labels(test_ds))



def test_XJTU_dataset():
    """
    Test Function for the Preprocessing of the PU Dataset
    """
    # 1) ROOT folder for your PU data
    #    Adjust this to match your actual folder structure
    ROOT = r"raw_data\XJTU"

    # 2) create dataset object
    ds = XJTU(data_dir=ROOT, normlizetype="minus_one_one", rand=42)

    # 3) test "raw loading" via the class
    data, labels = ds._get_files()

    print("\nLoaded samples:", len(data))
    print("Loaded labels :", len(labels))
    assert len(data) == len(labels), "data/labels length mismatch"
    assert len(data) > 0, "no samples loaded from PU dataset"
    print("\nRaw label counts:")
    print(Counter(labels))

    # 4) check one sample shape
    print("All Labels in PU dataset:", sorted(set(labels)))
    x0 = data[0]
    y0 = labels[0]
    print("First sample type:", type(x0),
          "shape:", getattr(x0, "shape", None),
          "label:", y0)

    # sanity checks: each window should have 1024 points
    # (or (1024, 1) depending on how you keep fl)
    assert x0.shape[0] == 1024, f"expected 1024 rows, got {x0.shape}"
    print("✅ Basic _get_files() test passed for PU")

    # 5) test the public API: train/val datasets
    train_ds, val_ds, test_ds = ds.data_prepare()
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    print("✅ train/val and test dataset objects created")
    print("Length of train, val, test", len(train_ds), len(val_ds), len(test_ds))
    train_data, train_l = train_ds[-1]
    print(train_data)
    print(train_l)
    
    def count_dataset_labels(ds):
        labels = [int(ds[i][1]) for i in range(len(ds))]
        return Counter(labels)

    print("\nTrain label counts:", count_dataset_labels(train_ds))
    print("Val label counts  :", count_dataset_labels(val_ds))
    print("Test label counts :", count_dataset_labels(test_ds))



# -----Testing ------
if __name__ == "__main__":
    #test_CWRU_dataset()
    #test_PU_dataset()
    test_XJTU_dataset()