# Global imports 
import os


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
    ds = CWRU(data_dir=ROOT, normlizetype="-1-1", rand=42)

    # 3) test "raw loading" via the class 
    data, labels = ds._get_files()

    print("\nLoaded samples:", len(data))
    print("Loaded labels :", len(labels))
    assert len(data) == len(labels), "data/labels length mismatch"
    assert len(data) > 0, "no samples loaded"

    # 4) check one sample shape
    x0 = data[0]
    y0 = labels[0]
    print("First sample type:", type(x0), "shape:", getattr(x0, "shape", None), "label:", y0)

    # sanity checks: each window should have 1024 points (or (1024,1) depending on mat format)
    assert x0.shape[0] == 1024, f"expected 1024 rows, got {x0.shape}"
    print("✅ Basic get_files() test passed")

        # 5) test the public API: train/val datasets
    train_ds, val_ds = ds.data_preprare(test=False)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    print("✅ train/val dataset objects created")

    # 6) test the test=True path
    test_ds = ds.data_preprare(test=True)
    assert len(test_ds) > 0
    print("✅ test dataset object created")





# -----Testing ------
if __name__ == "__main__":
    test_CWRU_dataset()
