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


    # 3) call your get_files and validate shapes/counts
    data, labels = get_files(ROOT)

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





# -----Testing ------
if __name__ == "__main__":
    test_CWRU_dataset()
