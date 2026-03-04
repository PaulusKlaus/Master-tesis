import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import repeat

from datasets_aug.sequence_dataset import *  
from datasets_aug.sequence_aug import *
from .data_utils import *


WC1 = ["ib600_2.csv", "n600_3_2.csv", "ob600_2.csv", "tb600_2.csv"]
WC2 = ["ib800_2.csv", "n800_3_2.csv", "ob800_2.csv", "tb800_2.csv"]
WC3 = ["ib1000_2.csv", "n1000_3_2.csv", "ob1000_2.csv", "tb1000_2.csv"]

label1 = [i for i in range(0, 4)]
label2 = [i for i in range(4, 8)]
label3 = [i for i in range(8, 12)]


HBdata = ["n600_3_2.csv", "n800_3_2.csv","n1000_3_2.csv"]
or_faults  = ["ob600_2.csv","ob800_2.csv", "ob1000_2.csv"]
b_fault = [ "tb600_2.csv", "tb800_2.csv", "tb1000_2.csv"]
ir_faults  = ["ib600_2.csv","ib800_2.csv","ib1000_2.csv",]

samples = (
    list(zip(HBdata,     repeat("healthy"))) +
    list(zip(ir_faults,  repeat("inner_race"))) +
    list(zip(or_faults,  repeat("outer_race"))) +
    list(zip(b_fault,  repeat("ball"))) 
)

# stable mapping
class_to_idx = {"healthy": 0, "inner_race": 1, "outer_race": 2, "ball" : 3}

ALL_DATA  = [sid for sid, _ in samples]
ALL_LABEL = [class_to_idx[c] for _, c in samples]

def cap_per_class(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """
    Keep at most n_per_class samples per label (balanced cap).
    If a class has fewer than n_per_class, it keeps all available samples.
    """
    if n_per_class is None:
        return df

    # sample within each label group deterministically
    return (df.groupby("label", group_keys=False)
              .sample(n=n_per_class, random_state=seed, replace=False)
              .reset_index(drop=True))



class JNU(object):

    def __init__(self, data_dir, rand, normlizetype, augmentype_1 = "normal", augmentype_2 = "fft"):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.random_state = rand
        self.augmentation_1 = augmentype_1
        self.augmentation_2 = augmentype_2 

    def _get_files(self):
        root = self.data_dir

        data = []
        lab =[]

        for i in tqdm(range(len(ALL_DATA))):
            bearing = ALL_DATA[i]
            label = ALL_LABEL[i]

            path1 = os.path.join(root, bearing)
            data1, lab1 = self._data_load(path1, label=label)
            data += data1
            lab +=lab1

        return [data, lab]
    
    def _data_load(self, filename, label, signal_size=1024):

        fl = np.loadtxt(filename)
        fl = fl.reshape(-1, 1)

        data, lab = [], []
        start, end = 0, signal_size
        while end <= fl.shape[0]:
            data.append(fl[start:end])
            lab.append(label)
            start += signal_size
            end += signal_size

        return data, lab

    def data_prepare(self, split="RA", view=OneViewDataset):
        """
        Returns: train_dataset, val_dataset, test_dataset
        split:
        - "RA"   : random split, augmentation on train
        - "R_NA" : random split, no augmentation
        - "O_A"  : ordered split, augmentation on train
        view: OneViewDataset or TwoViewDataset
        """
        data, labels = self._get_files()
        data_pd = pd.DataFrame({"data": data, "label": labels})

        # --- choose split function + splitting params ---
        if split in ("RA", "R_NA"):
            # stratified random split
            train_pd, temp_pd = train_test_split(
                data_pd,
                test_size=0.30,
                random_state=self.random_state,
                stratify=data_pd["label"],
            )
            val_temp, test_pd = train_test_split(
                temp_pd,
                test_size=0.5,
                random_state=self.random_state,
                stratify=temp_pd["label"],
            )
            val_pd, classifier_pd = train_test_split(
                val_temp,
                test_size=0.5,
                random_state=self.random_state,
                stratify=val_temp["label"],
            )

        elif split == "O_A":
            # ordered split (your custom)
            train_pd, temp_pd = train_test_split_order(data_pd, test_size=0.30)
            val_temp, test_pd   = train_test_split_order(temp_pd, test_size=0.5)
            val_pd, classifier_pd   = train_test_split_order(val_temp, test_size=0.5)
        else:
            raise ValueError(f"Unknown split='{split}'. Use 'RA', 'R_NA', or 'O_A'.")

        # --- choose transforms ---
        if split in ("RA", "O_A"):  # augmentation on train only
            train_t1 = data_transforms(self.augmentation_1, self.normlizetype)
            train_t2 = data_transforms(self.augmentation_2, self.normlizetype)
        else:  # no augmentation
            train_t1 = data_transforms("normal", self.normlizetype)
            train_t2 = data_transforms("normal", self.normlizetype)

        eval_t1 = data_transforms("normal", self.normlizetype)
        eval_t2 = data_transforms("normal", self.normlizetype)
        
        train_pd = cap_per_class(train_pd, n_per_class=100, seed=self.random_state)
        test_pd = cap_per_class(test_pd, n_per_class=100, seed=self.random_state)
        val_pd = cap_per_class(val_pd, n_per_class=100, seed=self.random_state)
        classifier_pd = cap_per_class(classifier_pd, n_per_class=10, seed=self.random_state)

        # --- build datasets ---
        train_dataset = view(train_pd, transform_1=train_t1, transform_2=train_t2)
        val_dataset   = view(val_pd,   transform_1=eval_t1,  transform_2=eval_t2)
        test_dataset  = view(test_pd,  transform_1=eval_t1,  transform_2=eval_t2)
        classifier_dataset = view(classifier_pd,  transform_1=eval_t1,  transform_2=eval_t2)

        return train_dataset, val_dataset, test_dataset, classifier_dataset



