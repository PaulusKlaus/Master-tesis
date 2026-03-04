import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from datasets_aug.sequence_dataset import *  #dataset 
from datasets_aug.sequence_aug import *
from .data_utils import *

datasetname =["Normal Baseline Data", "12k Drive End Bearing Fault Data"]
normalname = ["0_N_0.mat", "0_N_1.mat", "0_N_2.mat", "0_N_3.mat"]

ir_faults = ['1_IR007_0.mat', '1_IR007_1.mat', '1_IR007_2.mat', '1_IR007_3.mat']
b_faults = ['2_B007_0.mat', '2_B007_1.mat', '2_B007_2.mat', '2_B007_3.mat']
or_faults = ['3_OR007_6_0.mat', '3_OR007_6_1.mat', '3_OR007_6_2.mat', '3_OR007_6_3.mat']

load_0 = ['1_IR007_0.mat', '2_B007_0.mat', '3_OR007_6_0.mat', ]
load_1 = ['1_IR007_1.mat', '2_B007_1.mat', '3_OR007_6_1.mat', ]
load_2 = ['1_IR007_2.mat', '2_B007_2.mat', '3_OR007_6_2.mat', ]
load_3 = ['1_IR007_3.mat', '2_B007_3.mat', '3_OR007_6_3.mat', ]

load_ALL = ir_faults + b_faults + or_faults
label_ALL = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
label = [1, 2, 3]
axis = ["_DE_time", "_FE_time", "_BA_time"]

import pandas as pd

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



def find_de_channel_key(mat_dict: dict):
    """
    Try to find the Drive End channel key. Robust to keys like:
    'X012_DE_time', 'DE_time', 'DE', 'DE_time_1', etc.
    Returns the key or None if not found.
    """
    keys = [k for k in mat_dict.keys() if not k.startswith('__')]
    # Prefer exact-ish DE matches first
    for k in keys:
        low = k.lower()
        if 'de' in low and ('time' in low or low.endswith('de') or '_de_' in low):
            return k
    # Fall back to anything containing 'de'
    for k in keys:
        if 'de' in k.lower():
            return k
    return None



class CWRU(object):
    def __init__(self, data_dir, rand, normlizetype, augmentype_1 = "normal", augmentype_2 = "fft"):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.random_state = rand
        self.augmentation_1 = augmentype_1
        self.augmentation_2 = augmentype_2 

    def _get_files(self):
        '''
        This function is used to generate the final training set and test set.
        root:The location of the data set
        normalname:List of normal data
        dataname:List of failure data
        '''
        data_root1 = os.path.join(self.data_dir, datasetname[0]) # Normal data 
        data_root2 = os.path.join(self.data_dir, datasetname[1]) # Drive End Bearing Fault 

        data, lab = [], []

        # load ALL normal files
        for nf in normalname:
            path1 = os.path.join(data_root1, nf)
            data_n, lab_n = self._data_load(path1, channel_suffix=axis[0], label=0)
            data += data_n
            lab  += lab_n


        for i in tqdm(range(len(load_ALL))):
            load = load_ALL[i]
            label_i = label_ALL[i]
            path2 = os.path.join(data_root2, load)

            data1, lab1 = self._data_load(path2, channel_suffix=axis[0], label=label_i)
            data += data1
            lab += lab1
        return [data, lab]
    
    def _data_load(self, filename, channel_suffix="_DE_time", label=0, signal_size=1024):
        """
        Load a CWRU .mat and extract the signal array whose key ends with channel_suffix.
        Then slice into consecutive windows of length signal_size.
        """
        mat = loadmat(filename)

        # Find the actual variable name(s) that contain the channel you want
        candidates = [k for k in mat.keys()
                    if k.startswith("X") and k.endswith(channel_suffix)]

        if not candidates:
            x_keys = [k for k in mat.keys() if k.startswith("X")]
            raise KeyError(
                f"No key ending with '{channel_suffix}' found in {filename}.\n"
                f"Available X* keys: {x_keys}"
            )

        key = candidates[0]  # usually exactly one match
        fl = mat[key]

        # Make it 1D: CWRU often stores as (N,1)
        #fl = np.asarray(fl).squeeze()
        fl = fl.reshape(-1,1)
        data, lab = [], []
        for start in range(0, len(fl) - signal_size + 1, signal_size):
            end = start + signal_size
            data.append(fl[start:end])
            lab.append(label)

        return data, lab

    def data_prepare(self, split="RA", view=TwoViewDataset):
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
                test_size=0.50,
                random_state=self.random_state,
                stratify=data_pd["label"],
            )
            val_temp, test_pd = train_test_split(
                temp_pd,
                test_size=0.4,
                random_state=self.random_state,
                stratify=temp_pd["label"],
            )
            val_pd, classifier_pd = train_test_split(
                val_temp,
                test_size=0.3,
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

        # --- build datasets ---
        # --- optional per-class caps ---
        train_pd = cap_per_class(train_pd, n_per_class=100, seed=self.random_state)
        test_pd = cap_per_class(test_pd, n_per_class=100, seed=self.random_state)
        val_pd = cap_per_class(val_pd, n_per_class=100, seed=self.random_state)
        classifier_pd = cap_per_class(classifier_pd, n_per_class=10, seed=self.random_state)

        train_dataset = view(train_pd, transform_1=train_t1, transform_2=train_t2)
        val_dataset   = view(val_pd,   transform_1=eval_t1,  transform_2=eval_t2)
        test_dataset  = view(test_pd,  transform_1=eval_t1,  transform_2=eval_t2)
        classifier_dataset = view(classifier_pd,  transform_1=eval_t1,  transform_2=eval_t2)

        return train_dataset, val_dataset, test_dataset, classifier_dataset
