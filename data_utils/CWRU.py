import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from datasets_aug.sequence_dataset import *  #dataset 
from datasets_aug.sequence_aug import *


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



def train_test_split_order(data_pd, test_size=0.3, labels=None):
    """
    Ordered split within each class:
    - first (1 - val_size) fraction -> train
    - last val_size fraction -> val
    Preserves the existing order in data_pd for each class.
    """
    if labels is None:
        labels = sorted(data_pd["label"].unique())

    train_parts = []
    val_parts = []

    for lbl in labels:
        cls = data_pd[data_pd["label"] == lbl].reset_index(drop=True)
        n = len(cls)
        if n == 0:
            continue

        cut = int((1 - test_size) * n)  # number of train samples
        train_parts.append(cls.iloc[:cut][["data", "label"]])
        val_parts.append(cls.iloc[cut:][["data", "label"]])

    train_pd = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=["data","label"])
    test_pd   = pd.concat(val_parts,   ignore_index=True) if val_parts else pd.DataFrame(columns=["data","label"])
    return train_pd, test_pd


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

    def data_preprare(self, split = "RA", view = OneViewDataset ):
        """
        Data preparation fuction that returns:
        Train, validation and test OneViewDataset classes 
        view can ve either OneViewDataset or TwoViewDataset
        """
        list_data = self._get_files()
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        if split == "RA":
            "Random split with augmentation"
            train_pd, temp_pd = train_test_split(
                data_pd,
                test_size=0.30,  # 70% train
                random_state=self.random_state,
                stratify=data_pd["label"]
            )
            val_pd, test_pd = train_test_split(
                temp_pd,
                test_size=0.5,   # 15% test, 15% val
                random_state=self.random_state,
                stratify=temp_pd["label"]
            )
            # Returning tensors for data and labels 
            train_dataset = view(train_pd,
                                 transform_1=data_transforms(self.augmentation_1, self.normlizetype),
                                 transform_2=data_transforms(self.augmentation_2, self.normlizetype))
            val_dataset   = view(val_pd, 
                                 transform_1=data_transforms('normal', self.normlizetype),
                                 transform_2=data_transforms('normal', self.normlizetype))
            test_dataset  = view(test_pd, 
                                 transform_1=data_transforms('normal', self.normlizetype),
                                 transform_2=data_transforms('normal', self.normlizetype))
        
        elif split == "R_NA":
            "Random split with no augmentation"
            train_pd, temp_pd = train_test_split(
                data_pd,
                test_size=0.30,  # 20% val + 10% test
                random_state=self.random_state,
                stratify=data_pd["label"]
            )
            val_pd, test_pd = train_test_split(
                temp_pd,
                test_size=0.5,  
                random_state=self.random_state,
                stratify=temp_pd["label"]
            )

            train_dataset = view(train_pd,
                                 transform_1=data_transforms('normal', self.normlizetype),
                                 transform_2=data_transforms('normal', self.normlizetype))
            val_dataset   = view(val_pd,
                                 transform_1=data_transforms('normal',   self.normlizetype),
                                 transform_2=data_transforms('normal', self.normlizetype))
            test_dataset  = view(test_pd,
                                 transform_1=data_transforms('normal',   self.normlizetype),
                                 transform_2=data_transforms('normal', self.normlizetype))

        elif split == "O_A":
            "Order split with augmentation"
            train_pd, temp_pd = train_test_split_order(
                data_pd,
                test_size=0.30, 
            )

            val_pd, test_pd = train_test_split_order(
                temp_pd,
                test_size=0.5, 
            )
            train_dataset = view(train_pd, 
                                 transform_1=data_transforms(self.augmentation_1, self.normlizetype),
                                 transform_2=data_transforms(self.augmentation_2, self.normlizetype))
            val_dataset   = view(val_pd, 
                                 transform_1=data_transforms('normal', self.normlizetype),
                                 transform_2=data_transforms('normal', self.normlizetype))
            test_dataset  = view(test_pd, 
                                 transform_1=data_transforms('normal', self.normlizetype),
                                 transform_2=data_transforms('normal', self.normlizetype))

        return train_dataset, val_dataset, test_dataset
        

