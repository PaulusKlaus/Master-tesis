import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets_aug.sequence_dataset import *  
from datasets_aug.sequence_aug import *
from data_utils.data_utils import *


WC1 = ["ib600_2.csv", "n600_3_2.csv", "ob600_2.csv", "tb600_2.csv"]
WC2 = ["ib800_2.csv", "n800_3_2.csv", "ob800_2.csv", "tb800_2.csv"]
WC3 = ["ib1000_2.csv", "n1000_3_2.csv", "ob1000_2.csv", "tb1000_2.csv"]

label1 = [i for i in range(0, 4)]
label2 = [i for i in range(4, 8)]
label3 = [i for i in range(8, 12)]


class JNU(object):
    num_classes = 12
    inputchannel = 1

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

        for i in tqdm(range(len(WC1))):
            path1 = os.path.join(root, WC1[i])
            data1, lab1 = self._data_load(path1, label=label1[i])
            data += data1
            lab +=lab1

        for j in tqdm(range(len(WC2))):
            path2 = os.path.join(root, WC2[j])
            data2, lab2 = self._data_load(path2, label=label2[j])
            data += data2
            lab += lab2

        for k in tqdm(range(len(WC3))):
            path3 = os.path.join(root, WC3[k])
            data3, lab3 = self._data_load(path3, label=label3[k])
            data += data3
            lab += lab3

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
            val_pd, test_pd = train_test_split(
                temp_pd,
                test_size=0.5,
                random_state=self.random_state,
                stratify=temp_pd["label"],
            )
        elif split == "O_A":
            # ordered split (your custom)
            train_pd, temp_pd = train_test_split_order(data_pd, test_size=0.30)
            val_pd, test_pd   = train_test_split_order(temp_pd, test_size=0.5)
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
        train_dataset = view(train_pd, transform_1=train_t1, transform_2=train_t2)
        val_dataset   = view(val_pd,   transform_1=eval_t1,  transform_2=eval_t2)
        test_dataset  = view(test_pd,  transform_1=eval_t1,  transform_2=eval_t2)

        return train_dataset, val_dataset, test_dataset



