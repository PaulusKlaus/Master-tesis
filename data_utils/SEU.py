import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import islice

from datasets_aug.sequence_dataset import *  
from datasets_aug.sequence_aug import *
from data_utils.data_utils import *



#Data names of 5 bearing fault types under two working conditions
Bdata = ["ball_20_0.csv","comb_20_0.csv","health_20_0.csv","inner_20_0.csv","outer_20_0.csv","ball_30_2.csv","comb_30_2.csv","health_30_2.csv","inner_30_2.csv","outer_30_2.csv"]
label1 = [i for i in range(0,10)]
#Data names of 5 gear fault types under two working conditions
Gdata = ["Chipped_20_0.csv","Health_20_0.csv","Miss_20_0.csv","Root_20_0.csv","Surface_20_0.csv","Chipped_30_2.csv","Health_30_2.csv","Miss_30_2.csv","Root_30_2.csv","Surface_30_2.csv"]
labe12 = [i for i in range(10,20)]



class SEU(object):

    def __init__(self, data_dir, rand, normlizetype, augmentype_1 = "normal", augmentype_2 = "fft"):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.random_state = rand
        self.augmentation_1 = augmentype_1
        self.augmentation_2 = augmentype_2 

    def _get_files(self):

        root = self.data_dir

        datasetname = os.listdir(os.path.join(root, os.listdir(root)[0]))  # 0:bearingset, 2:gearset
        root1 = os.path.join(root, os.listdir(root)[0], datasetname[0]) #Path of bearingset
        root2 = os.path.join(root, os.listdir(root)[1], datasetname[1]) #Path of gearset

        data, lab = [], []
        
        for i in tqdm(range(len(Bdata))):
            path1 = os.path.join(root1,Bdata[i])
            data1, lab1 = self._data_load(path1, dataname=Bdata[i], label=label1[i])
            data += data1
            lab += lab1

        for j in tqdm(range(len(Gdata))):
            path2 = os.path.join(root2, Gdata[j])
            data2, lab2 = self._data_load(path2, dataname=Gdata[j], label=labe12[j])
            data += data2
            lab += lab2

        return [data, lab]

    def _data_load(self, filename, dataname, label, signal_size = 1024):

        f = open(filename, "r", encoding='gb18030', errors='ignore')
        fl = []
        if dataname == "ball_20_0.csv":
            for line in islice(f, 16, None):  #Skip the first 16 lines
                line = line.rstrip()
                word = line.split(",", 8)   #Separated by commas
                fl.append(eval(word[1]))   # Take a vibration signal in the x direction as input
        else:
            for line in islice(f, 16, None):  #Skip the first 16 lines
                line = line.rstrip()
                word = line.split("\t", 8)   #Separated by \t
                fl.append(eval(word[1]))   # Take a vibration signal in the x direction as input
        fl = np.array(fl)
        fl = fl.reshape(-1, 1)

        data, lab = [], []
        start, end=0, signal_size

        while end <= fl.shape[0] / 10:
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

