import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from datasets_aug.sequence_dataset import *  
from datasets_aug.sequence_aug import *
from .data_utils import *

# 1 Undamaged (healthy) bearings(6X)
HBdata = ['K001',"K002",'K003','K004','K005','K006']
label1=[0,1,2,3,4,5]  #The undamaged (healthy) bearings data is labeled 0-5

# 2 Artificially damaged bearings(12X)
ADBdata = ['KA01','KA03', 'KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']
label2=[6,7,8,9,10,11,12,13,14,15,16,17]    # The artificially damaged bearings data is labeled 6-17

# 3 Bearings with real damages caused by accelerated lifetime tests(14x)
RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI04','KI14','KI16','KI17','KI18','KI21']
label3=[i for i in range(18,18+len(RDBdata))]

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
state = WC[0] #WC[0] can be changed to different working states

ALL_DATA  = HBdata + ADBdata + RDBdata
ALL_LABEL = label1 + label2 + label3



class PU(object):
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
        '''
        data = []
        lab = []

        for k in tqdm(range(len(ALL_DATA))):
            bearing = ALL_DATA[k]
            label = ALL_LABEL[k]

            #NOTE: This only uses 1 of 20 samples, which ends with _1.mat 
            name = state + "_" + bearing + "_1"
            path = os.path.join(self.data_dir, bearing, name + ".mat")        
            d, l = self._data_load(path, name=name, label=label)
            data += d
            lab += l

        return [data,lab]

    def _data_load(self, filename, name, label, signal_size = 1024):
        '''
        This function is mainly used to generate test data and training data.
        filename:Data location
        '''
        data = [] 
        lab = []
        start, end = 0, signal_size

        fl = loadmat(filename)[name]
        fl = fl[0][0][2][0][6][2]  #Take out the data
        fl = fl.reshape(-1,1)

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
