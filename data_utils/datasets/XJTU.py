import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from datasets_aug.sequence_dataset import *  #dataset 
from datasets_aug.sequence_aug import *
from .data_utils import *


label1 = [i for i in range(0,5)]
label2 = [i for i in range(5,10)]
label3 = [i for i in range(10,15)]




def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]



#--------------------------------------------------------------------------------------------------------------------



class XJTU(object):

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
        root = self.data_dir

        WC = os.listdir(root)  # Three working conditions WC0:35Hz12kN WC1:37.5Hz11kN WC2:40Hz10kN

        datasetname1 = os.listdir(os.path.join(root, WC[0]))
        datasetname2 = os.listdir(os.path.join(root, WC[1]))
        datasetname3 = os.listdir(os.path.join(root, WC[2]))

        data = []
        lab =[]

        for i in tqdm(range(len(datasetname1))):
            files = os.listdir(os.path.join(root, WC[0], datasetname1[i]))
            for ii in [-4, -3, -2, -1]: # Take the data of the last three CSV files
                path1 = os.path.join(root, WC[0], datasetname1[i], files[ii])
                data1, lab1 = self._data_load(path1, label = label1[i])
                data += data1
                lab += lab1

        for j in tqdm(range(len(datasetname2))):
            files = os.listdir(os.path.join(root, WC[1], datasetname2[j]))
            for jj in [-4, -3, -2, -1]:
                path2 = os.path.join(root, WC[1], datasetname2[j], files[jj])
                data2, lab2 = self._data_load(path2, label = label2[j])
                data += data2
                lab += lab2

        for k in tqdm(range(len(datasetname3))):
            files = os.listdir(os.path.join(root, WC[2], datasetname3[k]))
            for kk in [-4, -3, -2, -1]:
                path3 = os.path.join(root, WC[2], datasetname3[k], files[kk])
                data3, lab3 = self._data_load(path3, label = label3[k])
                data += data3
                lab += lab3

        return [data, lab]
    
    def _data_load(self, filename, label, signal_size=1024):
        '''
        This function is mainly used to generate test data and training data.
        filename:Data location
        '''
        fl = pd.read_csv(filename)
        fl = fl["Horizontal_vibration_signals"].values.reshape(-1,1)

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
