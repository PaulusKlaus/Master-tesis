import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import repeat

from datasets_aug.sequence_dataset import *  #dataset 
from datasets_aug.sequence_aug import *
from .data_utils import *


label1 = [2, 2, 2, 3, 1]
label2 = [1, 2, 3, 2, 2]
label3 = [2, 1, 1, 1, 2]
#--------------------------------------------------------------------------------------------------------------------

HBdata = []
or_faults  = [0,1,2, 5, 7, 8, 9, 14]
cage_fault = []
ir_faults  = [4, 5, 11, 12, 13]

samples = (
    list(zip(HBdata,     repeat("healthy"))) +
    list(zip(ir_faults,  repeat("inner_race"))) +
    list(zip(or_faults,  repeat("outer_race"))) +
    list(zip(cage_fault,  repeat("ball"))) 
)

# stable mapping
class_to_idx = {"healthy": 0, "inner_race": 1, "outer_race": 2, "cage" : 3}

ALL_DATA  = [sid for sid, _ in samples]
ALL_LABEL = [class_to_idx[c] for _, c in samples]

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
        
        # --- WC[0] ---
        for i in tqdm(range(len(datasetname1))):
            folder = os.path.join(root, WC[0], datasetname1[i])
            files = os.listdir(folder)
            # Faults
            for ii in [-4, -3, -2, -1]: # Take the data of the last three CSV files
                path = os.path.join(folder, files[ii])
                data1, lab1 = self._data_load(path, label = label1[i])
                data += data1
                lab += lab1
            # Normals 
            for nn in [0,1,2,3]: # Take the data of the last three CSV files
                path = os.path.join(folder, files[nn])
                data1, lab1 = self._data_load(path, label = 0)
                data += data1
                lab += lab1

        for j in tqdm(range(len(datasetname2))):
            folder = os.path.join(root, WC[1], datasetname2[j])
            files = os.listdir(folder)

            for jj in [-4, -3, -2, -1]:
                path = os.path.join(folder, files[jj])
                data2, lab2 = self._data_load(path, label = label2[j])
                data += data2
                lab += lab2
            for nn in [0,1,2,3]:
                path = os.path.join(folder, files[jj])
                data2, lab2 = self._data_load(path, label = 0)
                data += data2
                lab += lab2

        for k in tqdm(range(len(datasetname3))):
            folder = os.path.join(root, WC[2], datasetname3[k])
            files = os.listdir(folder)
            for kk in [-4, -3, -2, -1]:
                path3 = os.path.join(folder, files[kk])
                data3, lab3 = self._data_load(path3, label = label3[k])
                data += data3
                lab += lab3
            for nn in [0,1,2,3]:
                path3 = os.path.join(folder, files[nn])
                data3, lab3 = self._data_load(path3, label = 0)
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

    def data_prepare(self, split="RA", view=TwoViewDataset, per_class_num = None, classifier_num = None):
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

            
        elif split =="O_N":   # Train on only normal dataset 
            "train: only normal"
            "val: only normal "
            "test: all classes"
            "classifier: all classes"
            normal_pd = data_pd[data_pd["label"] == 0].reset_index(drop=True)
            faults_pd = data_pd[data_pd["label"] != 0].reset_index(drop=True)   

            # split normal into train/val and rest for test and classifier
            train_pd, temp = train_test_split(
                normal_pd,
                test_size=0.40,  # 70% normal -> train, 30% normal -> rest
                random_state=self.random_state,
                shuffle=True,
            )
            val_pd, test_classifier_normals = train_test_split(
                temp,
                test_size=0.50,  # half of the rest -> val, half stays for test/classifier
                random_state=self.random_state,
                shuffle=True,
            ) 
            test_normals, classifier_normals = train_test_split(
                test_classifier_normals,
                test_size=0.50,
                random_state=self.random_state,
                shuffle=True,
            )
            # Split FAULTS into test/classifier (both contain all fault classes)
            #    Stratify by label so each fault type appears in both sets
            test_faults, classifier_faults = train_test_split(
                faults_pd,
                test_size=0.50,
                random_state=self.random_state,
                stratify=faults_pd["label"] if len(faults_pd["label"].unique()) > 1 else None,
            )
            # Add normal samples to test/classifier 
            test_pd = pd.concat([test_normals, test_faults], ignore_index=True).sample(
                frac=1.0, random_state=self.random_state
            ).reset_index(drop=True)

            classifier_pd = pd.concat([classifier_normals, classifier_faults], ignore_index=True).sample(
                frac=1.0, random_state=self.random_state
            ).reset_index(drop=True)

        elif split == "O_A":
            # ordered split (your custom)
            train_pd, temp_pd = train_test_split_order(data_pd, test_size=0.30)
            val_temp, test_pd   = train_test_split_order(temp_pd, test_size=0.5)
            val_pd, classifier_pd   = train_test_split_order(val_temp, test_size=0.5)
        else:
            raise ValueError(f"Unknown split='{split}'. Use 'RA', 'R_NA', or 'O_A'.")

        # --- choose transforms ---
        if split in ("RA", "O_A", "O_N"):  # augmentation on train only
            print(" Augmentetion Hi")
            train_t1 = data_transforms(self.augmentation_1, self.normlizetype)
            train_t2 = data_transforms(self.augmentation_2, self.normlizetype)
        else:  # no augmentation
            train_t1 = data_transforms("normal", self.normlizetype)
            train_t2 = data_transforms("normal", self.normlizetype)

        eval_t1 = data_transforms("normal", self.normlizetype)
        eval_t2 = data_transforms("normal", self.normlizetype)
    
        train_pd = cap_per_class(train_pd, n_per_class=per_class_num, seed=self.random_state)
        test_pd = cap_per_class(test_pd, n_per_class=per_class_num, seed=self.random_state)
        val_pd = cap_per_class(val_pd, n_per_class=per_class_num, seed=self.random_state)
        classifier_pd = cap_per_class(classifier_pd, n_per_class=classifier_num, seed=self.random_state)


        # --- build datasets ---
        train_dataset = view(train_pd, transform_1=train_t1, transform_2=train_t2)
        val_dataset   = view(val_pd,   transform_1=eval_t1,  transform_2=eval_t2)
        test_dataset  = view(test_pd,  transform_1=eval_t1,  transform_2=eval_t2)
        classifier_dataset = view(classifier_pd,  transform_1=eval_t1,  transform_2=eval_t2)

        return train_dataset, val_dataset, test_dataset, classifier_dataset
