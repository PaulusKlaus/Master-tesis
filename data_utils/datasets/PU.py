import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import repeat

from datasets_aug.sequence_dataset import *  
from datasets_aug.sequence_aug import *
from .data_utils import *

# 1 Undamaged (healthy) bearings(6X)
#label1=[0,1,2,3,4,5]  #The undamaged (healthy) bearings data is labeled 0-5
#label1 = [0,0,0,0,0,0]
# 2 Artificially damaged bearings(12X)
#ADBdata = ['KA01','KA03', 'KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']
#label2=[6,7,8,9,10,11,12,13,14,15,16,17]    # The artificially damaged bearings data is labeled 6-17

# 3 Bearings with real damages caused by accelerated lifetime tests(14x)
#RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI04','KI14','KI16','KI17','KI18','KI21']
#label3=[i for i in range(18,18+len(RDBdata))]

#HBdata = ['K001',"K002",'K003','K004','K005','K006']
#or_faults  = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KA04','KA15','KA16','KA22','KA30']
#com_faults = ['KB23','KB24','KB27']
#ir_faults  = ['KI01','KI03','KI05','KI07','KI08','KI04','KI14','KI16','KI17','KI18','KI21']

# Data split suggested by the PU paper 
HBdata = ['K001',"K002",'K003','K004','K005']
#or_faults  = ['KA04','KA15','KA16','KA22','KA30']
#com_faults = ['KB23','KB24','KB27']
#ir_faults  = ['KI04','KI14','KI16','KI18','KI21']


# Big artificial damage
or_faults_a  = ['KA03','KA06','KA08','KA09']  
#real_damage_faults = ['KA16','KB24','KB23','KI16', 'KI18' ]
ir_faults_a  = ['KI07','KI08']

# big real damage 
or_fault_r = ['KA16']
ir_faults_r = ['KI16', 'KI18' ]
#combined faults 
com_fault_re = ['KB24','KB23']



samples = (
    list(zip(HBdata,     repeat("healthy"))) +
    list(zip(ir_faults_a,  repeat("inner_race_a"))) +
    list(zip(or_faults_a,  repeat("outer_race_a")))+
    list(zip(ir_faults_r,  repeat("inner_race_r"))) +
    list(zip(or_fault_r,  repeat("outer_race_r")))+
    list(zip(com_fault_re,  repeat("com_fault_re")))
)

# stable mapping
#class_to_idx = {"healthy": 0, "inner_race": 1, "outer_race": 2} #  "outer_race": 3
#class_to_idx = {"healthy": 0, "inner_race": 1,  "outer_race": 2}
class_to_idx = {"healthy": 0, "inner_race_a": 1, "outer_race_a": 2, "inner_race_r": 3, "outer_race_r": 4, "com_fault_re":5}



ALL_DATA  = [sid for sid, _ in samples]
ALL_LABEL = [class_to_idx[c] for _, c in samples]
#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
state = WC[0] #WC[0] can be changed to different working states




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

            for i in range(1, 20):
                name = f"{state}_{bearing}_{i}"
                d, l = self._data_load(
                    os.path.join(self.data_dir, bearing, f"{name}.mat"),
                    name=name,
                    label=label
                )
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
                test_size=0.60,
                random_state=self.random_state,
                stratify=data_pd["label"],
            )
            val_pd, test_classiferier_temp = train_test_split(
                temp_pd,
                test_size=0.5,
                random_state=self.random_state,
                stratify=temp_pd["label"],
            )
            test_pd, classifier_temp = train_test_split(
                test_classiferier_temp,
                test_size=0.20,
                random_state=self.random_state,
                stratify=test_classiferier_temp["label"],
            )

            classifier_pd, classifier_val_pd = train_test_split(
                classifier_temp,
                test_size=0.50,
                random_state=self.random_state,
                stratify=classifier_temp["label"],
            )

        elif split == "O_A":
            # ordered split (your custom)
            train_pd, temp = train_test_split_order(data_pd, test_size=0.60)
            val_pd, test_class_temp   = train_test_split_order(temp, test_size=0.5)
            test_pd, classifier_temp   = train_test_split_order(test_class_temp, test_size=0.5)
            classifer_val_pd, classifier_pd   = train_test_split_order(classifier_temp, test_size=0.5)
         
        elif split =="O_N":   # Train on only normal dataset 
            "train: only normal"
            "val: only normal "
            "test: all classes"
            "classifier: all classes"
            normal_temp = data_pd[data_pd["label"] == 0].reset_index(drop=True)
            faults_temp = data_pd[data_pd["label"] != 0].reset_index(drop=True)   

            # split normal into train/val and rest for test and classifier
            train_pd, temp = train_test_split(
                normal_temp,
                test_size=0.50,  # 70% normal -> train, 30% normal -> rest
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
                test_size=0.20,
                random_state=self.random_state,
                shuffle=True,
            )
            # Split FAULTS into test/classifier (both contain all fault classes)
            #    Stratify by label so each fault type appears in both sets
            test_faults, classifier_faults = train_test_split(
                faults_temp,
                test_size=0.20,
                random_state=self.random_state,
                stratify=faults_temp["label"] if len(faults_temp["label"].unique()) > 1 else None,
            )
            # Add normal samples to test/classifier 
            test_pd = pd.concat([test_normals, test_faults], ignore_index=True).sample(
                frac=1.0, random_state=self.random_state
            ).reset_index(drop=True)

            classifier_temp = pd.concat([classifier_normals, classifier_faults], ignore_index=True).sample(
                frac=1.0, random_state=self.random_state
            ).reset_index(drop=True)

            classifier_pd, classifier_val_pd = train_test_split(
                classifier_temp,
                test_size=0.50,
                random_state=self.random_state,
                stratify=classifier_temp["label"] if len(classifier_temp["label"].unique()) > 1 else None,
            )
        
        else:
            raise ValueError(f"Unknown split='{split}'. Use 'RA', 'R_NA', or 'O_A'.")

        # --- choose transforms ---
        if split in ("RA", "O_A", "O_N"):  # augmentation on train only

            train_t1 = data_transforms(self.augmentation_1, self.normlizetype)
            train_t2 = data_transforms(self.augmentation_2, self.normlizetype)
        else:  # no augmentation
            train_t1 = data_transforms("normal", self.normlizetype)
            train_t2 = data_transforms("normal", self.normlizetype)

        eval_t1 = data_transforms("normal", self.normlizetype)
        eval_t2 = data_transforms("normal", self.normlizetype)

        # --- build datasets ---
        # --- optional per-class caps ---
        n_p_class= per_class_num
        train_pd = cap_per_class(train_pd, n_per_class=n_p_class, seed=self.random_state)
        test_pd = cap_per_class(test_pd, n_per_class=n_p_class, seed=self.random_state)
        val_pd = cap_per_class(val_pd, n_per_class=n_p_class, seed=self.random_state)
        classifier_pd = cap_per_class(classifier_pd, n_per_class=classifier_num, seed=self.random_state)
        classifer_val_pd= cap_per_class(classifier_val_pd, n_per_class=classifier_num, seed= self.random_state+1)

        train_dataset = view(train_pd, transform_1=train_t1, transform_2=train_t2)
        val_dataset   = view(val_pd,   transform_1=eval_t1,  transform_2=eval_t2)
        test_dataset  = view(test_pd,  transform_1=eval_t1,  transform_2=eval_t2)
        classifier_dataset = view(classifier_pd,  transform_1=eval_t1,  transform_2=eval_t2)
        classifier_val_dataset = view(classifer_val_pd, transform_1=eval_t1, transform_2=eval_t2)

        return train_dataset, val_dataset, test_dataset, classifier_dataset, classifier_val_dataset

