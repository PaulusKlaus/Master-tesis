import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from datasets_aug.sequence_dataset import *  
from datasets_aug.sequence_aug import *
from data_utils.data_utils import *


# 1 Undamaged (healthy) bearings(6X)
HBdata = ['K001',"K002",'K003','K004','K005','K006']
label1=[0,1,2,3,4,5]  #The undamaged (healthy) bearings data is labeled 0-5

# 2 Artificially damaged bearings(12X)
ADBdata = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']
label2=[6,7,8,9,10,11,12,13,14,15,16,17]    # The artificially damaged bearings data is labeled 6-17

# 3 Bearings with real damages caused by accelerated lifetime tests(14x)
RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI14','KI16','KI17','KI18','KI21']
label3=[i for i in range(18,18+len(RDBdata))]

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
state = WC[0] #WC[0] can be changed to different working states

ALL_DATA  = HBdata + ADBdata + RDBdata
ALL_LABEL = label1 + label2 + label3



class PU(object):
    def __init__(self, data_dir, normlizetype, rand):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.random_state = rand

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

    def data_prepare(self, test = False):
        list_data = self._get_files()

        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        if test:
            # You can keep labels even if test=True; OneViewDataset just won't use them
            return OneViewDataset(data_pd, test=True, transform=None)

        train_pd, val_pd = train_test_split(
            data_pd,
            test_size=0.20,
            random_state=self.random_state,
            stratify=data_pd["label"]
        )

        train_dataset = OneViewDataset(train_pd, transform=data_transforms('train', self.normlizetype))
        val_dataset   = OneViewDataset(val_pd,   transform=data_transforms('val',   self.normlizetype))
        return train_dataset, val_dataset
