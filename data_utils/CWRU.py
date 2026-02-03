import os
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
ir_faults = ['3_OR007_6_0.mat', '3_OR007_6_1.mat', '3_OR007_6_2.mat', '3_OR007_6_3.mat']

load_0 = ['1_IR007_0.mat', '2_B007_0.mat', '3_OR007_6_0.mat', ]
load_1 = ['1_IR007_1.mat', '2_B007_1.mat', '3_OR007_6_1.mat', ]
load_2 = ['1_IR007_2.mat', '2_B007_2.mat', '3_OR007_6_2.mat', ]
load_3 = ['1_IR007_3.mat', '2_B007_3.mat', '3_OR007_6_3.mat', ]


label = [1, 2, 3]
axis = ["_DE_time", "_FE_time", "_BA_time"]



def get_files(root):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data_root1 = os.path.join(root, datasetname[0]) # Normal data 
    data_root2 = os.path.join(root, datasetname[1]) # Drive End Bearing Fault 

    path1 = os.path.join(data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    data, lab = data_load(path1, channel_suffix=axis[0], label=0)  # nThe label for normal data is 0

    for i in tqdm(range(len(load_0))):
        path2 = os.path.join(data_root2, load_0[i])

        data1, lab1 = data_load(path2, channel_suffix=axis[0], label=label[i])
        data += data1
        lab += lab1
    return [data, lab]


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



def data_load(filename, channel_suffix="_DE_time", label=0, signal_size=1024):
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
    fl = np.asarray(fl).squeeze()

    data, lab = [], []
    for start in range(0, len(fl) - signal_size + 1, signal_size):
        end = start + signal_size
        data.append(fl[start:end])
        lab.append(label)

    return data, lab


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    "A composed transform pipeline for dataset preprocessing / augmentation depending on whether it's training or validation."
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            AddGaussian(),
            # TODO: Check why are there every one of them here, does it mean that they are all done allt he time?
            Scale(),
            RandomStretch(),
            RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]

class CWRU(object):
    num_classes = 4
    inputchannel = 1

    def __init__(self, data_dir,normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):

        list_data = get_files(self.data_dir)
        if test:
            test_dataset = OneViewDataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.20, random_state=40, stratify=data_pd["label"])
            train_dataset = OneViewDataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = OneViewDataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset
