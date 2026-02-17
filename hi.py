import logging
import os
import warnings
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import tqdm
import math

import data_utils.datasets as datasets
import datasets_aug.sequence_dataset as views
from datasets_aug.sequence_aug import *
import models as models


import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)




if torch.cuda.is_available():
    device = torch.device("cuda")
    device_count = torch.cuda.device_count()
    print('using {} gpus'.format(device_count))

else:
    warnings.warn("gpu is not available")
    device = torch.device("cpu")
    device_count = 1
    print('using {} cpu'.format(device_count))

