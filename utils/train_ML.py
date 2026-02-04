import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import models



class Trainer(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def save_checkpoint(self):
        pass

    def setup(self):
        pass

    def _train_epoch(self,):
        pass

    def _validate(self,):
        pass

    def train(self):
        pass