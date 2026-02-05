import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import argparse
import sys
from datetime import datetime


import data_utils.datasets as datasets
import datasets_aug.sequence_dataset as views
from datasets_aug.sequence_aug import *
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters

    # Model parameters 
    parser.add_argument('--model_name', type=str, choices = [], default='resnet18_1d', help='the name of the model')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # Data parameters 
    parser.add_argument('--data_name', type=str, choices=['CWRU', 'JNU', 'PU', 'SEU', 'XJTU'], default='PU', help='the name of the data')
    parser.add_argument('--data_dir', type=str,choices=[r"raw_data\CWRU",
                                                          r"raw_data\JNU\JNU-Bearing-Dataset-main",
                                                            r"raw_data\PU", r"raw_data\SEU\gearbox",
                                                              r"raw_data\XJTU"],
                                                         default= r"raw_data\PU", help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=["zero_one", "minus_one_one", 'mean_std'], default='minus_one_one', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['RA', 'R_NA', 'O_A'], default='R_A',
                        help='RA: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--aug_1', type=str, choices=['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft'], default='normal', help='Augmentation type on the online pipeline')
    parser.add_argument('--aug_2', type=str, choices=['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft'], default='fft', help='Augmentation type on the target pipeline')
    parser.add_argument('--data_view', type=str, choices=['OneViewDataset', 'TwoViewDataset'], default='OneViewDataset', help='Dataset view with either one or two tensors')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')


    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=50, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=50, help='the interval of log training information')
    args = parser.parse_args()
    return args



class Trainer(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def save_checkpoint(self, epoch, model, optimizer, scheduler, epoch_loss, acc, filename, msg):
        """
        Save model checkpoint.
    
        Args:
            epoch (int): Current epoch number.
            model (nn.Module): Model to save.
            optimizer (Optimizer): Optimizer to save.
            acc (float): Accuracy value to save.
            filename (str): Path to save the checkpoint file.
            msg (str): Message to display after saving.
        """
        state = {
            'epoch': epoch,
            'arch': self.args.arch,  #backbone
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch_loss':epoch_loss,
            'top1_acc': acc
        }
        torch.save(state, filename)
        print(msg)
    
    def load_checkpoint(self, model, optimizer, filename):
        """
        Load model checkpoint.

        Args:
            model (nn.Module): Model to load checkpoint into.
            optimizer (Optimizer): Optimizer to load checkpoint into.
            filename (str): Path to the checkpoint file.
    
        Returns:
            tuple: (start_epoch, model, optimizer)
        """
        checkpoint = torch.load(filename, map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimoptimizer_state_dictizer'])
        print("Loaded weights, start_epoch and optimizer from chechpoint")

        return start_epoch, model, optimizer


    def setup(self):
        """
        Initialise model, dataset, loss, and optimizer from the argparse arguments
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the dataset
        Dataset = getattr(datasets, args.data_name)
        dataset_view = getattr(views, args.data_view)

        print("Dataset: ", Dataset)

        self.datasets = {}
        self.datasets['train'], self.datasets['val'], self.datasets['test'] = Dataset(data_dir = args.data_dir, 
                                                                                      normlizetype= args.normlizetype,
                                                                                      augmentype_1 = args.aug_1,
                                                                                      augmentype_2 = args.aug_2,
                                                                                      rand = 42).data_prepare(split = args.processing_type,
                                                                                                              view = dataset_view)
        
        


    def _train_epoch(self,):
        pass

    def _validate(self,):
        pass

    def train(self):
        pass


if __name__ == "__main__":

    args = parse_args()
    sub_dir = args.model_name+'_'+args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    #setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    trainer = Trainer(args, save_dir)