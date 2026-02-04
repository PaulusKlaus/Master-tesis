import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import models
import argparse
import sys



def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='resnet18_1d', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='PU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default= r"C:\Users\Palina Yermakova\Documents\PU data", help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='R_A',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

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

        """
        Arguments that should be included inside the args:
        model: torch.nn.Module, 
        train_loader,
        val_loader
        device =None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[Any] = None,
        max_epochs: int = 100,
        workdir: str = "./runs",
        checkpoint_every: int = 1,
        log_fn=print,
        amp: bool = False,
        """

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
        pass

    def _train_epoch(self,):
        pass

    def _validate(self,):
        pass

    def train(self):
        pass