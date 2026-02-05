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
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

import data_utils.datasets as datasets
import datasets_aug.sequence_dataset as views
from datasets_aug.sequence_aug import *
import models as models
from .logger import setlogger

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # Model parameters 
    parser.add_argument('--model_name', type=str, choices = ['CNN_1d', 'SimSiam', 'SSF', 'resnet18_1d', 'AE_1d', 'MLP'], default='resnet18_1d', help='the name of the model')
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
    parser.add_argument('--processing_type', type=str, choices=['RA', 'R_NA', 'O_A'], default='RA',
                        help='RA: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--aug_1', type=str, choices=['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft'], default='normal', help='Augmentation type on the online pipeline')
    parser.add_argument('--aug_2', type=str, choices=['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft'], default='fft', help='Augmentation type on the target pipeline')
    parser.add_argument('--data_view', type=str, choices=['OneViewDataset', 'TwoViewDataset'], default='OneViewDataset', help='Dataset view with either one or two tensors')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['cos', 'exp', 'stepLR', 'fix'], default='cos', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--eta_min', type=float, default=0.001, help='learning rate scheduler parameter for cos ')


    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=5, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=1, help='the interval of log training information')
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

        # --- Load the dataset ----
        Dataset = getattr(datasets, args.data_name) # PU, CWRU ...
        dataset_view = getattr(views, args.data_view) # OneViewDataset, TwoViewDataset

        print("Dataset: ", Dataset)

        self.train_ds, self.val_ds, self.test_ds = Dataset(data_dir = args.data_dir, 
                                                                                      normlizetype= args.normlizetype,
                                                                                      augmentype_1 = args.aug_1,
                                                                                      augmentype_2 = args.aug_2,
                                                                                      rand = 42).data_prepare(split = args.processing_type,
                                                                                                              view = dataset_view)
        

        # ---- DataLoader -----
        self.train_loader = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=False)
        #drop_last=True,              # keep pairs aligned for contrastive loss
        self.val_loader = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=args.batch_size, shuffle=False)

        # ---- Define model -----
        self.model = getattr(models, args.model_name)(in_channel = 1, out_channel = 32)
        
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # ---- Define Optimizer -----
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")
        

        # ---- Define the Learning rate decay ----
        if args.lr_scheduler == 'cos':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = args.max_epoch, eta_min=args.eta_min)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(9)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")
        

        # ---- Load checkpoint ------
        # TODO: implement
        self.start_epoch = 0
        self.model.to(self.device)

        # This need to be different for different models !!!!!
        self.criterion = nn.CrossEntropyLoss()

    def _train_epoch(self, epoch):
        """
        Run one training epoch.
        Return: dict with metrics {'loss': .., 'acc': ..}
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0
        t_samples = 0

        step_start = time.time()
        step = 0

        train_loader = self.train_loader
        device = self.device
        loop = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch in loop:
            # Detect batch format:
            # - OneView: (input, labels)
            # - TwoView: (input1, input2, labels)
            #print("Length of the batch in the loop: ", len(batch))
            if len(batch) == 2:
                inputs, labels = batch 
                inputs = inputs.to(device, non_blocking = True)
                labels = labels.to(device, non_blocking = True)

                # forward pass 
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                pred = outputs.argmax(dim=1)
        
            elif len(batch) == 3: 
                x1, x2, labels = batch
                x1 = x1.to(device, non_blocking = True); x2 = x2.to(device, non_blocking = True); labels = labels.to(device, non_blocking = True)

                # forward pass for contrastive learning 
                z_1, z_2, p_1, p_2 = self.model(x1, x2)
                loss = self.criterion(z_1, z_2, p_1, p_2) #TODO: Need to implement correct 

                #TODO: Add std calculation per channel with torch.no_grad 
            else: 
                raise ValueError("Unexpected batch format from DataLoader")
            
            # backward 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            # Number of samples in a batch
            assert labels.size(0) == inputs.size(0), "Batch size mismatch"
            # loss.item() is the avarage loss per sample in this batch 
            # accumulate metrics
            temp_loss = loss.item() * labels.size(0)  # input.size(0)/labels  is the batch size 
            epoch_loss += temp_loss
            epoch_acc += (pred == labels).sum().item()
            t_samples += labels.size(0)

            avg_loss = epoch_loss / t_samples
            avg_acc = epoch_acc / t_samples
            loop.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
        
        metrics = {"loss": avg_loss, "acc": avg_acc}
        # Print the training information in logging 
        if step % args.print_step == 0:
            batch_loss = epoch_loss / t_samples
            batch_acc = epoch_acc / t_samples
            temp_time = time.time()
            train_time = temp_time - step_start
            step_start = temp_time
            batch_time = train_time / args.print_step if step != 0 else train_time
            sample_per_sec = 1.0 * t_samples / train_time
            logging.info('Epoch: [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                            '{:.1f} examples/sec {:.2f} sec/batch'.format(
                epoch, 
                batch_loss, batch_acc, sample_per_sec, batch_time
            ))
            batch_acc = 0
            batch_loss = 0.0
            t_samples = 0
        step += 1

        return metrics



    def _validate_epoch(self, epoch):
        pass

    def train(self):
        """
        High level training organization
        """
        # Set up and arguments 
        args = self.args
        self.setup()
        
        step = 0
        best_acc = 0.0

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            
            start_time = time.time()
            train_metric = self._train_epoch(epoch)
            val_metric = self._validate_epoch(epoch)

            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
                try: 
                    self.lr_scheduler.step()
                except Exception:
                    self.lr_scheduler.step(val_metric.get("val_loss", None))
            else:
                logging.info('current lr: {}'.format(args.lr))

            

            





if __name__ == "__main__":

    args = parse_args()
    sub_dir = args.model_name+'_'+args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    trainer = Trainer(args, save_dir)
    trainer.train()