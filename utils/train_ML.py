import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import tqdm
import math

import data_utils.datasets as datasets
import datasets_aug.sequence_dataset as views
from datasets_aug.sequence_aug import *
import models as models
from .logger import setlogger
from .loss_SSL  import SimSiamLoss



DATA_DIRS = {
    "CWRU": [r"raw_data/CWRU", 3],
    "JNU": [r"raw_data/JNU/JNU-Bearing-Dataset-main", 12],
    "PU": [r"raw_data/PU", 32],
    "SEU": [r"raw_data/SEU/gearbox", 10],
    "XJTU": [r"raw_data/XJTU", 10]  # TODO: Check this 
}

MODEL_CONFIG = {
    "CNN_1d": {
        "data_view": "OneViewDataset",
        "task": "supervised",
        "criterion": nn.CrossEntropyLoss,
    },
    "AE_1d": {
        "data_view": "OneViewDataset",
        "task": "reconstruction",
        "criterion": nn.MSELoss,
    },
    "SimSiam": {
        "data_view": "TwoViewDataset",
        "task": "self_supervised",
        "criterion": SimSiamLoss,
    },
    "resnet18_1d": {
        "data_view": "OneViewDataset",
        "task": "supervised",
        "criterion": nn.CrossEntropyLoss,
    },
    "MLP": {
        "data_view": "OneViewDataset",
        "task": "supervised",
        "criterion": nn.CrossEntropyLoss,
    },
    "SSF": {
        "data_view": "TwoViewDataset",
        "task": "self_supervised",
        "criterion": SimSiamLoss,
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # Model parameters 
    parser.add_argument('--model_name', 
                        type=str, 
                        choices = MODEL_CONFIG.keys(),
                        default='SimSiam', 
                        help='the name of the model'
                        )
    parser.add_argument('--data_view', 
                        type=str, 
                        default=None, 
                        help='Dataset view with either one or two tensors'
                        )
    
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    #parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    
    # Data parameters 
    parser.add_argument("--data_name",
                        type=str,
                        choices=DATA_DIRS.keys(),
                        default="PU",
                        help="the name of the dataset",
                    )
    parser.add_argument("--data_dir",
                        type=str,
                        default=None,
                        help="optional override for dataset directory",
                    )
    parser.add_argument("--out_channel",
                        type=int,
                        default=None,
                        help="output classes",
                    )
                    
                    
    parser.add_argument('--normlizetype', type=str, choices=["zero_one", "minus_one_one", 'mean_std'], default='minus_one_one', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['RA', 'R_NA', 'O_A'], default='RA',
                        help='RA: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--aug_1', type=str, choices=['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft'], default='normal', help='Augmentation type on the online pipeline')
    parser.add_argument('--aug_2', type=str, choices=['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft'], default='fft', help='Augmentation type on the target pipeline')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['cos', 'exp', 'stepLR', 'fix'], default='cos', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--eta_min', type=float, default=0.00001, help='learning rate scheduler parameter for cos ')


    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=20, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=2, help='the interval of log training information')
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
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded weights, start_epoch and optimizer from chechpoint")

        return start_epoch, model, optimizer
    
    
    def _build_criterion(self, model_name: str) -> nn.Module:
        # supervised classifiers
        if model_name in {"CNN_1d", "resnet18_1d", "MLP"}:
            return nn.CrossEntropyLoss()

        # self-supervised
        if model_name in {"SimSiam", "SSF"}:
            return SimSiamLoss()

        # example: autoencoder reconstruction
        if model_name == "AE_1d":
            return nn.MSELoss()

        # if SSF has its own loss, put it here
        if model_name == "SSF":
            # TODO: replace with actual SSF loss
            return nn.CrossEntropyLoss()

        raise ValueError(f"Unknown model_name for criterion: {model_name}")
    
    def _data_loading(self):

        args = self.args

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

    def _optimizer_lr_sch(self):
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

        self._data_loading()

        # ---- Define model -----
        self.model = getattr(models, args.model_name)(in_channel = 1, out_channel = args.out_channel)
        if self.device_count > 1:
                            self.model = torch.nn.DataParallel(self.model)
        self._optimizer_lr_sch()
        
        # ---- Load checkpoint ------
        # TODO: implement
        self.start_epoch = 0
        self.model.to(self.device)

        # This need to be different for different models !!!!!
        
        self.criterion = args.critetion()

    def _train_epoch(self, epoch):
        """
        Run one training epoch.
        Return: dict with metrics {'loss': .., 'acc': ..}
        """
        args = self.args
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0
        t_samples = 0

        step_start = time.time()
        step = 0

        train_loader = self.train_loader
        device = self.device
        loop = tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(loop):
            self.optimizer.zero_grad()

            # ---- Supervised ----
            if args.task == "supervised":
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking = True)
                labels = labels.to(device, non_blocking = True)

                output = self.model(inputs)
                loss = self.criterion(output, labels)
                preds = output.argmax(dim=1)
                bs = labels.size(0)

            # ---- Self-Supervised (two views) ----
            elif args.task == "self_supervised":
                x1, x2, labels = batch
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

                z1, z2, p1, p2 = self.model(x1, x2)
                loss = self.criterion(z1, z2, p1, p2)

                preds = None
                bs = labels.size(0)
                labels = None 

            elif args.task == "reconstruction":
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)

                recon = self.model(inputs)
                loss = self.criterion(recon, inputs)
                bs = inputs.size(0)
                preds = None
                labels = None
            else: 
                raise ValueError("Unexpected batch format from DataLoader")
            
            # backward + step
            loss.backward()
            self.optimizer.step()

            # metrics
            epoch_loss += loss.item() * bs
            t_samples += bs

            avg_loss = epoch_loss / t_samples

            # accuracy only for supervised
            if args.task == "supervised":
                epoch_acc += (preds == labels).sum().item()
                avg_acc = epoch_acc / t_samples
                loop.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
            else:
                loop.set_postfix(loss=f"{avg_loss:.4f}")
        
            # return metrics
        metrics = {"loss": avg_loss}
        if args.task == "supervised":
            metrics["acc"] = avg_acc

        return metrics



    def _val_test_epoch(self, val : bool = True):
        #TODO:Check this implemetnation 
        """
        Run validation epoch (no_grad).
        Returns: dict with metrics
        """
        args = self.args 
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        loader = self.val_loader if val else self.test_loader
        device = self.device
        prefix = "val" if val else "test"

        with torch.no_grad():
            loop = tqdm.tqdm(loader, desc=f"{prefix} Epoch", leave=False)
            for batch in loop:
                # ----- supervised -----
                if args.task == "supervised":
                    inputs, labels = batch
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    bs = labels.size(0)
                    total_correct += (preds == labels).sum().item()

                # ----- self-supervised -----
                elif args.task == "self_supervised":
                    x1, x2, _labels = batch
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)

                    # SimSiam-style forward
                    z1, z2, p1, p2 = self.model(x1, x2)
                    loss = self.criterion(z1, z2, p1, p2)


                    bs = x1.size(0)

                # ----- reconstruction -----
                elif args.task == "reconstruction":
                    inputs, _labels = batch
                    inputs = inputs.to(device, non_blocking=True)

                    recon = self.model(inputs)
                    loss = self.criterion(recon, inputs)

                    bs = inputs.size(0)

                else:
                    raise ValueError(f"Unknown task: {args.task}")

           
                total_loss += loss.item() * bs
                total_samples += bs

                avg_loss = total_loss / total_samples

                if args.task == "supervised":
                    avg_acc = total_correct / total_samples
                    loop.set_postfix(**{f"{prefix}_loss": f"{avg_loss:.4f}",
                                        f"{prefix}_acc": f"{avg_acc:.4f}"})
                else:
                    loop.set_postfix(**{f"{prefix}_loss": f"{avg_loss:.4f}"})

        metrics = {f"{prefix}_loss": avg_loss}
        if args.task == "supervised":
            metrics[f"{prefix}_acc"] = avg_acc
        return metrics
    

    

    def train(self):
        """
        High level training organization
        """
        # Set up and arguments 
        args = self.args
        self.setup()
        

        # pick selection metric based on task
        if args.task == "supervised":
            best_value = -math.inf
            better = lambda current, best: current > best
            select_key = "val_acc"
        else:
            best_value = math.inf
            better = lambda current, best: current < best
            select_key = "val_loss"

        best_ckpt_path = os.path.join(self.save_dir, "best_pt")

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            
            train_metric = self._train_epoch(epoch)
            val_metric = self._val_test_epoch()

            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info(f"current lr: {self.lr_scheduler.get_last_lr()[0]:.6g}")
                self.lr_scheduler.step()

            # ----- save best checkpoint -----
            current_value = val_metric[select_key]
            if better(current_value, best_value):
                best_value = current_value
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
                        "best_key": select_key,
                        "best_value": best_value,
                        "task": args.task,
                    },
                    best_ckpt_path,
                )
                logging.info(f"Saved best checkpoint to {best_ckpt_path} ({select_key}={best_value:.4f})")


            # ----- logging (conditional acc) -----
            msg = f"Epoch {epoch:03d} Train loss {train_metric['loss']:.4f}"
            if "acc" in train_metric:
                msg += f" acc {train_metric['acc']:.4f}"

            msg += f" | Val loss {val_metric['val_loss']:.4f}"
            if "val_acc" in val_metric:
                msg += f" acc {val_metric['val_acc']:.4f}"

            logging.info(msg)

         # ---- test at end (current model) ----
        test_metric = self._val_test_epoch(val=False)
        msg = f"TEST (last): loss {test_metric['test_loss']:.4f}"
        if "test_acc" in test_metric:
            msg += f" acc {test_metric['test_acc']:.4f}"
        logging.info(msg)

        # ---- test best checkpoint ----
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logging.info(f"Loaded best checkpoint from {best_ckpt_path} "
                        f"({ckpt.get('best_key')}={ckpt.get('best_value')})")

            test_metric = self._val_test_epoch(val=False)
            msg = f"TEST (best): loss {test_metric['test_loss']:.4f}"
            if "test_acc" in test_metric:
                msg += f" acc {test_metric['test_acc']:.4f}"
            logging.info(msg)

        




if __name__ == "__main__":

    args = parse_args()

    if args.data_dir is None:
        args.data_dir = DATA_DIRS[args.data_name][0]
        args.out_channel = DATA_DIRS[args.data_name][1]

    args.data_view =  MODEL_CONFIG[args.model_name]["data_view"]
    args.task = MODEL_CONFIG[args.model_name]["task"]
    args.critetion = MODEL_CONFIG[args.model_name]["criterion"]

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
