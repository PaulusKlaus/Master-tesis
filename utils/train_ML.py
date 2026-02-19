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


from collections import Counter

def count_labels(loader):
    c = Counter()
    for batch in loader:
        *_, y = batch
        c.update(y.numpy().tolist())
    return c

def uniq(loader):
    s = set()
    for *_, y in loader:
        s |= set(y.numpy().tolist())
    return sorted(s)





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
       
    def _data_loading(self):

        args = self.args

        # --- Load the dataset ----
        Dataset = getattr(datasets, args.data_name) # PU, CWRU ...
        dataset_view = getattr(views, args.data_view) # OneViewDataset, TwoViewDataset

        print("Dataset: ", Dataset)
        logging.info("Dataset class: %s", Dataset)
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
        logging.info("Split sizes: train=%d val=%d test=%d",
                    len(self.train_ds), len(self.val_ds), len(self.test_ds))

        logging.info("Label counts train: %s", count_labels(self.train_loader))
        logging.info("Label counts val:   %s", count_labels(self.val_loader))
        logging.info("Label counts test:  %s", count_labels(self.test_loader))
        print("train uniq:", uniq(self.train_loader))
        print("val uniq:", uniq(self.val_loader))
        print("test uniq:", uniq(self.test_loader))

    def _optimizer_lr_sch(self):
        args = self.args
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
        # For the self-supervised models the latent space is 16 dimentional features, 
        # while for the traditional models output the classes 

        if args.model_name in {"CNN_1d", "resnet18_1d", "MLP"}:
            latent_dim = args.out_channel
        else: 
            latent_dim = 16 
            # Define the classifier
            self.classifier = models.cls(latent_dim = latent_dim, classes = args.out_channel )
            self.cls_opt = optim.SGD(self.classifier.parameters(), 0.01, momentum=args.momentum, weight_decay=args.weight_decay)
            self.cls_lr = optim.lr_scheduler.CosineAnnealingLR(self.cls_opt,  T_max = 60, eta_min=1e-05 )
            self.cls_criterion = nn.CrossEntropyLoss()
            #self.cls_opt = torch.optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=args.weight_decay)
            self.cls_lr = None


        self.model = getattr(models, args.model_name)(in_channel = 1, out_channel = latent_dim)

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

    def _train_classifier (self, frozen_encoder):
        args = self.args
        device = self.device
        encoder = frozen_encoder.eval()

        # 2) Build head  
        classifier = self.classifier.to(device)
        optimizer = self.cls_opt
        lr_sch = self.cls_lr
        criterion = self.cls_criterion

        # 3) Train Loop 
        best_acc = -math.inf
        best_state = None
        no_improve = 0

        for epoch in range (60):
            classifier.train()
            tot_loss = 0.0
            tot_correct = 0
            tot_samp = 0
            # TODO: I think i should train only on a part of the trainin g loader of a part of the validation loader ???
            loop = tqdm.tqdm(self.train_loader, desc = f"Classifier train {epoch}", leave=False)
            for x1, x2, y in loop:
                x1, x2, y = x1.to(device, non_blocking  = True), x2.to(device, non_blocking  = True), y.to(device, non_blocking= True)

                optimizer.zero_grad()

                with torch.no_grad():
                    z1, z2, p1, p2 = encoder(x1, x2)

                outputs = classifier(z1)

                loss = criterion(outputs, y)
                preds = outputs.argmax(dim=1)
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                tot_loss += loss.item()*bs
                tot_samp += bs
                tot_correct += (preds == y ).sum().item()
                loop.set_postfix(loss=f"{tot_loss/tot_samp:.4f}", acc=f"{tot_correct/tot_samp:.4f}")
            
            train_loss = tot_loss / max(1, tot_samp)
            train_acc  = tot_correct / max(1, tot_samp)
            # --- validation -----
            classifier.eval()
            val_correct = 0 
            val_samp = 0
            val_loss = 0.0 

            with torch.no_grad():
                vloop = tqdm.tqdm(self.val_loader, desc=f"LP Val {epoch}", leave=False)
                for  x1, x2, y in vloop:
                    x1, x2, y = x1.to(device, non_blocking  = True), x2.to(device, non_blocking  = True), y.to(device, non_blocking= True)

                    z1, z2, p1, p2 = encoder(x1, x2)
                    outputs = classifier(z1)
                    loss = criterion(outputs, y)

                    bs = y.size(0)
                    val_loss += loss.item() * bs
                    val_samp += bs
                    val_correct += (outputs.argmax(1) == y).sum().item()

                    vloop.set_postfix(val_loss=f"{val_loss/val_samp:.4f}",
                                    val_acc=f"{val_correct/val_samp:.4f}")

            val_acc = val_correct / max(1, val_samp)

            logging.info(
                f"[LP epoch {epoch:02d}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                #f"lr={lr_sch.get_last_lr()[0]:.6g}"
            )
            if val_acc > best_acc:
                best_acc = val_acc 
                best_state = {
                    "epoch": epoch,
                    "backbone_state_dict": encoder.state_dict(),
                    "head_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_acc,
                }
            if lr_sch is not None:

                lr_sch.step()
            
        # 5) Save best
        out_path = os.path.join(self.save_dir, "linear_probe_best.pt")
        torch.save(best_state, out_path)
        logging.info(f"Saved linear-probe checkpoint: {out_path} (best_val_acc={best_acc:.4f})")

        # 6) Test best
        encoder.load_state_dict(best_state["backbone_state_dict"])
        classifier.load_state_dict(best_state["head_state_dict"])
        encoder.eval()
        classifier.eval()

        test_correct = 0
        test_samples = 0
        test_loss = 0.0

        with torch.no_grad():
            for x1, x2, y in self.test_loader:
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                
                y = y.to(device, non_blocking=True)
                z1, z2, p1, p2 = encoder(x1, x2)
                logits = classifier(z1)
                loss = criterion(logits, y)

                bs = y.size(0)
                test_loss += loss.item() * bs
                test_samples += bs
                test_correct += (logits.argmax(1) == y).sum().item()

        test_acc = test_correct / max(1, test_samples)
        logging.info(f"TEST linear-probe: loss={test_loss/test_samples:.4f} acc={test_acc:.4f}")

        return {"best_val_acc": best_acc, "test_acc": test_acc, "ckpt_path": out_path}


    def train(self, pretrained =True, continue_from_best_val_loss_checkopoint = False):
        """
        High level training organization
        """
        # Set up and arguments 
        args = self.args
        self.setup()

        if pretrained == False: 

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
# ------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
        else: 
            best_ckpt_path = './checkpoint/SimSiamResNet_PU_0211-164930/best_pt'
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logging.info(f"Loaded best checkpoint from {best_ckpt_path} " 
                     f"({ckpt.get('best_key')}={ckpt.get('best_value')})")
# --------------------Important -----------------------------------
# Classification head lears from the best checkpont model not the latest 
        # ---- test best checkpoint ----
        if continue_from_best_val_loss_checkopoint: 

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

        # Freeze weights 

        for p in self.model.parameters():
            p.requires_grad = False 

        encoder = self.model.eval()

        self._train_classifier(frozen_encoder = encoder)
         

        

