
import argparse
import os
from datetime import datetime
import logging
from torch import nn

from utils.loss_SSL import SimSiamLoss
from utils.logger import setlogger
from utils.train_ML import Trainer



# ---- This files the data preprocessing and model building and trainin gtogether with the augument passing ------


DATA_DIRS = {
    "CWRU": [r"raw_data/CWRU", 4],
    "JNU": [r"raw_data/JNU/JNU-Bearing-Dataset-main", 12],
    "PU": [r"raw_data/PU", 4],
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
        "SimSiamResNet": {
        "data_view": "TwoViewDataset",
        "task": "self_supervised",
        "criterion": SimSiamLoss,
    },
}

max_epoc = 20


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # Model parameters 
    parser.add_argument('--model_name', type=str, choices = MODEL_CONFIG.keys(),default='SSF', help='the name of the model')
        # Data parameters 
    parser.add_argument("--data_name",type=str, choices=DATA_DIRS.keys(), default="SEU", help="the name of the dataset",
                    )
    parser.add_argument('--aug_1', type=str, choices=['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft'], default='normal', help='Augmentation type on the online pipeline')
    parser.add_argument('--aug_2', type=str, choices=['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft'], default='randomcrop', help='Augmentation type on the target pipeline')
    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=max_epoc, help='max number of epoch')
    parser.add_argument('--classifier_epoch', type=int, default=50, help='max number of epoch')

    parser.add_argument('--data_view', type=str, default=None, help='Dataset view with either one or two tensors')
    
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    #parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')
    

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


    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['cos', 'exp', 'stepLR', 'fix'], default='cos', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--eta_min', type=float, default=0.00001, help='learning rate scheduler parameter for cos ')



    args = parser.parse_args()
    return args



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
    for i in range (2):
        i+=1
        trainer = Trainer(args, save_dir)
        trainer.train(pretrained=False)





