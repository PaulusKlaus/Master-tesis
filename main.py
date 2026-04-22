
import argparse
import os
from datetime import datetime
import logging
from torch import nn
from itertools import combinations_with_replacement


from utils.tsne import tsne
from utils.loss_SSL import SimSiamLoss
from utils.logger import setlogger
from utils.train_ML import Trainer
from utils.anomaly_detection import *



# ---- This files the data preprocessing and model building and trainin gtogether with the augument passing ------


DATA_DIRS = {
    "CWRU": [r"raw_data/CWRU", 4],
    "JNU": [r"raw_data/JNU/JNU-Bearing-Dataset-main", 4], # {"healthy": 0, "inner_race": 1, "outer_race": 2, "ball" : 3}
    "PU": [r"raw_data/PU", 6], # {"healthy": 0, "inner_race": 1, "combined": 2, "outer_race": 3}
    "SEU": [r"raw_data/SEU/gearbox", 5], # {"healthy": 0, "inner_race": 1, "combined": 2, "outer_race": 3, "ball" : 4}
    "XJTU": [r"raw_data/XJTU/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets", 4]  # TODO: Check this 
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

# ========================= z
# Training Defaults
# =========================
MAX_EPOCH = 50


# =========================
# Argument Parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train")

    # -------------------------------------------------
    # Model Parameters
    # -------------------------------------------------
    parser.add_argument(
        "--model_name",
        type=str,
        choices=MODEL_CONFIG.keys(),
        default="SSF",
        help="The name of the model",
    )

    # -------------------------------------------------
    # Data Parameters
    # -------------------------------------------------
    parser.add_argument(
        "--data_name", # SEU, JNU ,PU , CWRU
        type=str,
        choices=DATA_DIRS.keys(),
        default="CWRU",  # SEU, JNU ,PU , CWRU
        help="The name of the dataset",
    )

    parser.add_argument(
        "--aug_1",
        type=str,
        choices=["gaussian", "normal", "scale", "randomstrech", "randomcrop", "fft"],
        default="normal",
        help="Augmentation type on the online pipeline",
    )

    parser.add_argument(
        "--aug_2",
        type=str,
        choices=["gaussian", "normal", "scale", "randomstrech", "randomcrop", "fft"],
        default="randomcrop",
        help="Augmentation type on the target pipeline",
    )

    parser.add_argument(
        "--data_view",
        type=str,
        default=None,
        help="Dataset view with either one or two tensors",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional override for dataset directory",
    )

    parser.add_argument(
        "--out_channel",
        type=int,
        default=None,
        help="Output classes",
    )

    parser.add_argument(
        "--normlizetype",
        type=str,
        choices=["zero_one", "minus_one_one", "mean_std", "mean"],
        default="mean_std",
        help="Data normalization method",
    )

    parser.add_argument(
        "--processing_type",
        type=str,
        choices=["RA", "R_NA", "O_A", "O_N"],
        default="O_N",
        help=(
            "RA: random split with augmentation | "
            "R_NA: random split without augmentation | "
            "O_A: order split with augmentation | "
            "O_N: order split without augmentation"
        ),
    )

    # -------------------------------------------------
    # Training Parameters
    # -------------------------------------------------
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=MAX_EPOCH,
        help="Maximum number of training epochs",
    )

    parser.add_argument(
        "--classifier_epoch",
        type=int,
        default=50,
        help="Maximum number of classifier epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )

    parser.add_argument(
        "--cuda_device",
        type=str,
        default="0",
        help="CUDA device ID",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoint",
        help="Directory to save the model",
    )

    # -------------------------------------------------
    # Optimization Parameters
    # -------------------------------------------------
    parser.add_argument(
        "--opt",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer type",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Initial learning rate",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["cos", "exp", "stepLR", "fix"],
        default="cos",
        help="Learning rate scheduler type",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="LR scheduler parameter (step / exp)",
    )

    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-5,
        help="Minimum LR for cosine scheduler",
    )

    # -------------------------------------------------
    # Model-Specific Parameters
    # -------------------------------------------------
    parser.add_argument(
        "--latent_space",
        type=int,
        default=128,
        help="Size of the latent space",
    )

    parser.add_argument(
        "--num_blocks_ssf",
        type=int,
        default=5,
        help="Number of convolutional blocks in SSF model",
    )

    parser.add_argument(
        "--hidden_channel",
        type=int,
        default=256,
        help="Hidden channel size",
    )

    parser.add_argument(
        "--per_class_samples",
        type=int,
        default=None,
        help="Number of samples per class",
    )

    parser.add_argument(
        "--classifier_samples",
        type=int,
        default=10,
        help="Number of classifier samples",
    )

    return parser.parse_args()



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

    import random
    augmentations = ['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop']
    pairs = list(combinations_with_replacement(augmentations, 2))
    # pick 3 random pairs
    random_pairs = random.sample(pairs, 5)

    print(random_pairs)
    aug_pairs_best_hiden = [
        ("gaussian", "scale"),        # 0.7622
        ("randomstrech", "gaussian"),       # 0.7511
        ("randomcrop", "randomstrech"),   # 0.7474
        ("randomcrop", "randomcrop"),      # 0.7452
        ("scale", "randomstrech"),           # 0.7437

    ]
    aug_pairs_latent = [
        ("gaussian", "gaussian"),        # 0.7622
        ("scale", "randomstrech"),       # 0.7511
        ("scale", "randomcrop"),   # 0.7474
        ("randomstrech", "randomcrop"),           # 0.7452
        ("gaussian", "randomstrech"),      # 0.7452
      #  ("scale", "scale"),           # 0.7437

    ]
    latent_space = [32,64,128,160,256, 512]
    #latent_space = [256]
    hidden_channel=[160]
    #hidden_channel =[32,64,128,160,256]
    #number_blocks=[1,2,3,4,5,6,7,8,9,10]

   # latent_space = [192]
   # hidden_channel =[128]
    number_blocks=[8]
    batch_sizes =[64]
    
     #"--data_name", # SEU, JNU ,PU , CWRU

    norm = ["mean_std"]

    for pair in aug_pairs_latent:  
        for hidden_size in hidden_channel:
            for features in latent_space:
                for blocks in number_blocks:
                    for normalization in norm: 
                        for batch_size in batch_sizes:
                            for seed in range (3):  # seeds 
                                args.aug_1, args.aug_2 = pair
                                args.latent_space = features
                                args.hidden_channel = hidden_size
                                args.num_blocks_ssf=blocks
                                args.per_class_samples = 100
                                args.classifier_samples = 10
                                args.batch_size = batch_size
                                args.normlizetype=normalization
                                run_id = f"aug={pair} hidden={hidden_size} latent={features} blocks={blocks} seed={seed}"
                                logging.info("=" * 80)
                                logging.info("RUN: %s", run_id)

                                # save the args
                                for k, v in args.__dict__.items():
                                    logging.info("{}: {}".format(k, v))

                                trainer = Trainer(args, save_dir)
                                encoder = trainer.train(pretrained=False, pretrained_dir="./anomaly_detection/SSF_PU_0310-131203/best_pt")
                                device = next(encoder.parameters()).device  # gets cuda or cpu automatically
                                
                                

                                test_pred, test_labels, threshold = run_anomaly_detection(
                                    device, encoder, trainer.train_loader, trainer.test_loader,
                                    normal_class=0, std_factor=2.0, metric="euclidean",
                                    n_jobs=-1, verbose=True,
                                )

                                acc, cm, report = anomaly_metrics_from_multiclass(test_labels, test_pred, normal_class=0)
                                logging.info("Threshold: %s", threshold)
                                logging.info("Binary accuracy: %.6f", acc)
                                logging.info("Confusion matrix:\n%s", cm)
                                logging.info("Report:\n%s", report)

                                rates = per_fault_detection_rate(test_labels, test_pred, normal_class=0)
                                logging.info("Rates: %s", rates)

                                # Classification 
                                trainer.train_classifier(encoder)
                            tsne(device, encoder, trainer.test_loader)
            


