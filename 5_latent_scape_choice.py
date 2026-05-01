import matplotlib.pyplot as plt
from utils.log_parser import summarize

import seaborn as sns
import matplotlib.pyplot as plt


def latent_space_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "latent_dim", group_by_blocks)
    return summary_batch_size



paths =[# # hidden channel 160 
        #"checkpoint/SSF_PU_0423-100558/training.log",  # hidden channel 128 

        # No normalization testing (hidden layer 160)
        "checkpoint/SSF_PU_0428-154042/training.log",

       #"checkpoint/SSF_CWRU_0422-091220/training.log", # hidden channel 160 
       #"checkpoint/SSF_CWRU_0423-140218/training.log" # hidden channel 128
        ]
latent_space_test(paths)
