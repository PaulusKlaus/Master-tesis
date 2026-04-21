import matplotlib.pyplot as plt
from utils.log_parser import summarize

import seaborn as sns
import matplotlib.pyplot as plt


def hidden_layer_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "hidden_channel", group_by_blocks)
    return summary_batch_size



paths =["checkpoint/SSF_PU_0420-145721/training.log", #6-10 blocks
        "checkpoint/SSF_PU_0421-085011/training.log", #1-5 blocks 

       #"checkpoint/SSF_CWRU_0421-130608/training.log",
        ]
hidden_layer_test(paths)
