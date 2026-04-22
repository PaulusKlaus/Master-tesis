import matplotlib.pyplot as plt
from utils.log_parser import summarize

import seaborn as sns
import matplotlib.pyplot as plt


def latent_space_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "latent_dim", group_by_blocks)
    return summary_batch_size



paths =["checkpoint/SSF_PU_0422-095246/training.log", #6-10 blocks

       #"checkpoint/SSF_CWRU_0422-091220/training.log",
        ]
latent_space_test(paths)
