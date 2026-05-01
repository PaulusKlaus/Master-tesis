import pandas as pd
import matplotlib.pyplot as plt
from utils.log_parser import parse_training_log, summarize

import seaborn as sns
import matplotlib.pyplot as plt


def normalization_test(paths, group_by_blocks=False):
    summary_norm = summarize(paths, "normalization", group_by_blocks)
    return summary_norm

def plot_normalization(df):

    sns.boxplot(data=df, x="normalization", y="binary_acc")
    plt.title("Binary Accuracy by Normalization")
    plt.show()

    sns.boxplot(data=df, x="normalization", y="test_acc")
    plt.title("Test Accuracy by Normalization")
    plt.show()

    sns.scatterplot(data=df, x="test_acc", y="binary_acc", hue="normalization")
    plt.title("Trade-off: Test vs Binary Accuracy")
    plt.show()





paths =["checkpoint/SSF_PU_0415-141949/training.log", # aug pu 

       # "checkpoint/SSF_PU_0422-145545/training.log",

        #"checkpoint/SSF_CWRU_0415-161214/training.log",
        #"checkpoint/SSF_CWRU_0416-105859/training.log"  # aug pu
        ]
normalization_test(paths)


df = parse_training_log("checkpoint/SSF_CWRU_0416-105859/training.log")
#plot_normalization(df)