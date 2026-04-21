import matplotlib.pyplot as plt
from utils.log_parser import parse_training_log, summarize

import seaborn as sns
import matplotlib.pyplot as plt


def batch_size_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "batch_size", group_by_blocks)
    return summary_batch_size

def plot_bs(df):

    sns.boxplot(data=df, x="batch_size", y="binary_acc")
    plt.title("Binary Accuracy by batch_size")
    plt.show()

    sns.boxplot(data=df, x="batch_size", y="test_acc")
    plt.title("Test Accuracy by batch_size")
    plt.show()

    sns.scatterplot(data=df, x="test_acc", y="binary_acc", hue="batch_size")
    plt.title("Trade-off: Test vs Binary Accuracy")
    plt.show()





paths =[#"checkpoint/SSF_PU_0416-122923/training.log",
        #"checkpoint/SSF_PU_0416-163602/training.log" # 128 batch size
        "checkpoint/SSF_CWRU_0416-155803/training.log",
       ########## "checkpoint/SSF_CWRU_0416-120355/training.log",  # not sure if itis the same augmentetion
        ]
batch_size_test(paths)


df = parse_training_log("checkpoint/SSF_CWRU_0416-105859/training.log")
#plot_bs(df)