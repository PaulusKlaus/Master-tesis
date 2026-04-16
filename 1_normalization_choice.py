import pandas as pd
import matplotlib.pyplot as plt
from utils.log_parser import parse_training_log

import seaborn as sns
import matplotlib.pyplot as plt


def normalization_test(paths, group_by_blocks=False):
    all_df = []
    for path in paths:
        df = parse_training_log(path)
        df["model"] = path.split("/")[-2]
        all_df.append(df)

    all_df = pd.concat(all_df, ignore_index=True)

    # -------------------------
    # Summary per normalization
    # -------------------------
    summary_norm = (
        all_df
        .groupby(["normalization"], as_index=False)
        .agg(
            n_runs=("test_acc", "count"),

            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),

            mean_loss=("test_loss", "mean"),
            std_loss=("test_loss", "std"),

            mean_bin_acc=("binary_acc", "mean"),
            std_bin_acc=("binary_acc", "std"),

            mean_val_acc=("best_val_acc", "mean"),
            mean_val_loss=("best_val_loss", "mean"),
        )
    )

    # Optional: combined score (you can tweak weights)
    summary_norm["score"] = 0.5 * summary_norm["mean_bin_acc"] + 0.5 * summary_norm["mean_acc"]

    summary_norm = summary_norm.sort_values("score", ascending=False)

    print("\n=== Normalization Summary ===")
    print(summary_norm.to_string(index=False))

    # -------------------------
    # Optional: per-block detail
    # -------------------------
    if group_by_blocks:
        summary_blocks = (
            all_df
            .groupby(["normalization"], as_index=False)
            .agg(
                n_runs=("test_acc", "count"),

                mean_acc=("test_acc", "mean"),
                std_acc=("test_acc", "std"),

                mean_bin_acc=("binary_acc", "mean"),
                std_bin_acc=("binary_acc", "std"),
            )
            .sort_values(["normalization", "num_blocks_ssf"])
        )

        print("\n=== Normalization x Blocks ===")
        print(summary_blocks.to_string(index=False))

        return summary_norm, summary_blocks

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





paths =[#"checkpoint/SSF_PU_0415-141949/training.log", # aug pu 
        #"checkpoint/SSF_CWRU_0415-161214/training.log",
        "checkpoint/SSF_CWRU_0416-105859/training.log"  # aug pu
        ]
normalization_test(paths)


df = parse_training_log("checkpoint/SSF_CWRU_0416-105859/training.log")
plot_normalization(df)