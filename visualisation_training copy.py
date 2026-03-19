import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

#path = "checkpoint/SSF_PU_0226-130304/training.log"   # <-- adjust path if needed
paths_latent = [
    #"checkpoint/SSF_PU_0224-134536/training.log",  #Not working
   # "checkpoint_old/SSF_PU_0224-154807/training.log", # Correlation between latent space and depth of the model on all of the data hidden size 258          normal , random crop
    #"checkpoint/SSF_PU_0226-083310/training.log", # Correlation between latent space and depth of the model ONLY REAL DAMAGE hidden size 258
    "checkpoint_old/SSF_PU_0226-131358/training.log", # Correlation between latent space and depth of the model ONLY REAL DAMAGE Hidden size 128       --------------STANDARD--------------------
   # "checkpoint/SSF_PU_0226-132229/training.log",  # Correlation between latent space and depth of the model ONLY REAL DAMAGE (without combined damage) Hidden size 128 
    "checkpoint_old/SSF_PU_0302-085542/training.log",  # Correlation between latent space and depth of the model ONLY REAL DAMAGE Hidden size 128          NORMAL Gausian
    "checkpoint_old/SSF_PU_0302-141641/training.log", # Correlation between latent space and depth of the model ONLY REAL DAMAGE Hidden size 96 
    "checkpoint_old/SSF_PU_0302-174623/training.log", # Correlation between latent space and depth of the model ONLY REAL DAMAGE Hidden size 64
    #"checkpoint/SSF_PU_0302-214830/training.log", # Correlation between latent space and depth of the model ONLY REAL DAMAGE Hidden size 32
]


def parse_training_log(path):

    latent_re = re.compile(r'latent.*?(\d+)', re.IGNORECASE)
    blocks_re = re.compile(r'num_blocks_ssf.*?(\d+)', re.IGNORECASE)
    hid_ch_size = re.compile(r'hidden_channel.*?(\d+)', re.IGNORECASE)
    lp_re = re.compile(r'\[LP epoch\s+(\d+)\].*?val_loss=([\d\.]+).*?val_acc=([\d\.]+)')

    # test linear probe for the classifcation 
    test_re = re.compile(r'TEST linear-probe:.*?loss=([\d\.]+).*?acc=([\d\.]+)', re.IGNORECASE)
        
    #parse augmentation lines like "aug_1: normal"
    aug1_re = re.compile(r'aug_1\s*:\s*([A-Za-z_]+)', re.IGNORECASE)
    aug2_re = re.compile(r'aug_2\s*:\s*([A-Za-z_]+)', re.IGNORECASE)

    thr_re = re.compile(r'Threshold:\s*([\d\.eE+-]+)', re.IGNORECASE)

    # binary test block
    bin_acc_re = re.compile(r'Binary accuracy:\s*([\d\.]+)', re.IGNORECASE)

    runs = []
    current = None

    with open(path, "r", errors="ignore") as f:
        for line in f:

            if "model_name:" in line:
                if current is not None:
                    runs.append(current)
                current = {
                    "latent_dim": None,
                    "num_blocks_ssf": None,
                    "aug_1": None,          
                    "aug_2": None,  

                    # binary test (your anomaly/normal evaluation)
                    "binary_acc": None,

                    "best_val_acc": -np.inf,
                    "best_val_loss": np.inf,
                    "test_acc": -np.inf,
                    "test_loss": np.inf,
                    "hidden_channel": None,

                    "threshold": None,
                }

            if current is None:
                continue
            m_aug1 = aug1_re.search(line)
            if m_aug1 and current["aug_1"] is None:
                current["aug_1"] = m_aug1.group(1).lower()

            m_aug2 = aug2_re.search(line)
            if m_aug2 and current["aug_2"] is None:
                current["aug_2"] = m_aug2.group(1).lower()

            m_latent = latent_re.search(line)
            if m_latent and current["latent_dim"] is None:
                current["latent_dim"] = int(m_latent.group(1))

            m_blocks = blocks_re.search(line)
            if m_blocks and current["num_blocks_ssf"] is None:
                current["num_blocks_ssf"] = int(m_blocks.group(1))

            m_h_ch_size = hid_ch_size.search(line)
            if m_h_ch_size and current["hidden_channel"] is None:
                current["hidden_channel"] = int(m_h_ch_size.group(1))

            m_lp = lp_re.search(line)
            if m_lp:
                val_loss = float(m_lp.group(2))
                val_acc = float(m_lp.group(3))
                current["best_val_acc"] = max(current["best_val_acc"], val_acc)
                current["best_val_loss"] = min(current["best_val_loss"], val_loss)

            m_test = test_re.search(line)
            if m_test:
                # test accucacy and loss fo rthe classification head
                current["test_loss"] = min(current["test_loss"] ,float(m_test.group(1)))
                current["test_acc"] = max(current["test_acc"],float(m_test.group(2)))

             # Binary accuracy (single number)
            m = bin_acc_re.search(line)
            if m:
                current["binary_acc"] = float(m.group(1))

            m_thr = thr_re.search(line)
            if m_thr:
                current["threshold"] = float(m_thr.group(1))

    if current is not None:
        runs.append(current)

    df = pd.DataFrame(runs)
    # Keep only "real runs" (must have aug info + latent/blocks)
    df = df.dropna(subset=["latent_dim", "num_blocks_ssf", "aug_1", "aug_2", "hidden_channel"])
    return df

def scatter_plots(paths, save_dir="figures/training_vis"):
    # 1️⃣ Latent dim vs best val acc
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        sc = plt.scatter(
        df["latent_dim"],
        df["test_acc"],
        c=df["num_blocks_ssf"],   # color by block number
        cmap="viridis",           # choose colormap
        alpha=0.8
    )

    plt.xlabel("latent_dim")
    plt.ylabel("test_acc")
    plt.title("Latent Dimension vs Best Validation Accuracy")

    plt.colorbar(sc, label="num_blocks_ssf")
    plt.savefig("figures/training_vis/1_latent_vs_test_acc.pdf", bbox_inches="tight")

    # 2️⃣ Latent dim vs best val loss
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        plt.scatter(
            df["latent_dim"],
            df["test_loss"],
            alpha=0.7,
            label=path.split("/")[-2]  # cleaner legend name
        )
    plt.xlabel("latent_dim")
    plt.ylabel("test_loss")
    plt.title("Latent Dimension vs Best Validation Loss")
    plt.legend()
    plt.savefig("figures/training_vis/2_latent_vs_test_loss.pdf", bbox_inches="tight")

    # 3️⃣ num_blocks_ssf vs best val acc
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        sc = plt.scatter(
        df["num_blocks_ssf"],
        df["test_acc"],
        c=df["latent_dim"],   # color by block number
        cmap="viridis",           # choose colormap
        alpha=0.8
    )
    plt.xlabel("num_blocks_ssf")
    plt.ylabel("test_acc")
    plt.title("num_blocks_ssf vs Best Validation Accuracy")
    plt.colorbar(sc, label="latent_dim")
    plt.savefig("figures/training_vis/3_blocks_vs_test_acc.pdf", bbox_inches="tight")


    # 7 Num_blocks vs anomaly detection accuracy
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        sc = plt.scatter(
        df["num_blocks_ssf"],
        df["binary_acc"],
        alpha=0.7
    )
    plt.xlabel("num_blocks_ssf")
    plt.ylabel("binary_acc")
    plt.title("num_blocks_ssf vs Best Binary Accuracy")
    plt.savefig("figures/training_vis/7_blocks_vs_binary_acc.pdf", bbox_inches="tight")
    plt.close()



    # 4️⃣ num_blocks_ssf vs best val loss
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        plt.scatter(
            df["num_blocks_ssf"],
            df["test_loss"],
            alpha=0.7,
            label=path.split("/")[-2]
        )
    plt.xlabel("num_blocks_ssf")
    plt.ylabel("test_loss")
    plt.title("num_blocks_ssf vs Best Validation Loss")
    plt.legend()
    plt.savefig("figures/training_vis/4_blocks_vs_test_loss.pdf", bbox_inches="tight")

    # 5 hidden_channel_size vs best val acc
    plt.figure()

    for path in paths:
        df = parse_training_log(path)
        sc = plt.scatter(
        df["hidden_channel"],
        df["test_acc"],
        c=df["num_blocks_ssf"],   # color by block number
        cmap="viridis",           # choose colormap
        alpha=0.8
    )
    plt.xlabel("hidden_channel")
    plt.ylabel("test_acc")
    plt.title("hidden_channel vs Best Validation Accuracy")
    plt.colorbar(sc, label="num_blocks_ssf")
    plt.savefig("figures/training_vis/5_hidden_vs_test_acc.pdf", bbox_inches="tight")

    # 6 hidden_channel_size vs best val loss
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        plt.scatter(
            df["hidden_channel"],
            df["test_loss"],
            alpha=0.7,
            label=path.split("/")[-2]
        )
    plt.xlabel("hidden_channel")
    plt.ylabel("test_loss")
    plt.title("hidden_channel vs Best Validation Loss")
    plt.legend()
    plt.savefig("figures/training_vis/6_hidden_channel_vs_test_loss.pdf", bbox_inches="tight")


def augmentation_test(paths, order_invariant):
    all_df = []
    for path in paths:
        df = parse_training_log(path)
        df["model"] = path.split("/")[-2]
        all_df.append(df)

    all_df = pd.concat(all_df, ignore_index=True)

    # make augmentation pairs order-invariant (recommended for SSL)
    if order_invariant:
        all_df[["aug_a", "aug_b"]] = np.sort(all_df[["aug_1", "aug_2"]].values, axis=1)
    else:
        all_df["aug_a"] = all_df["aug_1"]
        all_df["aug_b"] = all_df["aug_2"]

    # summary by aug-pair x num_blocks
    summary = (
        all_df
        .groupby(["aug_a", "aug_b", "num_blocks_ssf"], as_index=False)
        .agg(
            n_runs=("test_acc", "count"),
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            mean_loss=("test_loss", "mean"),
            std_loss=("test_loss", "std"),
            mean_bin_acc=("binary_acc", "mean"),
            std_bin_acc=("binary_acc", "std"),

        )
        .sort_values(["aug_a", "aug_b", "num_blocks_ssf"])
    )

    summary_aug = (
        all_df
        .groupby(["aug_a", "aug_b"], as_index=False)
        .agg(
            n_runs=("test_acc", "count"),
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            mean_loss=("test_loss", "mean"),
            std_loss=("test_loss", "std"),
            mean_bin_acc=("binary_acc", "mean"),
            std_bin_acc=("binary_acc", "std"),
        )
        .sort_values("mean_acc", ascending=False)
    )

    print(summary_aug.to_string(index=False))
    return summary


def aug_pair_vs_blocks_accuracy(paths, plot=True, save_dir="figures/training_vis", top_k=3, n_cols=3):
    summary = augmentation_test(paths, True)

    if not plot:
        return summary

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pre-group once: {(aug_a, aug_b): df}
    grouped = {
        k: g.sort_values("num_blocks_ssf")
        for k, g in summary.groupby(["aug_a", "aug_b"], sort=False)
    }

    def pair_order_for(metric_col):
        # mean std per pair, ascending -> most stable first
        return (
            summary.groupby(["aug_a", "aug_b"], sort=False)[metric_col]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

    def plot_grid(pair_order, mean_col, std_col, ylabel, out_name, suptitle):
        n_pairs = len(pair_order)
        n_rows = math.ceil(n_pairs / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            sharex=True, sharey=True,
            squeeze=False
        )
        axes = axes.ravel()

        for idx, pair in enumerate(pair_order):
            ax = axes[idx]
            g = grouped[pair]

            ax.errorbar(
                g["num_blocks_ssf"],
                g[mean_col],
                yerr=g[std_col],
                marker="o",
                linestyle="none",
                capsize=3,
            )
            a, b = pair
            ax.set_title(f"{a}+{b}")
            ax.set_xlabel("num_blocks_ssf")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.2)

        # hide unused axes (IMPORTANT: use the correct axes array!)
        for j in range(n_pairs, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(suptitle, fontsize=16)
        fig.tight_layout()
        fig.savefig(save_dir / out_name, bbox_inches="tight")
        plt.close(fig)

    def plot_topk(pair_order, mean_col, std_col, ylabel, out_name, title):
        best_pairs = pair_order[: min(top_k, len(pair_order))]

        fig, ax = plt.subplots(figsize=(7, 5))
        for (a, b) in best_pairs:
            g = grouped[(a, b)]
            ax.errorbar(
                g["num_blocks_ssf"],
                g[mean_col],
                yerr=g[std_col],
                marker="o",
                linestyle="none",
                capsize=3,
                label=f"{a}+{b}",
            )

        ax.set_xlabel("num_blocks_ssf")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / out_name, bbox_inches="tight")
        plt.close(fig)

    # ---- ACC ----
    order_acc = pair_order_for("mean_acc") # can be changed with std_acc
    plot_grid(
        order_acc, "mean_acc", "std_acc",
        ylabel="mean test_acc (± std)",
        out_name="augpair_vs_blocks_test_acc.pdf",
        suptitle="Augmentation pair vs num_blocks_ssf"
    )
    plot_topk(
        order_acc, "mean_acc", "std_acc",
        ylabel="mean test_acc (± std)",
        out_name=f"augpair_vs_blocks_test_acc_top{top_k}.pdf",
        title=f"Top {top_k} augmentation pairs with highest mean accuracy"
    )

    # ---- BIN ACC ----
    order_bin = pair_order_for("mean_bin_acc")
    plot_grid(
        order_bin, "mean_bin_acc", "std_bin_acc",
        ylabel="mean binary_acc (± std)",
        out_name="augpair_vs_blocks_binary_acc.pdf",
        suptitle="Augmentation pair vs num_blocks_ssf"
    )
    plot_topk(
        order_bin, "mean_bin_acc", "std_bin_acc",
        ylabel="mean binary_acc (± std)",
        out_name=f"augpair_vs_blocks_bin_acc_top{top_k}.pdf",
        title=f"Top {top_k} augmentation pairs with highest mean binary accuracy"
    )

    return summary


def blocks_vs_binary_acc_with_threshold(paths, save_path="figures/training_vis/8_blocks_vs_binary_acc.pdf"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load all runs ---
    all_df = []
    for path in paths:
        df = parse_training_log(path)
        all_df.append(df)

    df = pd.concat(all_df, ignore_index=True)

    # --- Compute mean threshold per block ---
    thr_summary = (
        df.dropna(subset=["threshold"])
          .groupby("num_blocks_ssf", as_index=False)
          .agg(mean_threshold=("threshold", "mean"),
               std_threshold=("threshold", "std"))
          .sort_values("num_blocks_ssf")
    )

    # ------------------ PLOT ------------------
    fig, ax1 = plt.subplots()

    # Binary accuracy (BLUE)
    acc_color = "tab:blue"
    ax1.scatter(
        df["num_blocks_ssf"],
        df["binary_acc"],
        color=acc_color,
        alpha=0.7,
        label="Binary accuracy"
    )
    ax1.set_xlabel("num_blocks_ssf")
    ax1.set_ylabel("Binary accuracy", color=acc_color)
    ax1.tick_params(axis="y", labelcolor=acc_color)

    # Threshold (RED) on second axis
    ax2 = ax1.twinx()
    thr_color = "tab:red"
    ax2.errorbar(
        thr_summary["num_blocks_ssf"],
        thr_summary["mean_threshold"],
        yerr=thr_summary["std_threshold"].fillna(0.0),
        marker="o",
        linestyle="none",
        capsize=4,
        color=thr_color,
        label="Mean threshold ± std"
    )
    ax2.set_ylabel("Threshold", color=thr_color)
    ax2.tick_params(axis="y", labelcolor=thr_color)

    plt.title("num_blocks_ssf vs Binary Accuracy + Mean Threshold")

    # Optional: combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


paths_augmentetion = [

    #Augmentatioon testing 
    #"checkpoint/SSF_CWRU_0312-090033/training.log",    #Full test on cwru for 10 block and all augmentation pairs 
   # "checkpoint/SSF_PU_0312-102943/training.log",
  # "checkpoint/SSF_JNU_0316-103643/training.log"
 # "checkpoint/SSF_XJTU_0317-090229/training.log"
 # "checkpoint/SSF_SEU_0317-143145/training.log"

 # Latent space vs hidden size 
 #"checkpoint/SSF_CWRU_0318-103724/training.log"
 "checkpoint/SSF_PU_0319-085609/training.log" 
]

aug_pair_vs_blocks_accuracy(paths_augmentetion)

#augmentation_test(paths_augmentetion)
scatter_plots(paths_augmentetion)
blocks_vs_binary_acc_with_threshold(paths_augmentetion)