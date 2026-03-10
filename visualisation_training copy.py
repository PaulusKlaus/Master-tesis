import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


paths_augmentetion = [
  #  "checkpoint/SSF_PU_0303-093038/training.log",
    #"checkpoint/SSF_PU_0303-112311/training.log",
    #"checkpoint/SSF_PU_0303-140932/training.log"
   # "checkpoint/SSF_PU_0304-105004/training.log",
    "checkpoint/SSF_CWRU_0304-133140_working/training.log"
   # "checkpoint/SSF_PU_0304-124405/training.log"  # trying to overcome overfitting
]

def parse_training_log(path):

    latent_re = re.compile(r'latent.*?(\d+)', re.IGNORECASE)
    blocks_re = re.compile(r'num_blocks_ssf.*?(\d+)', re.IGNORECASE)
    hid_ch_size = re.compile(r'hidden_channel.*?(\d+)', re.IGNORECASE)
    lp_re = re.compile(r'\[LP epoch\s+(\d+)\].*?val_loss=([\d\.]+).*?val_acc=([\d\.]+)')
    test_re = re.compile(r'TEST linear-probe:.*?loss=([\d\.]+).*?acc=([\d\.]+)', re.IGNORECASE)
        
    #parse augmentation lines like "aug_1: normal"
    aug1_re = re.compile(r'aug_1\s*:\s*([A-Za-z_]+)', re.IGNORECASE)
    aug2_re = re.compile(r'aug_2\s*:\s*([A-Za-z_]+)', re.IGNORECASE)

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
                    "best_val_acc": -np.inf,
                    "best_val_loss": np.inf,
                    "test_acc": None,
                    "test_loss": None,
                    "hidden_channel": None
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
                current["test_loss"] = float(m_test.group(1))
                current["test_acc"] = float(m_test.group(2))

    if current is not None:
        runs.append(current)

    df = pd.DataFrame(runs)
    # Keep only "real runs" (must have aug info + latent/blocks)
    df = df.dropna(subset=["latent_dim", "num_blocks_ssf", "aug_1", "aug_2", "hidden_channel"])
    return df

def sted_mean_val(paths):
# Calculation of the std and mean per seed
    results = []
    for path in paths:
        df = parse_training_log(path)
        results.append({
            "model": path.split("/")[-2],
            "mean_acc": df["best_val_acc"].mean(),
            "std_acc": df["best_val_acc"].std(),
            "mean_loss": df["best_val_loss"].mean(),
            "std_loss": df["best_val_loss"].std(),
        })
    summary_df = pd.DataFrame(results)
    print(summary_df)


def scatter_plots(paths):

    # 1️⃣ Latent dim vs best val acc
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        sc = plt.scatter(
        df["latent_dim"],
        df["best_val_acc"],
        c=df["num_blocks_ssf"],   # color by block number
        cmap="viridis",           # choose colormap
        alpha=0.8
    )

    plt.xlabel("latent_dim")
    plt.ylabel("best_val_acc")
    plt.title("Latent Dimension vs Best Validation Accuracy")

    plt.colorbar(sc, label="num_blocks_ssf")
    plt.savefig("figures/training_vis/1_latent_vs_val_acc.pdf", bbox_inches="tight")

    # 2️⃣ Latent dim vs best val loss
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        plt.scatter(
            df["latent_dim"],
            df["best_val_loss"],
            alpha=0.7,
            label=path.split("/")[-2]  # cleaner legend name
        )
    plt.xlabel("latent_dim")
    plt.ylabel("best_val_loss")
    plt.title("Latent Dimension vs Best Validation Loss")
    plt.legend()
    plt.savefig("figures/training_vis/2_latent_vs_val_loss.pdf", bbox_inches="tight")

    # 3️⃣ num_blocks_ssf vs best val acc
    plt.figure()


    for path in paths:
        df = parse_training_log(path)
        sc = plt.scatter(
        df["num_blocks_ssf"],
        df["best_val_acc"],
        c=df["latent_dim"],   # color by block number
        cmap="viridis",           # choose colormap
        alpha=0.8
    )
    plt.xlabel("num_blocks_ssf")
    plt.ylabel("best_val_acc")
    plt.title("num_blocks_ssf vs Best Validation Accuracy")
    plt.colorbar(sc, label="latent_dim")
    plt.savefig("figures/training_vis/3_blocks_vs_val_acc.pdf", bbox_inches="tight")

    # 4️⃣ num_blocks_ssf vs best val loss
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        plt.scatter(
            df["num_blocks_ssf"],
            df["best_val_loss"],
            alpha=0.7,
            label=path.split("/")[-2]
        )
    plt.xlabel("num_blocks_ssf")
    plt.ylabel("best_val_loss")
    plt.title("num_blocks_ssf vs Best Validation Loss")
    plt.legend()
    plt.savefig("figures/training_vis/4_blocks_vs_val_loss.pdf", bbox_inches="tight")

    # 5 hidden_channel_size vs best val acc
    plt.figure()

    for path in paths:
        df = parse_training_log(path)
        sc = plt.scatter(
        df["hidden_channel"],
        df["best_val_acc"],
        c=df["num_blocks_ssf"],   # color by block number
        cmap="viridis",           # choose colormap
        alpha=0.8
    )
    plt.xlabel("hidden_channel")
    plt.ylabel("best_val_acc")
    plt.title("hidden_channel vs Best Validation Accuracy")
    plt.colorbar(sc, label="num_blocks_ssf")
    plt.savefig("figures/training_vis/5_hidden_vs_val_acc.pdf", bbox_inches="tight")

    # 6 hidden_channel_size vs best val loss
    plt.figure()
    for path in paths:
        df = parse_training_log(path)
        plt.scatter(
            df["hidden_channel"],
            df["best_val_loss"],
            alpha=0.7,
            label=path.split("/")[-2]
        )
    plt.xlabel("hidden_channel")
    plt.ylabel("best_val_loss")
    plt.title("hidden_channel vs Best Validation Loss")
    plt.legend()
    plt.savefig("figures/training_vis/6_hidden_channel_vs_val_loss.pdf", bbox_inches="tight")


def augmentation_test(paths):
    all_df = []

    for path in paths:
        df = parse_training_log(path)
        df["model"] = path.split("/")[-2]  # keep experiment id
        all_df.append(df)

    all_df = pd.concat(all_df, ignore_index=True)

    # Option A (recommended for SSL): treat (aug_1, aug_2) same as (aug_2, aug_1)
    # -> sort the pair so it's order-invariant
    all_df[["aug_a", "aug_b"]] = np.sort(all_df[["aug_1", "aug_2"]].values, axis=1)

    summary_aug = (
        all_df
        .groupby(["aug_a", "aug_b"], as_index=False)
        .agg(
            n_runs=("best_val_acc", "count"),
            mean_acc=("best_val_acc", "mean"),
            std_acc=("best_val_acc", "std"),
            mean_loss=("best_val_loss", "mean"),
            std_loss=("best_val_loss", "std"),
        )
        .sort_values("mean_acc", ascending=False)
    )

    print(summary_aug.to_string(index=False))


augmentation_test(paths_latent)
scatter_plots(paths_latent)