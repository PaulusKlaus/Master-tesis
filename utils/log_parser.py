import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path


def parse_training_log(path):


    # -------------1------------
    per_class_sample_re = latent_re = re.compile(r'per_class_samples.*?(\d+)', re.IGNORECASE)
    classifier_sample_re = latent_re = re.compile(r'classifier_samples.*?(\d+)', re.IGNORECASE)


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

    norm_type_re = re.compile(r'normlizetype\s*:\s*([A-Za-z_]+)', re.IGNORECASE)

    bs_re = re.compile(r'batch_size.*?(\d+)', re.IGNORECASE) # batch size

    split_type_re = re.compile(r'processing_type\s*:\s*([A-Za-z_]+)', re.IGNORECASE)
    
    #f1-score and recall 
    macro_avg_re = re.compile(
        r'macro avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)',
        re.IGNORECASE
    )

    normal_re = re.compile(
        r'normal\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)',
        re.IGNORECASE
    )

    anomaly_re = re.compile(
        r'anomaly\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)',
        re.IGNORECASE
    )



    runs = []
    current = None

    with open(path, "r", errors="ignore") as f:
        for line in f:

            if "model_name:" in line:
                if current is not None:
                    runs.append(current)
                current = {

                    # ------------------ 2---------------
                    "class_sample":None,
                    "classifier_sample": None,
                    "latent_dim": None,
                    "num_blocks_ssf": None,
                    "aug_1": None,          
                    "aug_2": None,  

                    "split_type": None,

                    # binary test (your anomaly/normal evaluation)
                    "binary_acc": None,

                    "best_val_acc": -np.inf,
                    "best_val_loss": np.inf,
                    "test_acc": -np.inf,
                    "test_loss": np.inf,
                    "hidden_channel": None,
                    "threshold": None,
                    "normalization":None,
                    "batch_size": None,

                    "macro_precision": None,
                    "macro_recall": None,
                    "macro_f1": None,
                    "normal_precision": None,
                    "normal_recall": None,
                    "normal_f1": None,
                    "anomaly_precision": None,
                    "anomaly_recall": None,
                    "anomaly_f1": None,
                }
            # -------------------- 3 -------------------
            m_sample = per_class_sample_re.search(line)
            if m_sample and current["class_sample"] is None:
                current["class_sample"] = int(m_sample.group(1))

            m_samlpe_classifier = classifier_sample_re.search(line)
            if m_samlpe_classifier and current["classifier_sample"] is None:
                current["classifier_sample"] = int(m_samlpe_classifier.group(1))


            if current is None:
                continue
            m_aug1 = aug1_re.search(line)
            if m_aug1 and current["aug_1"] is None:
                current["aug_1"] = m_aug1.group(1).lower()

            m_aug2 = aug2_re.search(line)
            if m_aug2 and current["aug_2"] is None:
                current["aug_2"] = m_aug2.group(1).lower()

            split_type = split_type_re.search(line)
            if split_type: 
                current["split_type"] = split_type.group(1)


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
            
            m_norm = norm_type_re.search(line)
            if m_norm and current["normalization"] is None:
                current["normalization"] = m_norm.group(1).lower()

            m_bs = bs_re.search(line)
            if m_bs and current["batch_size"] is None:
                current["batch_size"] = int(m_bs.group(1))


            m_macro = macro_avg_re.search(line)
            if m_macro:
                current["macro_precision"] = float(m_macro.group(1))
                current["macro_recall"] = float(m_macro.group(2))
                current["macro_f1"] = float(m_macro.group(3))

            m_normal = normal_re.search(line)
            if m_normal:
                current["normal_precision"] = float(m_normal.group(1))
                current["normal_recall"] = float(m_normal.group(2))
                current["normal_f1"] = float(m_normal.group(3))

            m_anomaly = anomaly_re.search(line)
            if m_anomaly:
                current["anomaly_precision"] = float(m_anomaly.group(1))
                current["anomaly_recall"] = float(m_anomaly.group(2))
                current["anomaly_f1"] = float(m_anomaly.group(3))

            

    if current is not None:
        runs.append(current)

    df = pd.DataFrame(runs)
    # Keep only "real runs" (must have aug info + latent/blocks)
    df = df.dropna(subset=["latent_dim", "num_blocks_ssf", "aug_1", "aug_2", "hidden_channel"])  # removes the columbs with Nan
    return df


def summarize(paths, group_col, group_by_blocks=False):
    all_df = []
    for path in paths:
        df = parse_training_log(path)
        df["model"] = path.split("/")[-2]
        all_df.append(df)

    all_df = pd.concat(all_df, ignore_index=True)

    # -------------------------
    # Summary per batch_size
    # -------------------------
    summary_main = (
        all_df
        .groupby([group_col], as_index=False)
        .agg(
            n_runs=("test_acc", "count"),

            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),

            mean_loss=("test_loss", "mean"),
            std_loss=("test_loss", "std"),

            #mean_bin_acc=("binary_acc", "mean"),
            #std_bin_acc=("binary_acc", "std"),

           # mean_val_acc=("best_val_acc", "mean"),
            #mean_val_loss=("best_val_loss", "mean"),

            mean_f1=("macro_f1", "mean"), 
            std_f1=("macro_f1", "std"),
        )
    )

    # Optional: combined score (you can tweak weights)
    summary_main["score"] = 0.5 * summary_main["mean_f1"] + 0.5 * summary_main["mean_acc"]

    summary_main = summary_main.sort_values(group_col, ascending=True)

    print(f"\n=== Summary by {group_col} ===")
    print(summary_main.to_string(index=False))

    # -------------------------
    # Optional: per-block detail
    # -------------------------
    if group_by_blocks:
        summary_blocks = (
            all_df
            .groupby([group_col], as_index=False)
            .agg(
                n_runs=("test_acc", "count"),

                mean_acc=("test_acc", "mean"),
                std_acc=("test_acc", "std"),

             #   mean_bin_acc=("binary_acc", "mean"),
               # std_bin_acc=("binary_acc", "std"),

                mean_f1=("macro_f1", "mean"),
                std_f1=("macro_f1", "std"),
            )
            .sort_values(["batch_size", "num_blocks_ssf"])
        )

        print(f"\n=== Summary by {group_col} x Blocks ===")
        print(summary_blocks.to_string(index=False))

        return summary_main, summary_blocks

    return summary_main