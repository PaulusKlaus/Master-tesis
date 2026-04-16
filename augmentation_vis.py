import re
import pandas as pd
import matplotlib.pyplot as plt

path = r"checkpoint/SSF_CWRU_0416-105859/training.log"

# Regex patterns for lines in your log
re_run = re.compile(r"RUN:\s+aug=\('([^']+)'\s*,\s*'([^']+)'\).*seed=(\d+)")
re_aug1 = re.compile(r"aug_1:\s+(\S+)")
re_aug2 = re.compile(r"aug_2:\s+(\S+)")
re_best_val = re.compile(r"Loaded best checkpoint .*?\(val_loss=([-\d.]+)\)")
re_lp_test = re.compile(r"TEST linear-probe:\s+loss=([-\d.]+)\s+acc=([-\d.]+)")

rows = []
current = None

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Start of a run
        m = re_run.search(line)
        if m:
            # save previous run if it has lp results
            if current and ("lp_test_acc" in current):
                rows.append(current)

            aug_a, aug_b, seed = m.groups()
            current = {
                "aug_a": aug_a,
                "aug_b": aug_b,
                "seed": int(seed),
            }
            continue

        if current is None:
            continue

        # Sometimes aug is also printed as aug_1/aug_2
        m = re_aug1.search(line)
        if m:
            current["aug_a"] = m.group(1)
            continue
        m = re_aug2.search(line)
        if m:
            current["aug_b"] = m.group(1)
            continue

        # Best SSL checkpoint val_loss
        m = re_best_val.search(line)
        if m:
            current["ssl_best_val_loss"] = float(m.group(1))
            continue

        # Linear probe test metrics (this is usually what you summarized later)
        m = re_lp_test.search(line)
        if m:
            current["lp_test_loss"] = float(m.group(1))
            current["lp_test_acc"] = float(m.group(2))
            continue

# append last run
if current and ("lp_test_acc" in current):
    rows.append(current)

df = pd.DataFrame(rows)

# Sanity check
print("Parsed runs:", len(df))
print(df.head())

# ---- Aggregate across seeds (mean/std) per augmentation pair ----
grouped = (
    df.groupby(["aug_a", "aug_b"])
      .agg(
          n_runs=("seed", "count"),
          mean_acc=("lp_test_acc", "mean"),
          std_acc=("lp_test_acc", "std"),
          mean_loss=("lp_test_loss", "mean"),
          std_loss=("lp_test_loss", "std"),
      )
      .reset_index()
)

# If you only have 1 seed for some combos, std becomes NaN; replace with 0.0 if you want
grouped[["std_acc", "std_loss"]] = grouped[["std_acc", "std_loss"]].fillna(0.0)

# ---- Plot mean loss ----
df_sorted = grouped.sort_values("mean_loss")

plt.figure(figsize=(10, 6))
plt.bar(range(len(df_sorted)), df_sorted["mean_loss"])
plt.xticks(
    range(len(df_sorted)),
    df_sorted["aug_a"] + "+" + df_sorted["aug_b"],
    rotation=90,
)
plt.ylabel("Mean LP Test Loss")
plt.title("Augmentation Comparison (Linear Probe Test Loss)")
plt.tight_layout()
plt.tight_layout()
plt.savefig("figures/training_vis/augmentetion.pdf", format="pdf", dpi=300)
plt.close()  