import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "checkpoint/SSF_PU_0226-083310/training.log"   # <-- adjust path if needed

# ---------- Regex patterns ----------
latent_re = re.compile(r'latent.*?(\d+)', re.IGNORECASE)
blocks_re = re.compile(r'num_blocks_ssf.*?(\d+)', re.IGNORECASE)
lp_re = re.compile(r'\[LP epoch\s+(\d+)\].*?val_loss=([\d\.]+).*?val_acc=([\d\.]+)')
test_re = re.compile(r'TEST linear-probe:.*?loss=([\d\.]+).*?acc=([\d\.]+)', re.IGNORECASE)

runs = []
current = None

with open(path, "r", errors="ignore") as f:
    for line in f:

        # Detect new run
        if "model_name:" in line:
            if current is not None:
                runs.append(current)
            current = {
                "latent_dim": None,
                "num_blocks_ssf": None,
                "best_val_acc": -np.inf,
                "best_val_loss": np.inf,
                "test_acc": None,
                "test_loss": None
            }

        if current is None:
            continue

        # Extract latent dim
        m_latent = latent_re.search(line)
        if m_latent and current["latent_dim"] is None:
            current["latent_dim"] = int(m_latent.group(1))

        # Extract num_blocks_ssf
        m_blocks = blocks_re.search(line)
        if m_blocks and current["num_blocks_ssf"] is None:
            current["num_blocks_ssf"] = int(m_blocks.group(1))

        # Extract validation metrics
        m_lp = lp_re.search(line)
        if m_lp:
            val_loss = float(m_lp.group(2))
            val_acc = float(m_lp.group(3))
            current["best_val_acc"] = max(current["best_val_acc"], val_acc)
            current["best_val_loss"] = min(current["best_val_loss"], val_loss)

        # Extract test metrics
        m_test = test_re.search(line)
        if m_test:
            current["test_loss"] = float(m_test.group(1))
            current["test_acc"] = float(m_test.group(2))

# Append last run
if current is not None:
    runs.append(current)

df = pd.DataFrame(runs)
df = df.dropna(subset=["latent_dim", "num_blocks_ssf"])

# =========================================================
# 📊 Covariance & Correlation
# =========================================================

cov_matrix = df[["latent_dim","num_blocks_ssf","best_val_acc","best_val_loss"]].cov()
corr_matrix = df[["latent_dim","num_blocks_ssf","best_val_acc","best_val_loss"]].corr()

print("\nCovariance Matrix:\n")
print(cov_matrix)

print("\nPearson Correlation Matrix:\n")
print(corr_matrix)

# =========================================================
# 📈 Scatter Plots
# =========================================================

# 1️⃣ Latent dim vs best val acc
plt.figure()
plt.scatter(df["latent_dim"], df["best_val_acc"])
plt.xlabel("latent_dim")
plt.ylabel("best_val_acc")
plt.title("Latent Dimension vs Best Validation Accuracy")
plt.show()

# 2️⃣ Latent dim vs best val loss
plt.figure()
plt.scatter(df["latent_dim"], df["best_val_loss"])
plt.xlabel("latent_dim")
plt.ylabel("best_val_loss")
plt.title("Latent Dimension vs Best Validation Loss")
plt.show()

# 3️⃣ num_blocks_ssf vs best val acc
plt.figure()
plt.scatter(df["num_blocks_ssf"], df["best_val_acc"])
plt.xlabel("num_blocks_ssf")
plt.ylabel("best_val_acc")
plt.title("num_blocks_ssf vs Best Validation Accuracy")
plt.show()

# 4️⃣ num_blocks_ssf vs best val loss
plt.figure()
plt.scatter(df["num_blocks_ssf"], df["best_val_loss"])
plt.xlabel("num_blocks_ssf")
plt.ylabel("best_val_loss")
plt.title("num_blocks_ssf vs Best Validation Loss")
plt.show()