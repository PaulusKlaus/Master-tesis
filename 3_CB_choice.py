from utils.log_parser import summarize
from visualisation_training import aug_pair_vs_blocks_accuracy, augmentation_test, scatter_plots



def block_size_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "num_blocks_ssf", group_by_blocks)
    return summary_batch_size



paths =[#"checkpoint/SSF_PU_0420-145721/training.log", #6-10 blocks
        #"checkpoint/SSF_PU_0421-085011/training.log", #1-5 blocks 

        # Another hidden layer was used for this 256 was changed to  128 -> another augmentation pairs were used as well, resulting in a higher standard deviation 
       # "checkpoint/SSF_PU_0423-152012/training.log", # Pu with no normalization for 5-9 blocks 
        #"checkpoint/SSF_PU_0427-140057/training.log",  # No normalization pu 3- 4 blocks

        # PU no normalization 
        "checkpoint/SSF_PU_0518-110247/training.log",

        

       #"checkpoint/SSF_CWRU_0421-130608/training.log",
        ]
block_size_test(paths)
augmentation_test(paths)
#aug_pair_vs_blocks_accuracy(paths, top_k = 5 )

scatter_plots(paths)




from pathlib import Path
import re
import matplotlib.pyplot as plt

import pandas as pd

log_path = Path("checkpoint/SSF_PU_0427-144302/training.log")
text = log_path.read_text(errors="ignore")

# Try multiple regex patterns for threshold and binary accuracy
patterns = [
    r"Threshold[:=]\s*([0-9]*\.?[0-9]+).*?Binary accuracy[:=]\s*([0-9]*\.?[0-9]+)",
    r"threshold[:=]\s*([0-9]*\.?[0-9]+).*?binary accuracy[:=]\s*([0-9]*\.?[0-9]+)",
]

matches = []
for pattern in patterns:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        break

if not matches:
    # fallback: extract separately and pair by order
    thresholds = re.findall(r"Threshold[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    accuracies = re.findall(r"Binary accuracy[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    matches = list(zip(thresholds, accuracies))

if not matches:
    raise ValueError("Could not find Threshold and Binary accuracy values in the uploaded log.")

# Convert to dataframe
df = pd.DataFrame(matches, columns=["Threshold", "Binary Accuracy"])
df["Threshold"] = df["Threshold"].astype(float)
df["Binary Accuracy"] = df["Binary Accuracy"].astype(float)

# Remove duplicates and sort
df = df.drop_duplicates().sort_values("Threshold")

# Create graph
plt.figure(figsize=(8, 5))
plt.scatter(df["Threshold"], df["Binary Accuracy"])
plt.xlabel("Threshold")
plt.ylabel("Binary Accuracy")
plt.title("Relationship Between Threshold and Binary Accuracy")
plt.grid(True)

plot_path = "figures/threshold_binary_accuracy.pdf"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

