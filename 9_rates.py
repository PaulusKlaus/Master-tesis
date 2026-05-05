import re
import ast
import pandas as pd


paths_norm =["checkpoint/SSF_PU_0415-141949/training.log", # aug pu 

       "checkpoint/SSF_PU_0422-145545/training.log",]

paths_bz =["checkpoint/SSF_PU_0416-122923/training.log",
        "checkpoint/SSF_PU_0416-163602/training.log" # 128 batch size]
]

paths_CB =["checkpoint/SSF_PU_0420-145721/training.log", #6-10 blocks
        "checkpoint/SSF_PU_0421-085011/training.log", #1-5 blocks 

        # Another hidden layer was used for this 256 was changed to  128
        "checkpoint/SSF_PU_0423-152012/training.log", # Pu with no normalization for 5-9 blocks 
        "checkpoint/SSF_PU_0427-140057/training.log",  # No normalization pu 3- 4 blocks
]        

paths_hl =[
        "checkpoint/SSF_PU_0421-150551/training.log", #Latent space 256 meanstd 8 blocks 
        "checkpoint/SSF_PU_0429-112909/training.log",  # latent space 256 no normalization 8 blocks 
]

paths_ls=[# # hidden channel 160 
      "checkpoint/SSF_PU_0423-100558/training.log",  # hidden channel 128 

        # No normalization testing (hidden layer 160)
        "checkpoint/SSF_PU_0428-154042/training.log",]


paths_au =["checkpoint/SSF_PU_0427-144302/training.log", # no normalization, all augmentations 3 times  block size 7 

       "checkpoint/SSF_PU_0428-101427/training.log",  #  no normalization, different blck sizes  5,6,8,9
                # Another hidden layer was used for this 256 was changed to  128
        "checkpoint/SSF_PU_0423-152012/training.log", # Pu with no normalization for 5-9 blocks 
        "checkpoint/SSF_PU_0427-140057/training.log",  # No normalization pu 3- 4 blocks

       #"checkpoint/SSF_CWRU_0428-084416/training.log", # No normalization 
       #"checkpoint/SSF_CWRU_0428-094703/training.log"   # mean-std normalization 

        ]

paths_split =["checkpoint/SSF_PU_0501-150014/training.log",# Ordered Split 
    "checkpoint/SSF_PU_0504-102859/training.log",  # Only normal split 
    "checkpoint/SSF_PU_0501-130723/training.log", # Random Split 
    ]
rows = []

for path in paths_split:
    current_samples = None

    with open(path, "r") as f:
        for line in f:
            m = re.search(r"classifier_samples:\s*(\d+)", line)
            if m:
                current_samples = int(m.group(1))

            m = re.search(r"Rates:\s*(\{.*\})", line)
            if m:
                rates = ast.literal_eval(m.group(1))

                row = {
                    "file": path,
                    "classifier_samples": current_samples,
                }

                for cls, rate in rates.items():
                    row[f"class_{cls}"] = rate * 100

                rows.append(row)

df = pd.DataFrame(rows)

rate_cols = [c for c in df.columns if c.startswith("class_")]
# Summary per split and classifier_samples
summary = (
    df
    .groupby("classifier_samples")[rate_cols]
    .agg(["mean", "std"])
)

print(summary)
print(summary.to_latex(float_format="%.2f"))