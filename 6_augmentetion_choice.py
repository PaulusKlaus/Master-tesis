

from visualisation_training import aug_pair_vs_blocks_accuracy, augmentation_test, scatter_plots


paths =["checkpoint/SSF_PU_0427-144302/training.log", # no normalization, all augmentations 3 times  block size 7 

       "checkpoint/SSF_PU_0428-101427/training.log",  #  no normalization, different blck sizes  5,6,8,9
                # Another hidden layer was used for this 256 was changed to  128
        "checkpoint/SSF_PU_0423-152012/training.log", # Pu with no normalization for 5-9 blocks 
        "checkpoint/SSF_PU_0427-140057/training.log",  # No normalization pu 3- 4 blocks

       #"checkpoint/SSF_CWRU_0428-084416/training.log", # No normalization 
       #"checkpoint/SSF_CWRU_0428-094703/training.log"   # mean-std normalization 

        ]
augmentation_test(paths)

#plotting 
aug_pair_vs_blocks_accuracy(paths,top_k=5)

scatter_plots(paths)




