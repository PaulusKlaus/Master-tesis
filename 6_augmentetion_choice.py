

from visualisation_training import aug_pair_vs_blocks_accuracy, augmentation_test


paths =["checkpoint/SSF_PU_0427-144302/training.log", # no normalization, all augmentations 3 times 

       #"checkpoint/SSF_PU_0422-095246/training.log",


       "checkpoint/SSF_CWRU_0428-084416/training.log", # No normalization 
      # "checkpoint/SSF_CWRU_0428-094703/training.log"   # mean-std normalization 

        ]
augmentation_test(paths)

#plotting 
#aug_pair_vs_blocks_accuracy(paths)




