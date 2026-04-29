from utils.log_parser import summarize
from visualisation_training import aug_pair_vs_blocks_accuracy, augmentation_test, scatter_plots



def block_size_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "num_blocks_ssf", group_by_blocks)
    return summary_batch_size



paths =[#"checkpoint/SSF_PU_0420-145721/training.log", #6-10 blocks
        #"checkpoint/SSF_PU_0421-085011/training.log", #1-5 blocks 

        # Another hidden layer was used for this 256 was changed to  128
        "checkpoint/SSF_PU_0423-152012/training.log", # Pu with no normalization for 5-9 blocks 
        "checkpoint/SSF_PU_0427-140057/training.log",  # No normalization pu 3- 4 blocks
        

        

      # "checkpoint/SSF_CWRU_0421-130608/training.log",
        ]
block_size_test(paths)

aug_pair_vs_blocks_accuracy(paths, top_k = 5 )

scatter_plots(paths)
