from visualisation_training import aug_pair_vs_blocks_accuracy, augmentation_test, scatter_plots

from utils.log_parser import summarize

def latent_space_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "latent_dim", group_by_blocks)
    return summary_batch_size


paths =[
    # Only normal split 
   # "checkpoint/SSF_PU_0504_102859/training.log", 
    # Random Split 
    #"checkpoint/SSF_PU_0501_130723/training.log", 

    # Ordered Split 
    "checkpoint/SSF_PU_0501_150014/training.log", 

        ]
#augmentation_test(paths)

#plotting 
#aug_pair_vs_blocks_accuracy(paths,top_k=5)

#scatter_plots(paths)

latent_space_test(paths)