from visualisation_training import aug_pair_vs_blocks_accuracy, augmentation_test, scatter_plots
from utils.log_parser import summarize

def classifier_for_CNN_space_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "classifier_sample", group_by_blocks)
    return summary_batch_size

def classifier_for_SSF_space_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "class_sample", group_by_blocks)
    return summary_batch_size

paths_CNN =["checkpoint/CNN_1d_PU_0504-152809/training.log", 

    ]

paths_SSF =["checkpoint/SSF_PU_0504-150630/training.log",

    ]
#augmentation_test(paths)
classifier_for_CNN_space_test(paths_CNN) # Look at the test accuracies 
classifier_for_SSF_space_test(paths_SSF) # look at the f1-score 