datasetname =["Normal Baseline Data", "12k Drive End Bearing Fault Data"]
normalname = ["0_N_0.mat", "0_N_1.mat", "0_N_2.mat", "0_N_3.mat"]

ir_faults = ['1_IR007_0.mat', '1_IR007_1.mat', '1_IR007_2.mat', '1_IR007_3.mat']
b_faults = ['2_B007_0.mat', '2_B007_1.mat', '2_B007_2.mat', '2_B007_3.mat']
or_faults = ['3_OR007_6_0.mat', '3_OR007_6_1.mat', '3_OR007_6_2.mat', '3_OR007_6_3.mat']

load_0 = ['1_IR007_0.mat', '2_B007_0.mat', '3_OR007_6_0.mat', ]
load_1 = ['1_IR007_1.mat', '2_B007_1.mat', '3_OR007_6_1.mat', ]
load_2 = ['1_IR007_2.mat', '2_B007_2.mat', '3_OR007_6_2.mat', ]
load_3 = ['1_IR007_3.mat', '2_B007_3.mat', '3_OR007_6_3.mat', ]

load_ALL = load_0 + load_1 + load_2 + load_3
print(load_ALL)