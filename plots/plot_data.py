# Description: This file contains the data used to plot the results of the experiments.

# # Baseline comparison
# Params (in Million)
glue_pt_params = {
	5: 0.01,
	10: 0.01,
	25: 0.03,
	50: 0.05,
	100: 0.10,
}

crux_pt_params = {
	5: 0.01,
	10: 0.02,
	25: 0.05,
	50: 0.10,
	100: 0.20,
}

glue_idpg_params = {
	5: 1.58,
	10: 2.89,
	25: 6.84,
	50: 13.42,
	100: 26.58,
}

crux_idpg_params = {
	5: 8.47,
	10: 16.34,
	25: 39.96,
	50: 79.34,
	100: 158.08,
}

glue_ours_r4_params = {
	5: 1.59,
	10: 1.60,
	25: 1.63,
	50: 1.68,
	100: 1.78,
}

glue_ours_t10_params = {
	1: 0.80,
	2: 1.07,
	4: 1.60,
}

# CodeBERT
# crux_ours_ds_t10_params = {
# 	1: 2.78,
# 	2: 4.37,
# 	4: 7.53,
# }

# CodeSage
crux_ours_ds_t10_params = {
	1: 4.23,
	2: 6.34,
	4: 10.56,
}

# CodeSage
crux_ours_phi2_t10_params = {
	1: 4.76,
	2: 7.39,
	4: 12.66,
}

crux_ours_ds_t50_params = {
	1: 2.90,
	2: 4.51,
	4: 7.74,
}

crux_ours_r1_params = {
	5: 2.77,
	10: 2.78,
	25: 2.83,
	50: 2.90,
	100: 3.04,
}

crux_ours_r2_params = {
	5: 4.35,
	10: 4.37,
	25: 4.42,
	50: 4.51,
	100: 4.69,
}

crux_ours_r4_params = {
	5: 7.51,
	10: 7.53,
	25: 7.61,
	50: 7.74,
	100: 7.99
}

# MRPC
lora_on_mrpc = 90.77

idpg_on_mrpc = {
	5: 77.98,
	10: 78.60,
	25: 79.04,
	50: 76.54,
	100: 79.94
}

pt_on_mrpc = {
	5: 72.73,
	10: 72.38,
	25: 71.55,
	50: 74.57,
	100: 74.75
}

ours_on_mrpc = {
	5: 91.28,
	10: 91.09,
	25: 91.11,
	50: 90.17,
	100: 90.93
}

# RTE
lora_on_rte = 85.66

idpg_on_rte = {
	5: 74.73,
	10: 77.26,
	25: 76.90,
	50: 75.09,
	100: 76.17
}

pt_on_rte = {
	5: 54.87,
	10: 53.07,
	25: 53.79,
	50: 53.43,
	100: 56.68
}

ours_on_rte = {
	5: 84.48,
	10: 83.39,
	25: 83.75,
	50: 81.23,
	100: 83.75
}

# SST2
lora_on_sst2 = 96.22

idpg_on_sst2 = {
	5: 95.76,
	10: 95.30,
	25: 96.10,
	50: 95.76,
	100: 95.87
}


pt_on_sst2 = {
	5: 80.62,
	10: 84.40,
	25: 91.63,
	50: 91.28,
	100: 90.37
}

ours_on_sst2 = {
	5: 95.64,
	10: 95.99,
	25: 95.76,
	50: 95.76,
	100: 96.33
}

# CruxEval-O (Deepseek Coder 1.3B)
lora_on_cruxevalO = 36.0

idpg_on_cruxevalO = {
	5: 27.3,
	10: 28.5,
	25: 29.8,
	50: 27.0,
	100: 11.5
}

pt_on_cruxevalO = {
	5: 32.5,
	10: 31.2,
	25: 34.5,
	50: 33.5,
	100: 34.0
}

# With CodeBERT encoder
ours_r1_on_cruxevalO = {
	5: 34.0,
	10: 34.5,
	25: 32.5,
	50: 34.8,
	100: 33.5
}

# With CodeBERT encoder
ours_r2_on_cruxevalO = {
	5: 33.0,
	10: 30.2,
	25: 36.0,
	50: 32.8,
	100: 31.8
}

# # Ablation on rank
# MRPC
rank_on_mrpc = {
	1: 83.44,
	2: 91.15,
	4: 91.09,
}

# RTE
rank_on_rte = {
	1: 79.78,
	2: 80.87,
	4: 83.39,
}

# SST2
rank_on_sst2 = {
	1: 95.18,
	2: 95.64,
	4: 95.99,
}

# CruxEval-I-10T
rank_on_cruxevalI_w_DS_10t = {
	1: 43.0,
	2: 41.5,
	4: 40.0,
}

# CruxEval-O-10T
rank_on_cruxevalO_w_DS_10t = {
	1: 34.5,
	2: 35.0,
	4: 33.2,
}

# CruxEval-I-10T
rank_on_cruxevalI_w_phi2_10t = {
	1: 43.0,
	2: 43.0,
	4: 39.5,
}

# CruxEval-O-10T
rank_on_cruxevalO_w_phi2_10t = {
	1: 37.2,
	2: 38.2,
	4: 35.0,
}

# CruxEval-I-50T
rank_on_cruxevalI_w_DS_50t = {
	1: 41.8,
	2: 40.8,
	4: 40.0,
}

# CruxEval-O-50T
rank_on_cruxevalO_w_DS_50t = {
	1: 34.8,
	2: 32.8,
	4: 31.8,
}

# MBPP-10T