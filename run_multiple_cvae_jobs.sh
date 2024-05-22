## # Rank experiments

# Debug
accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name rte --run_name debug_GLUE_rte_CVAE_r4_t10 --log_dir logging/debug_GLUE_rte_CVAE_r4_t10
accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name mrpc --run_name debug_GLUE_mrpc_CVAE_r4_t10 --log_dir logging/debug_GLUE_mrpc_CVAE_r4_t10


## RTE
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name rte --lp_rank 1 --run_name GLUE_rte_CVAE_r1 --log_dir logging/GLUE_rte_CVAE_r1_t10
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name rte --lp_rank 2 --run_name GLUE_rte_CVAE_r2 --log_dir logging/GLUE_rte_CVAE_r2_t10
#
## SST-2
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name sst2 --lp_rank 1 --run_name GLUE_sst2_CVAE_r1 --log_dir logging/GLUE_sst2_CVAE_r1_t10
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name sst2 --lp_rank 2 --run_name GLUE_sst2_CVAE_r2 --log_dir logging/GLUE_sst2_CVAE_r2_t10
#
## # Virtual tokens experiments
#
## MRPC
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name mrpc --num_virtual_tokens 5 --run_name GLUE_mrpc_CVAE_t5 --log_dir logging/GLUE_mrpc_CVAE_r4_t5
#rm -r logging/GLUE_mrpc_CVAE_r4_t5
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name mrpc --num_virtual_tokens 25 --run_name GLUE_mrpc_CVAE_t25 --log_dir logging/GLUE_mrpc_CVAE_r4_t25
#rm -r logging/GLUE_mrpc_CVAE_r4_t25
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name mrpc --num_virtual_tokens 50 --run_name GLUE_mrpc_CVAE_t50 --log_dir logging/GLUE_mrpc_CVAE_r4_t50
#rm -r logging/GLUE_mrpc_CVAE_r4_t50
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name mrpc --num_virtual_tokens 100 --run_name GLUE_mrpc_CVAE_t100 --log_dir logging/GLUE_mrpc_CVAE_r4_t100
#rm -r logging/GLUE_mrpc_CVAE_r4_t100
#
## RTE

#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name rte --num_virtual_tokens 25 --run_name GLUE_rte_CVAE_t25 --log_dir logging/GLUE_rte_CVAE_r4_t25
#rm -r logging/GLUE_rte_CVAE_r4_t25
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name rte --num_virtual_tokens 50 --run_name GLUE_rte_CVAE_t50 --log_dir logging/GLUE_rte_CVAE_r4_t50
#rm -r logging/GLUE_rte_CVAE_r4_t50
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name rte --num_virtual_tokens 100 --run_name GLUE_rte_CVAE_t100 --log_dir logging/GLUE_rte_CVAE_r4_t100
#rm -r logging/GLUE_rte_CVAE_r4_t100
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name rte --num_virtual_tokens 5 --run_name GLUE_rte_CVAE_t5 --log_dir logging/GLUE_rte_CVAE_r4_t5
#rm -r logging/GLUE_rte_CVAE_r4_t5

## SST-2
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name sst2 --num_virtual_tokens 5 --run_name GLUE_sst2_CVAE_t5 --log_dir logging/GLUE_sst2_CVAE_r4_t5
#rm -r logging/GLUE_sst2_CVAE_r4_t5
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name sst2 --num_virtual_tokens 25 --run_name GLUE_sst2_CVAE_t25 --log_dir logging/GLUE_sst2_CVAE_r4_t25
#rm -r logging/GLUE_sst2_CVAE_r4_t25
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name sst2 --num_virtual_tokens 50 --run_name GLUE_sst2_CVAE_t50  --log_dir logging/GLUE_sst2_CVAE_r4_t50
#rm -r logging/GLUE_sst2_CVAE_r4_t50
#accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --dataset_name sst2 --num_virtual_tokens 100 --run_name GLUE_sst2_CVAE_t100 --log_dir logging/GLUE_sst2_CVAE_r4_t100
#rm -r logging/GLUE_sst2_CVAE_r4_t100