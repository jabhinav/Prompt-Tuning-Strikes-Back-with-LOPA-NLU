#!/bin/bash

tasks=("mnli" "qqp")  # Add your list of task names here
num_virtual_tokens=(10)  # Add your list of num_virtual_tokens here

for task in "${tasks[@]}"; do
    # Iterate over each combination
    for t_num in "${num_virtual_tokens[@]}"; do
          # # Create a unique directory name
          log_dir="logging/${task}_t${t_num}"

          accelerate launch --config_file config_basic_nofp16.yaml tune_pt_baseline.py --dataset_name "$task" --num_virtual_tokens "$t_num" --run_name "GLUE_${task}_PT_t${t_num}" --log_dir "$log_dir"

#          # Remove the log file
#          rm -r "$log_dir"
#
#          accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml tune_idpg_baseline.py --dataset_name "$task" --num_virtual_tokens "$t_num" --run_name "GLUE_${task}_IDPG_t${t_num}" --log_dir "$log_dir"
#
#          # Remove the log file
#          rm -r "$log_dir"

    done
done
