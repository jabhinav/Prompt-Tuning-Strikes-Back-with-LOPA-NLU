#!/bin/bash

#tasks=("mnli" "qqp" "rte" "sst2" "mrpc" "qnli")  # Add your list of task names here. Full list
tasks=("mnli" "qqp" "qnli")  # Smaller dataset size for faster testing
peft="idpg"
num_virtual_tokens=(10)  # Add your list of num_virtual_tokens here
phm_n=(8 16 32)
for task in "${tasks[@]}"; do
    # Iterate over each combination
    for n in "${phm_n[@]}"; do
          # # Create a unique directory name
          log_dir="logging/GLUE_${task}_${peft}_phm_n${n}"

          accelerate launch --config_file config_files/config_basic_nofp16.yaml tune_foundation_model.py --peft_method "${peft}" --dataset_name "$task" --log_dir "$log_dir" --phm_n "${n}" --num_virtual_tokens 10 --wandb_logging

#          # Remove the log file
#          rm -r "$log_dir"

    done
done
