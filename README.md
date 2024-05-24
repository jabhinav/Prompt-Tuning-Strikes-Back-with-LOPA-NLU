# Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation

This repository contains the official implementation of the paper titled "Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation"
Here, the code for the NLU tasks are provided, rest of the instructions are same as the original repository.

## Training the Model

> To tune the model, you need to run the `tune_foundation_model.py` script. 
  
### Arguments

- `--peft_method`: Specifies the PEFT method to be used.
  - Possible values: `lora`, `pt`, `idpg`, `lopa`, `fft`

- `--dataset_name`: Name of the task to be trained on.
  - Possible values: `mnli`, `sst-2`, `qqp`, `qnli`, `rte`, `mrpc`

## Evaluating and Getting Results

> Evaluation is done while training the model. The results are logged.