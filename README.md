# Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation

This repository contains the official implementation of the paper titled "Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation" accepted at NIPS 24.

Here, the code for the NLU tasks are provided, rest of the instructions are same as the original repository.

## Training the Model

> To tune the model, you need to run the `tune_foundation_model.py` script. 
  
### Arguments

- `peft_method`: Specifies the PEFT method to be used.
  - Possible values: `lora`, `pt`, `idpg`, `lopa`, `fft`, `dept`, `prefix`, `ptuningv2`

- `dataset_name`: Name of the task to be trained on.
  - Possible values: `mnli`, `sst2`, `qqp`, `qnli`, `rte`, `mrpc`

- `dynamic_pad`: Whether to use dynamic padding or not.
  - Possible values: `True`, `False`
  - Default: `True`
  - Note: `dept` method does not support dynamic padding. Set this to `False` when using `dept`

## Evaluating and Getting Results

> Online: Evaluation on validation set is done while training the model. The results are logged.

> Offline: To evaluate the model on the test set, you need to run the `eval.py` script. 
### Offline Arguments
- `eval_data_type`: Specifies the data type to be evaluated on.
  - Possible values: `dev`, `test`
  - Default: `dev`
  - Note: On `test` we currently do not support computing evaluation metrics
- `do_evaluation`: Whether to evaluate the model or not.
  - Possible values: `True`, `False`
- `do_prediction`: Whether to predict the label using model or not.
  - Possible values: `True`, `False`
  - Note: The output is saved as `predictions.txt` in the logs directory
- `force_prediction_in_valid_labels`: Whether to force the prediction in valid labels set or not.
  - Possible values: `True`, `False`
  - Default: `True`

## Citation

```bibtex
@article{jain2024prompt,
  title={Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation},
  author={Jain, Abhinav and Chaudhuri, Swarat and Reps, Thomas and Jermaine, Chris},
  journal={arXiv preprint arXiv:2405.15282},
  year={2024}
}
```